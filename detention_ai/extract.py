"""Message -> MessageExtraction.

Layers that make the output trustworthy:
1. Code pre-filter: a message with no container number (regex) never reaches the
   model; it is ignored as unrelated (M13).
2. Grammar: llama.cpp turns the JSON schema into a GBNF grammar, so the model can
   only emit tokens that form schema-shaped JSON (syntax and types).
3. Pydantic validates the result (ranges, enums, extra keys). On failure: one
   retry, then the message goes to manual review instead of guessing.
4. Container grounding: the model's number must match a regex hit in the text.
5. Evidence checks (deterministic): every critical field quotes the message.
   The quote must be in the text, the date/number must be inside the quote,
   and the quote must contain the right kind of word (gate-out words for
   gate_out, return words for planned_return, terminal words for at_terminal).
   A failing field is dropped ("not mentioned") so it cannot override earlier,
   correct information in the merge.
6. Plausibility: gate-out after the message date, or a return on/before the
   gate-out of the same message -> field dropped.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError

from . import container
from .dates import resolve
from .messages import Message
from .schema import ContainerMention, MessageExtraction, llm_json_schema

# Vocabulary for evidence checks. Matched on the normalized evidence quote.
GATE_OUT_WORDS = ("gate", "released", "picked up", "pick-up", "pickup",
                  "terminalden çık", "terminalden çek", "limandan çık", "çekildi")
RETURN_WORDS = ("iade", "return", "geri getir", "geri bırak", "drop off", "drop-off",
                "araç ayarla", "araç gönder")  # "araç ayarlarız" = arranging the return truck
TERMINAL_WORDS = ("terminal", "liman", "port", "gümrük", "customs", "not yet gated")
FREE_WORDS = ("free", "serbest")
RATE_WORDS = ("usd", "$", "detention", "/day", "per day", "günlük")

SYSTEM_PROMPT = """You extract facts from container shipping operations messages (English or Turkish).
Answer with JSON only. One entry in "containers" per container number in the message.
For every field you fill, first copy the supporting words from the message into "evidence" (verbatim), then fill the value from that evidence.

Rules:
- container_no: copy exactly as written (keep spaces and dashes).
- Dates: copy the date expression into "raw" EXACTLY as written ("28 Sep 2026", "01 Oct", "6 Ekim", "perşembe", "haftaya bir ara"). Never convert, calculate or guess a date.
- gate_out: ONLY when the message explicitly says the container left the terminal: "gated out", "gate-out", "released", "picked up". A planned return date is NEVER a gate-out. Otherwise null.
- planned_return: ONLY from return wording ("iade", "return", arranging a truck for the return). Copy vague expressions as written ("haftaya bir ara"). If the return is "the same day" as another date, raw = that date expression. A gate-out date is NEVER a return. Otherwise null.
- A single date expression belongs to at most one of gate_out / planned_return.
- at_terminal_evidence: quote ONLY if the message explicitly says the container is still at the terminal / port / customs. Unloading, breakdowns or truck plans are NOT terminal status. Otherwise null.
- free_time: only if free time is stated. basis "business" for business/working days, "calendar" if calendar days are said, "unspecified" if only "days".
- rates: detention rates in order. A flat rate ("USD 90/day") is ONE tier with first_n_days null. first_n_days only for "first N days"; "thereafter" has first_n_days null. null if no rate is stated.
- Never infer facts that are not written in the message. When unsure, use null.

Example 1
Message: ABCU1234567 gated out 14 Mar 2026. Free time 6 working days. Detention USD 90 per day.
JSON: {"containers": [{"container_no": "ABCU1234567", "gate_out": {"evidence": "gated out 14 Mar 2026", "raw": "14 Mar 2026"}, "at_terminal_evidence": null, "free_time": {"evidence": "Free time 6 working days", "days": 6, "basis": "business"}, "rates": {"evidence": "Detention USD 90 per day", "tiers": [{"usd_per_day": 90, "first_n_days": null}]}, "planned_return": null}]}

Example 2
Message: JKLU2223334 released 2 Apr. 3 days free. First 2 days USD 50/day, thereafter USD 80/day.
JSON: {"containers": [{"container_no": "JKLU2223334", "gate_out": {"evidence": "released 2 Apr", "raw": "2 Apr"}, "at_terminal_evidence": null, "free_time": {"evidence": "3 days free", "days": 3, "basis": "unspecified"}, "rates": {"evidence": "First 2 days USD 50/day, thereafter USD 80/day", "tiers": [{"usd_per_day": 50, "first_n_days": 2}, {"usd_per_day": 80, "first_n_days": null}]}, "planned_return": null}]}

Example 3
Message: XYZU 765432-1 boşaltıldı, salı günü iade edilecek.
JSON: {"containers": [{"container_no": "XYZU 765432-1", "gate_out": null, "at_terminal_evidence": null, "free_time": null, "rates": null, "planned_return": {"evidence": "salı günü iade edilecek", "raw": "salı"}}]}

Example 4
Message: DEFU5556667'nin lastiği patladı, iade gelecek ay gibi olur.
JSON: {"containers": [{"container_no": "DEFU5556667", "gate_out": null, "at_terminal_evidence": null, "free_time": null, "rates": null, "planned_return": {"evidence": "iade gelecek ay gibi olur", "raw": "gelecek ay gibi"}}]}

Example 5
Message: PQRU2223331 tamiri 14 Mart Salı tamamlanacak, aynı gün iade.
JSON: {"containers": [{"container_no": "PQRU2223331", "gate_out": null, "at_terminal_evidence": null, "free_time": null, "rates": null, "planned_return": {"evidence": "14 Mart Salı tamamlanacak, aynı gün iade", "raw": "14 Mart Salı"}}]}

Example 6
Message: GHIU7654321 limanda bekliyor, ordino çıkmadı.
JSON: {"containers": [{"container_no": "GHIU7654321", "gate_out": null, "at_terminal_evidence": "limanda bekliyor", "free_time": null, "rates": null, "planned_return": null}]}
"""


@dataclass
class ExtractionResult:
    message: Message
    status: str  # "ok" | "ignored" | "needs_review"
    extraction: MessageExtraction | None = None
    latency_s: float | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def mentions(self) -> list[ContainerMention]:
        return self.extraction.containers if self.extraction else []


class Extractor(Protocol):
    def raw_extract(self, message: Message) -> str: ...


def _norm(s: str) -> str:
    s = s.replace("İ", "i").replace("’", "'").lower()
    return re.sub(r"\s+", " ", s).strip(" .,;:!?")


def _has(text: str, words: tuple[str, ...]) -> bool:
    return any(w in text for w in words)


def _check_evidence(msg: Message, m: ContainerMention) -> tuple[dict, list[str]]:
    """Returns (fields to null out, warnings)."""
    text, drop, warnings = _norm(msg.text), {}, []

    def reject(name: str, why: str) -> None:
        drop[name] = None
        warnings.append(f"{msg.id}: {name} atıldı ({why})")

    def quote_ok(name: str, evidence: str) -> bool:
        if _norm(evidence) not in text:
            reject(name, f"evidence '{evidence}' metinde yok")
            return False
        return True

    for name, words, kind in (("gate_out", GATE_OUT_WORDS, "gate-out"),
                              ("planned_return", RETURN_WORDS, "iade")):
        d = getattr(m, name)
        if d is None or not quote_ok(name, d.evidence):
            continue
        ev = _norm(d.evidence)
        if _norm(d.raw) not in ev:
            reject(name, f"'{d.raw}' kendi evidence'ında geçmiyor")
        elif not _has(ev, words):
            reject(name, f"'{d.evidence}' içinde {kind} ifadesi yok")

    if m.at_terminal_evidence is not None and quote_ok("at_terminal_evidence", m.at_terminal_evidence):
        if not _has(_norm(m.at_terminal_evidence), TERMINAL_WORDS):
            reject("at_terminal_evidence", f"'{m.at_terminal_evidence}' terminal durumu belirtmiyor")

    if m.free_time is not None and quote_ok("free_time", m.free_time.evidence):
        ev = _norm(m.free_time.evidence)
        if not _has(ev, FREE_WORDS) or str(m.free_time.days) not in ev:
            reject("free_time", f"'{m.free_time.evidence}' serbest süreyi göstermiyor")

    if m.rates is not None and quote_ok("rates", m.rates.evidence):
        ev = _norm(m.rates.evidence)
        missing = [t.usd_per_day for t in m.rates.tiers if f"{t.usd_per_day:g}" not in ev]
        if not _has(ev, RATE_WORDS) or missing:
            reject("rates", f"'{m.rates.evidence}' ücreti göstermiyor")
    return drop, warnings


def _check_plausibility(msg: Message, m: ContainerMention) -> tuple[dict, list[str]]:
    drop, warnings = {}, []
    gate = resolve(m.gate_out.raw, msg.date) if m.gate_out else None
    ret = resolve(m.planned_return.raw, msg.date) if m.planned_return else None
    if gate and gate.value and gate.value > msg.date:
        drop["gate_out"], gate = None, None
        warnings.append(f"{msg.id}: gate_out atıldı ('{m.gate_out.raw}' mesaj tarihinden sonra)")
    if ret and gate and (_norm(m.planned_return.raw) == _norm(m.gate_out.raw)
                         or (ret.value and gate.value and ret.value <= gate.value)):
        drop["planned_return"] = None
        warnings.append(f"{msg.id}: planned_return atıldı (gate-out ile aynı veya önce)")
    return drop, warnings


def _ground(message: Message, ext: MessageExtraction) -> tuple[MessageExtraction, list[str]]:
    warnings: list[str] = []
    hits = container.find_all(message.text)
    kept: list[ContainerMention] = []
    for m in ext.containers:
        parsed = container.parse(m.container_no)
        if parsed is None or parsed.value not in {h.value for h in hits}:
            if len(hits) == 1 and len(ext.containers) == 1:
                warnings.append(f"{message.id}: model numarası '{m.container_no}' metindeki "
                                f"{hits[0].value} ile değiştirildi")
                m = m.model_copy(update={"container_no": hits[0].value})
            else:
                warnings.append(f"{message.id}: '{m.container_no}' metinde yok, atlandı")
                continue
        for check in (_check_evidence, _check_plausibility):
            drop, w = check(message, m)
            warnings += w
            if drop:
                m = m.model_copy(update=drop)
        kept.append(m)
    return ext.model_copy(update={"containers": kept}), warnings


def extract(message: Message, extractor: Extractor, retries: int = 1) -> ExtractionResult:
    if not container.find_all(message.text):
        return ExtractionResult(message, "ignored",
                                warnings=[f"{message.id}: konteyner numarası yok, yok sayıldı"])
    errors: list[str] = []
    for _ in range(retries + 1):
        t0 = time.perf_counter()
        raw = extractor.raw_extract(message)
        latency = time.perf_counter() - t0
        try:
            ext = MessageExtraction.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as e:
            errors.append(str(e).splitlines()[0])
            continue
        ext, warnings = _ground(message, ext)
        if not ext.containers:
            # regex saw a container number but the model returned nothing usable
            return ExtractionResult(message, "needs_review", ext, latency,
                                    warnings + [f"{message.id}: numara var ama model bilgi çıkaramadı"])
        return ExtractionResult(message, "ok", ext, latency, warnings)
    return ExtractionResult(message, "needs_review",
                            warnings=[f"{message.id}: şemaya uymayan çıktı ({errors[-1]})"])


class FixtureExtractor:
    """Replays recorded/golden JSON per message id. Used by tests and offline demo."""

    def __init__(self, path: str | Path):
        self.data = json.loads(Path(path).read_text(encoding="utf-8"))

    def raw_extract(self, message: Message) -> str:
        return json.dumps(self.data[message.id], ensure_ascii=False)


class LlamaExtractor:
    """Qwen3-4B (GGUF Q4_K_M) via llama-cpp-python, Metal offload, schema grammar."""

    def __init__(self, model_path: str | Path, n_ctx: int = 4096, verbose: bool = False):
        from llama_cpp import Llama  # imported lazily: tests don't need it

        t0 = time.perf_counter()
        self.llm = Llama(
            model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=-1,
            seed=0, chat_format="chatml", verbose=verbose,
        )
        self.load_s = time.perf_counter() - t0
        self.schema = llm_json_schema()
        self.last_usage: dict | None = None

    def raw_extract(self, message: Message, max_tokens: int = 768) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                # No message date on purpose: the model must not resolve dates.
                {"role": "user", "content": f"Message: {message.text} /no_think"},
            ],
            response_format={"type": "json_object", "schema": self.schema},
            temperature=0.0,  # reduces randomness; does not make the model correct
            max_tokens=max_tokens,
        )
        self.last_usage = out.get("usage")
        return out["choices"][0]["message"]["content"]
