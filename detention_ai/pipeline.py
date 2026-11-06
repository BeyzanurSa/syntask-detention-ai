"""feed -> extract (LLM) -> resolve dates + merge (code) -> calculate (code) -> alerts (code)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .alerts import alert
from .calculator import ContainerResult, evaluate
from .extract import ExtractionResult, Extractor, extract
from .merge import merge
from .messages import Message


@dataclass
class Report:
    as_of: date
    results: list[ContainerResult]
    extractions: list[ExtractionResult]
    alerts: dict[str, str]

    @property
    def ignored(self) -> list[str]:
        return [e.message.id for e in self.extractions if e.status == "ignored"]

    @property
    def needs_review(self) -> list[str]:
        return [e.message.id for e in self.extractions if e.status == "needs_review"]


def run(messages: list[Message], extractor: Extractor, as_of: date | None = None,
        on_message=None) -> Report:
    extractions = []
    for m in messages:
        e = extract(m, extractor)
        extractions.append(e)
        if on_message:
            on_message(e)
    as_of = as_of or max(m.date for m in messages)  # default: latest message date
    facts = merge([(e.message, e.mentions) for e in extractions if e.status == "ok"])
    results = [evaluate(f, as_of) for f in facts.values()]
    alerts = {r.konteyner_no: a for r in results if (a := alert(r))}
    return Report(as_of, results, extractions, alerts)
