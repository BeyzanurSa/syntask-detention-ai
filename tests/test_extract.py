"""extract.py with scripted model outputs: schema validation, retry, grounding,
evidence checks and plausibility. No model needed.

The M3/M7/M9/M10/M11 cases replay the field mix-ups the real Qwen3-4B made with
the first schema version; the checks must neutralise each one."""

import json
from pathlib import Path

import pytest

from detention_ai.extract import extract
from detention_ai.messages import load_feed
from detention_ai.schema import llm_json_schema

FEED = {m.id: m for m in load_feed(Path(__file__).parent.parent / "data/messages.txt")}
GOLDEN = json.loads((Path(__file__).parent / "fixtures/golden_extractions.json").read_text())


class Scripted:
    def __init__(self, *outputs):
        self.outputs, self.calls = [o if isinstance(o, str) else json.dumps(o) for o in outputs], 0

    def raw_extract(self, message):
        self.calls += 1
        return self.outputs.pop(0)


def golden(mid, **override):
    c = {**GOLDEN[mid]["containers"][0], **override}
    return {"containers": [c]}


def run(mid, output):
    return extract(FEED[mid], Scripted(output))


def date(evidence, raw):
    return {"evidence": evidence, "raw": raw}


# --- golden outputs pass untouched -------------------------------------------

@pytest.mark.parametrize("mid", [f"M{i}" for i in range(1, 13)])
def test_golden_extraction_passes_all_checks(mid):
    r = run(mid, GOLDEN[mid])
    assert r.status == "ok" and r.warnings == []
    assert r.extraction.model_dump() == GOLDEN[mid]


# --- M3: gate-out copied into planned_return ---------------------------------

def test_m3_gate_out_as_return_is_dropped():
    r = run("M3", golden("M3", planned_return=date("gate-out 30/09/2026", "30/09/2026")))
    m = r.mentions[0]
    assert m.planned_return is None and m.gate_out.raw == "30/09/2026"
    assert any("iade ifadesi yok" in w for w in r.warnings)


def test_m3_return_without_evidence_is_dropped():
    r = run("M3", golden("M3", planned_return=date("iade 30/09/2026", "30/09/2026")))
    assert r.mentions[0].planned_return is None
    assert any("metinde yok" in w for w in r.warnings)


def test_m3_tiers_kept():
    tiers = run("M3", GOLDEN["M3"]).mentions[0].rates.tiers
    assert [(t.usd_per_day, t.first_n_days) for t in tiers] == [(120, 3), (200, None)]


# --- M7: "still at terminal" invented ----------------------------------------

def test_m7_invented_terminal_status_is_dropped():
    r = run("M7", golden("M7", at_terminal_evidence="şasisi arızalandı"))
    m = r.mentions[0]
    assert m.at_terminal_evidence is None and m.planned_return.raw == "12 Ekim"


def test_m7_terminal_quote_not_in_text_is_dropped():
    r = run("M7", golden("M7", at_terminal_evidence="hâlâ terminalde"))
    assert r.mentions[0].at_terminal_evidence is None


# --- M9: return weekday copied into gate_out ---------------------------------

def test_m9_return_weekday_as_gate_out_is_dropped():
    r = run("M9", golden("M9", gate_out=date("perşembe iade ediyoruz", "perşembe")))
    m = r.mentions[0]
    assert m.gate_out is None and m.planned_return.raw == "perşembe"
    assert any("gate-out ifadesi yok" in w for w in r.warnings)


# --- M10: return date copied into gate_out -----------------------------------

def test_m10_return_date_as_gate_out_is_dropped():
    r = run("M10", golden("M10", gate_out=date("9 Ekim Cuma bitecek", "9 Ekim Cuma")))
    m = r.mentions[0]
    assert m.gate_out is None and m.planned_return.raw == "9 Ekim Cuma"


def test_m10_bare_date_as_gate_out_is_dropped():
    r = run("M10", golden("M10", gate_out=date("9 Ekim Cuma", "9 Ekim Cuma")))
    assert r.mentions[0].gate_out is None


# --- M11: invented terminal status, vague return must survive ----------------

def test_m11_invented_terminal_status_dropped_vague_return_kept():
    r = run("M11", golden("M11", at_terminal_evidence="haftaya bir ara araç ayarlarız"))
    m = r.mentions[0]
    assert m.at_terminal_evidence is None
    assert m.planned_return.raw == "haftaya bir ara"


# --- M12: real terminal status is kept ----------------------------------------

def test_m12_terminal_status_kept():
    assert run("M12", GOLDEN["M12"]).mentions[0].at_terminal_evidence == "hâlâ terminalde"


# --- plausibility (second line of defence) -------------------------------------

def test_future_gate_out_dropped_even_with_gate_word():
    # a fake message where the gate-out wording is present but the date is after the message date
    from detention_ai.messages import Message
    m = Message(id="X1", seq=0, date=FEED["M9"].date, sender="x",
                text="MSCU 482915-3 gated out perşembe.")
    out = golden("M9", gate_out=date("gated out perşembe", "perşembe"), planned_return=None)
    r = extract(m, Scripted(out))
    assert r.mentions[0].gate_out is None
    assert any("mesaj tarihinden sonra" in w for w in r.warnings)


# --- free time / rates evidence ------------------------------------------------

def test_free_time_number_must_be_in_evidence():
    out = golden("M1", free_time={"evidence": "Free time: 5 calendar days", "days": 7,
                                  "basis": "calendar"})
    assert run("M1", out).mentions[0].free_time is None


def test_rate_number_must_be_in_evidence():
    out = golden("M1", rates={"evidence": "Detention: USD 150/day",
                              "tiers": [{"usd_per_day": 250, "first_n_days": None}]})
    assert run("M1", out).mentions[0].rates is None


# --- schema / retry / grounding -----------------------------------------------

def test_invalid_then_valid_retries_once():
    ex = Scripted("{not json", GOLDEN["M9"])
    assert extract(FEED["M9"], ex).status == "ok" and ex.calls == 2


def test_schema_violation_goes_to_manual_review():
    bad = golden("M1", free_time={"evidence": "x", "days": 5, "basis": "weekly"})
    r = extract(FEED["M1"], Scripted(bad, bad))
    assert r.status == "needs_review" and r.mentions == []


def test_extra_keys_rejected():
    bad = golden("M9", last_free_day="2026-10-03")
    assert extract(FEED["M9"], Scripted(bad, bad)).status == "needs_review"


def test_empty_containers_despite_number_goes_to_review():
    assert run("M9", {"containers": []}).status == "needs_review"


def test_no_container_number_skips_model():
    ex = Scripted()
    assert extract(FEED["M13"], ex).status == "ignored" and ex.calls == 0


def test_hallucinated_number_snapped_to_text():
    r = run("M9", golden("M9", container_no="MSCU4829154"))
    assert r.mentions[0].container_no == "MSCU4829153"
    assert any("değiştirildi" in w for w in r.warnings)


def test_llm_schema_has_no_refs_and_evidence_before_raw():
    schema = llm_json_schema()
    assert "$ref" not in json.dumps(schema)
    gate = schema["properties"]["containers"]["items"]["properties"]["gate_out"]["anyOf"][0]
    assert list(gate["properties"]) == ["evidence", "raw"]
