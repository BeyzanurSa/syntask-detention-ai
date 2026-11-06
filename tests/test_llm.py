"""Real model, real messages. Slow (model load + 12 calls, ~1 min). Skipped if the
GGUF file or llama_cpp is missing. Run: pytest -m llm

These check the model's RAW output (before evidence/plausibility checks) for the
messages where the first schema version mixed up fields."""

import json
from pathlib import Path

import pytest

from detention_ai.messages import load_feed
from detention_ai.pipeline import run

from .test_case_expected import FEED, assert_case_results

MODEL = Path(__file__).parent.parent / "models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(not MODEL.exists(), reason="model not downloaded"),
]
MSGS = {m.id: m for m in load_feed(FEED)}


@pytest.fixture(scope="module")
def llm():
    pytest.importorskip("llama_cpp")
    from detention_ai.extract import LlamaExtractor

    return LlamaExtractor(MODEL)


def raw(llm, mid):
    return json.loads(llm.raw_extract(MSGS[mid]))["containers"][0]


def test_m3_gate_out_only_no_return(llm):
    c = raw(llm, "M3")
    assert c["gate_out"]["raw"] == "30/09/2026" and "gate" in c["gate_out"]["evidence"].lower()
    assert c["planned_return"] is None and c["at_terminal_evidence"] is None
    assert [(t["usd_per_day"], t["first_n_days"]) for t in c["rates"]["tiers"]] == [(120, 3), (200, None)]


def test_m7_return_only_no_terminal(llm):
    c = raw(llm, "M7")
    assert c["planned_return"]["raw"] == "12 Ekim"
    assert c["gate_out"] is None and c["at_terminal_evidence"] is None


def test_m9_persembe_is_return_not_gate_out(llm):
    c = raw(llm, "M9")
    assert c["planned_return"]["raw"].lower() == "perşembe"
    assert c["gate_out"] is None


def test_m10_same_day_return(llm):
    c = raw(llm, "M10")
    assert c["planned_return"]["raw"].startswith("9 Ekim")
    assert c["gate_out"] is None and c["at_terminal_evidence"] is None


def test_m11_vague_return_kept_raw_no_terminal(llm):
    c = raw(llm, "M11")
    assert c["planned_return"]["raw"] == "haftaya bir ara"
    assert c["at_terminal_evidence"] is None


def test_m12_terminal_evidence(llm):
    assert "terminal" in raw(llm, "M12")["at_terminal_evidence"].lower()


def test_real_model_reaches_case_results(llm):
    report = run(load_feed(FEED), llm)
    assert report.needs_review == []
    assert_case_results(report)
