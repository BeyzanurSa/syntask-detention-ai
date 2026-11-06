"""End-to-end on the case's 13 messages with golden extractions (no model needed).
The same assertions run against the real model in test_llm.py."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from detention_ai.calculator import Risk
from detention_ai.extract import FixtureExtractor
from detention_ai.messages import load_feed
from detention_ai.pipeline import run

ROOT = Path(__file__).parent.parent
FEED = ROOT / "data/messages.txt"

EXPECTED = {
    #  no              risk               fee    last free          return             kalan
    "MSCU4829153": (Risk.HIGH, Decimal(750), date(2026, 10, 3), date(2026, 10, 8), -2),
    "MAEU7731042": (Risk.MEDIUM, Decimal(200), date(2026, 10, 7), date(2026, 10, 9), 2),
    "HLCU5520187": (Risk.HIGH, Decimal(760), date(2026, 10, 7), date(2026, 10, 12), 2),
    "CMAU3301986": (Risk.UNKNOWN, None, date(2026, 10, 7), None, 2),
    "TGHU9004512": (Risk.LOW, Decimal(0), date(2026, 10, 13), date(2026, 10, 8), 8),
    "OOLU6612408": (Risk.OUT_OF_SCOPE, None, None, None, None),
}


def assert_case_results(report):
    assert report.as_of == date(2026, 10, 5)
    got = {r.konteyner_no: r for r in report.results}
    assert set(got) == set(EXPECTED)
    for no, (risk, fee, lfd, ret, kalan) in EXPECTED.items():
        r = got[no]
        assert (r.risk, r.tahmini_ucret_usd, r.son_serbest_gun, r.planlanan_iade, r.kalan_gun) \
            == (risk, fee, lfd, ret, kalan), no
    assert "M13" in report.ignored
    assert "M4" in got["HLCU5520187"].sources and "M7" in got["HLCU5520187"].gerekce
    assert "demurrage" in got["OOLU6612408"].gerekce
    assert "haftaya bir ara" in got["CMAU3301986"].gerekce
    # alerts for risky containers only, numbers injected by code
    assert set(report.alerts) == {"MSCU4829153", "MAEU7731042", "HLCU5520187", "CMAU3301986"}
    assert "$750" in report.alerts["MSCU4829153"] and "$760" in report.alerts["HLCU5520187"]


@pytest.fixture(scope="module")
def report():
    return run(load_feed(FEED), FixtureExtractor(ROOT / "tests/fixtures/golden_extractions.json"))


def test_feed_parses_13_messages():
    msgs = load_feed(FEED)
    assert [m.id for m in msgs] == [f"M{i}" for i in range(1, 14)]
    assert msgs[8].text == "MSCU 482915-3 perşembe iade ediyoruz."


def test_case_expected_results(report):
    assert_case_results(report)


def test_m13_never_reaches_the_model(report):
    m13 = next(e for e in report.extractions if e.message.id == "M13")
    assert m13.status == "ignored" and m13.extraction is None


def test_as_of_changes_kalan_gun_and_risk():
    r = run(load_feed(FEED), FixtureExtractor(ROOT / "tests/fixtures/golden_extractions.json"),
            as_of=date(2026, 10, 8))
    maeu = next(x for x in r.results if x.konteyner_no == "MAEU7731042")
    assert maeu.kalan_gun == -1 and maeu.risk == Risk.HIGH  # last free day has passed



def test_recorded_qwen3_output_reaches_case_results():
    """Raw Qwen3-4B output recorded with scripts/record_fixture.py (current prompt/schema).
    Model-free regression test: same pipeline, real model text."""
    report = run(load_feed(FEED), FixtureExtractor(ROOT / "tests/fixtures/qwen3_4b_raw.json"))
    assert_case_results(report)
    assert report.needs_review == []
