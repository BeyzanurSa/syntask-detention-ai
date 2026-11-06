"""CLI rendering at different terminal widths."""

import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from detention_ai.__main__ import render, to_json
from detention_ai.extract import FixtureExtractor
from detention_ai.messages import load_feed
from detention_ai.pipeline import run

ROOT = Path(__file__).parent.parent
NUMBERS = ["MSCU4829153", "MAEU7731042", "HLCU5520187", "CMAU3301986", "TGHU9004512", "OOLU6612408"]


@pytest.fixture(scope="module")
def report():
    return run(load_feed(ROOT / "data/messages.txt"),
               FixtureExtractor(ROOT / "tests/fixtures/golden_extractions.json"))


def rendered(report, width):
    buf = io.StringIO()
    render(report, Console(file=buf, width=width, color_system=None, legacy_windows=False))
    return buf.getvalue()


@pytest.mark.parametrize("width", [40, 50, 80, 120])
def test_numbers_never_split_and_lines_fit(report, width):
    out = rendered(report, width)
    for no in NUMBERS:
        assert no in out
    assert "…" not in out  # nothing truncated
    assert max(len(line) for line in out.splitlines()) <= width


def test_80_columns_uses_table_with_two_line_headers(report):
    out = rendered(report, 80)
    lines = out.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith("container"))
    assert "last_free" in lines[i] and "_day" in lines[i + 1]
    row = next(line for line in lines if line.startswith("HLCU5520187"))
    assert "07.10.2026" in row and "$760" in row and "Yüksek" in row


def test_wide_terminal_shows_six_column_table(report):
    out = rendered(report, 120)
    header = next(line for line in out.splitlines() if "container_no" in line)
    for col in ("last_free_day", "remaining_days", "planned_return", "estimated_fee", "risk"):
        assert col in header
    assert "gerekce" not in header
    row = next(line for line in out.splitlines() if line.startswith("MSCU4829153"))
    assert "03.10.2026" in row and "$750" in row and "Yüksek" in row


def test_narrow_terminal_falls_back_to_cards(report):
    out = rendered(report, 50)
    assert "container" not in out
    assert "  last_free_day  03.10.2026" in out


def test_iso_warning_merged_into_one_line(report):
    out = rendered(report, 120)
    assert out.count("ISO 6346") == 1
    assert "6/6" in out and "reddedilmedi" in out


def test_sections_present(report):
    out = rendered(report, 80)
    for section in ("Açıklamalar", "Operasyon uyarıları", "Notlar"):
        assert section in out
    assert "M13 yok sayıldı" in out


def test_json_keeps_case_field_names(report):
    data = json.loads(to_json(report))
    row = data["containers"][0]
    assert list(row)[:7] == ["konteyner_no", "son_serbest_gun", "kalan_gun", "planlanan_iade",
                             "tahmini_ucret_usd", "risk", "gerekce"]
    assert data["ignored_messages"] == ["M13"]
