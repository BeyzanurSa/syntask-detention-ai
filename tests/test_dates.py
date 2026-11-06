from datetime import date

import pytest

from detention_ai.dates import resolve

REF = date(2026, 10, 5)  # Monday, date of M9/M10/M11


@pytest.mark.parametrize("expr,expected", [
    ("28 Sep 2026", date(2026, 9, 28)),
    ("01 Oct", date(2026, 10, 1)),
    ("30/09/2026", date(2026, 9, 30)),
    ("28.09.2026", date(2026, 9, 28)),
    ("2026-10-12", date(2026, 10, 12)),
    ("6 Ekim'de", date(2026, 10, 6)),
    ("12 Ekim'e", date(2026, 10, 12)),
    ("9 Ekim Cuma", date(2026, 10, 9)),
    ("Oct 9, 2026", date(2026, 10, 9)),
])
def test_absolute_dates(expr, expected):
    r = resolve(expr, REF)
    assert r.value == expected and r.kind == "absolute" and r.note is None


def test_persembe_resolves_against_message_date():
    assert resolve("perşembe", REF).value == date(2026, 10, 8)
    assert resolve("Perşembe", REF).kind == "weekday"


def test_weekday_same_day_counts_as_today():
    assert resolve("pazartesi", REF).value == REF


def test_haftaya_weekday_is_next_calendar_week():
    assert resolve("haftaya perşembe", REF).value == date(2026, 10, 15)


@pytest.mark.parametrize("expr", ["haftaya bir ara", "yakında", "en kısa sürede", "ay sonu gibi"])
def test_vague_expressions_are_unknown(expr):
    r = resolve(expr, REF)
    assert r.value is None and r.kind == "unknown"


def test_not_mentioned_is_none():
    assert resolve(None, REF) is None
    assert resolve("  ", REF) is None


def test_weekday_mismatch_is_flagged():
    r = resolve("9 Ekim Perşembe", REF)  # 9 Oct 2026 is a Friday
    assert r.value == date(2026, 10, 9) and "uyuşmuyor" in r.note


def test_year_rollover():
    assert resolve("3 Jan", date(2026, 12, 20)).value == date(2027, 1, 3)


def test_invalid_date_is_unknown():
    assert resolve("31 Şubat", REF).value is None


def test_relative_words():
    assert resolve("bugün", REF).value == REF
    assert resolve("yarın", REF).value == date(2026, 10, 6)
