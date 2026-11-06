"""Deterministic resolution of raw date expressions against the message date.

Supported: "28 Sep 2026", "01 Oct", "30/09/2026", "28.09.2026", "2026-09-28",
"6 Ekim'de", "9 Ekim Cuma", weekday names ("perşembe"), "haftaya perşembe",
"bugün"/"yarın". Anything else (e.g. "haftaya bir ara") -> unknown. We never guess.

Assumptions (documented, configurable later):
- Missing year -> message year; if that lands > 180 days before the message,
  roll forward one year (a December message about "3 Jan").
- A bare weekday means its next occurrence ON OR AFTER the message date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

MONTHS = {
    # English
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    # Turkish (with and without diacritics)
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4, "mayıs": 5,
    "mayis": 5, "haziran": 6, "temmuz": 7, "ağustos": 8, "agustos": 8,
    "eylül": 9, "eylul": 9, "eyl": 9, "ekim": 10, "eki": 10, "kasım": 11,
    "kasim": 11, "aralık": 12, "aralik": 12,
}
WEEKDAYS = {
    "pazartesi": 0, "salı": 1, "sali": 1, "çarşamba": 2, "carsamba": 2,
    "perşembe": 3, "persembe": 3, "cuma": 4, "cumartesi": 5, "pazar": 6,
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
    "saturday": 5, "sunday": 6,
}
TODAY = {"bugün", "bugun", "today"}
TOMORROW = {"yarın", "yarin", "tomorrow"}
NEXT_WEEK = {"haftaya", "next"}

Kind = Literal["absolute", "weekday", "relative", "unknown"]


@dataclass(frozen=True)
class ResolvedDate:
    raw: str
    value: date | None
    kind: Kind
    note: str | None = None


def _tr_lower(s: str) -> str:
    return s.replace("İ", "i").replace("I", "ı").lower()


def _tokens(expr: str) -> list[str]:
    # "Ekim'de" -> "ekim", "12 Ekim'e" -> ["12", "ekim"]
    text = re.sub(r"['’][a-zçğıöşü]*", "", _tr_lower(expr))
    return re.findall(r"[a-zçğıöşü]+|\d+", text)


def _infer_year(day: int, month: int, ref: date) -> date:
    candidate = date(ref.year, month, day)
    if candidate < ref - timedelta(days=180):
        candidate = date(ref.year + 1, month, day)
    return candidate


def _with_year(day: int, month: int, year: str | None, ref: date) -> date:
    if year is None:
        return _infer_year(day, month, ref)
    y = int(year)
    return date(y + 2000 if y < 100 else y, month, day)


def _absolute(expr: str, ref: date) -> date | None:
    s = _tr_lower(expr)
    if m := re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", s):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?\b", s):
        return _with_year(int(m[1]), int(m[2]), m[3], ref)
    toks = _tokens(expr)
    for i, tok in enumerate(toks):
        if tok in MONTHS:
            month = MONTHS[tok]
            # "28 Sep [2026]"
            if i > 0 and toks[i - 1].isdigit() and len(toks[i - 1]) <= 2:
                year = toks[i + 1] if i + 1 < len(toks) and len(toks[i + 1]) == 4 and toks[i + 1].isdigit() else None
                return _with_year(int(toks[i - 1]), month, year, ref)
            # "Sep 28[, 2026]"
            if i + 1 < len(toks) and toks[i + 1].isdigit() and len(toks[i + 1]) <= 2:
                year = toks[i + 2] if i + 2 < len(toks) and len(toks[i + 2]) == 4 and toks[i + 2].isdigit() else None
                return _with_year(int(toks[i + 1]), month, year, ref)
    return None


def resolve(expr: str | None, ref: date) -> ResolvedDate | None:
    """None means "not mentioned"; ResolvedDate(value=None) means "mentioned but unknown"."""
    if expr is None or not expr.strip():
        return None
    toks = _tokens(expr)
    weekday = next((WEEKDAYS[t] for t in toks if t in WEEKDAYS), None)

    try:
        absolute = _absolute(expr, ref)
    except ValueError:  # e.g. 31 Şubat
        return ResolvedDate(expr, None, "unknown", "geçersiz tarih")
    if absolute is not None:
        note = None
        if weekday is not None and absolute.weekday() != weekday:
            note = f"tarih ({absolute:%d.%m.%Y}) ile gün adı uyuşmuyor; tarih esas alındı"
        return ResolvedDate(expr, absolute, "absolute", note)

    if any(t in TODAY for t in toks):
        return ResolvedDate(expr, ref, "relative")
    if any(t in TOMORROW for t in toks):
        return ResolvedDate(expr, ref + timedelta(days=1), "relative")

    if weekday is not None:
        if any(t in NEXT_WEEK for t in toks):
            # "haftaya perşembe": that weekday in the following calendar week
            monday_next = ref - timedelta(days=ref.weekday()) + timedelta(days=7)
            return ResolvedDate(expr, monday_next + timedelta(days=weekday), "weekday")
        delta = (weekday - ref.weekday()) % 7
        return ResolvedDate(expr, ref + timedelta(days=delta), "weekday")

    return ResolvedDate(expr, None, "unknown", "belirli bir tarih değil")
