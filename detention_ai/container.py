"""Container number normalization and ISO 6346 check-digit validation.

Format rule (hard): after stripping spaces and dashes -> 4 letters + 7 digits.
Check digit (soft): ISO 6346 is reported as a warning only. None of the six
container numbers in the case pass it, so rejecting would drop every container.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 4 letters, then 7 digits that may be split by spaces or dashes
# ("MSCU 482915-3", "HLCU 552018-7'nin"). Lookarounds keep us from matching
# inside a longer token.
_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{4})[\s\-]*((?:\d[\s\-]*){6}\d)(?!\d)"
)
_STRICT = re.compile(r"^[A-Z]{4}\d{7}$")


@dataclass(frozen=True)
class ContainerNo:
    value: str  # normalized, e.g. "MSCU4829153"
    raw: str  # as written in the message
    check_digit_ok: bool

    @property
    def warning(self) -> str | None:
        if self.check_digit_ok:
            return None
        return (
            f"{self.value}: ISO 6346 kontrol hanesi tutmuyor "
            f"(beklenen {iso6346_check_digit(self.value[:10])}). Numara yine de işlendi."
        )


def normalize(raw: str) -> str | None:
    """Strip spaces/dashes, uppercase, and check the 4 letters + 7 digits format."""
    cleaned = re.sub(r"[\s\-]", "", raw).upper()
    return cleaned if _STRICT.match(cleaned) else None


def iso6346_check_digit(owner_and_serial: str) -> int:
    """ISO 6346 check digit for the first 10 characters (4 letters + 6 digits)."""
    # Letter values start at A=10 and skip multiples of 11 (11, 22, 33).
    values: dict[str, int] = {}
    v = 10
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if v % 11 == 0:
            v += 1
        values[ch] = v
        v += 1
    total = 0
    for i, ch in enumerate(owner_and_serial):
        n = values[ch] if ch.isalpha() else int(ch)
        total += n * (2**i)
    return total % 11 % 10


def parse(raw: str) -> ContainerNo | None:
    value = normalize(raw)
    if value is None:
        return None
    ok = iso6346_check_digit(value[:10]) == int(value[10])
    return ContainerNo(value=value, raw=raw, check_digit_ok=ok)


def find_all(text: str) -> list[ContainerNo]:
    """All container numbers in free text, de-duplicated, in order of appearance."""
    seen: dict[str, ContainerNo] = {}
    for m in _CANDIDATE.finditer(text):
        c = parse(m.group(0))
        if c and c.value not in seen:
            seen[c.value] = c
    return list(seen.values())
