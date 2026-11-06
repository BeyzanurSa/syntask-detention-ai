"""Deterministic detention rules. Reproducible under the same input and config.

- Free time starts the day after gate-out.
- Business-day free time skips Sat/Sun (public holidays are NOT modelled).
  Unspecified basis -> calendar days (case rule: "aksi halde takvim günüdür").
- Detention days are calendar days from the day after the last free day up to
  and including the return day -> (return - last_free).days.
- Tiered rates are applied in order; a tier with first_n_days=None covers the rest.
- kalan_gun = (last_free_day - as_of).days, calendar days, may be negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from enum import Enum

from .merge import ContainerFacts
from .schema import RateTier

HIGH_FEE_THRESHOLD = Decimal("500")


class Risk(str, Enum):
    HIGH = "Yüksek"
    MEDIUM = "Orta"
    LOW = "Düşük"
    UNKNOWN = "Belirsiz"
    OUT_OF_SCOPE = "Kapsam dışı"


@dataclass
class ContainerResult:
    konteyner_no: str
    son_serbest_gun: date | None
    kalan_gun: int | None
    planlanan_iade: date | None
    tahmini_ucret_usd: Decimal | None
    risk: Risk
    gerekce: str
    detention_gun: int | None = None
    check_digit_ok: bool = True  # ISO 6346; warning only, never a rejection
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def last_free_day(gate_out: date, free_days: int, business: bool) -> date:
    if not business:
        return gate_out + timedelta(days=free_days)
    day, counted = gate_out, 0
    while counted < free_days:
        day += timedelta(days=1)
        if day.weekday() < 5:
            counted += 1
    return day


def detention_days(last_free: date, return_day: date) -> int:
    return max(0, (return_day - last_free).days)


def money(d: Decimal) -> str:
    return f"{d:,.0f}" if d == d.to_integral_value() else f"{d:,.2f}"


def fee(days: int, tiers: list[RateTier]) -> tuple[Decimal, list[str]]:
    """Returns (total USD, breakdown lines like '3 x $120')."""
    total, remaining, parts = Decimal("0"), days, []
    for tier in tiers:
        if remaining <= 0:
            break
        n = remaining if tier.first_n_days is None else min(tier.first_n_days, remaining)
        rate = Decimal(str(tier.usd_per_day))
        total += n * rate
        parts.append(f"{n} x ${money(rate)}")
        remaining -= n
    if remaining > 0 and tiers:  # last tier had a fixed length: extend it
        rate = Decimal(str(tiers[-1].usd_per_day))
        total += remaining * rate
        parts.append(f"{remaining} x ${money(rate)}")
    return total, parts


def risk_level(total: Decimal, last_free: date, as_of: date) -> Risk:
    if total <= 0:
        return Risk.LOW
    if as_of > last_free or total >= HIGH_FEE_THRESHOLD:
        return Risk.HIGH
    return Risk.MEDIUM


def evaluate(f: ContainerFacts, as_of: date) -> ContainerResult:
    no = f.container.value
    base = dict(konteyner_no=no, sources=f.sources, warnings=list(f.warnings),
                check_digit_ok=f.container.check_digit_ok)
    gate = f.gate_out.value.value if f.gate_out else None

    at_terminal = f.still_at_terminal is not None and (
        f.gate_out is None or f.still_at_terminal.order > f.gate_out.order
    )
    if at_terminal:
        return ContainerResult(
            son_serbest_gun=None, kalan_gun=None, planlanan_iade=None,
            tahmini_ucret_usd=None, risk=Risk.OUT_OF_SCOPE,
            gerekce=f"Terminalde ({f.still_at_terminal.source}: \"{f.still_at_terminal.value}\"); "
                    "demurrage kapsamı, detention değil.",
            **base,
        )

    missing = [name for name, v in (("gate-out", gate), ("serbest süre", f.free_days),
                                    ("ücret", f.rate_tiers)) if v is None]
    if missing:
        return ContainerResult(
            son_serbest_gun=None, kalan_gun=None, planlanan_iade=None,
            tahmini_ucret_usd=None, risk=Risk.UNKNOWN,
            gerekce=f"Eksik bilgi: {', '.join(missing)}; manuel inceleme.", **base,
        )

    basis = f.free_days_type.value if f.free_days_type else "unspecified"
    business = basis == "business"
    lfd = last_free_day(gate, f.free_days.value, business)
    kalan = (lfd - as_of).days
    basis_tr = {"business": "iş günü", "calendar": "takvim günü",
                "unspecified": "takvim günü (tür yazılmamış)"}[basis]
    head = (f"Gate-out {gate:%d.%m} ({f.gate_out.source}) + {f.free_days.value} {basis_tr}"
            f" → son gün {lfd:%d.%m}.")

    planned = f.planned_return.value if f.planned_return else None
    if planned is None or planned.value is None:
        why = (f"{f.planned_return.source} \"{planned.raw}\" tarih değil"
               if planned else "hiç bildirilmemiş")
        return ContainerResult(
            son_serbest_gun=lfd, kalan_gun=kalan, planlanan_iade=None,
            tahmini_ucret_usd=None, risk=Risk.UNKNOWN,
            gerekce=f"{head} İade bilinmiyor ({why}); tahmin edilmedi, manuel inceleme.",
            **base,
        )

    ret = planned.value
    if planned.note:
        base["warnings"].append(f"{f.planned_return.source}: {planned.note}")
    days = detention_days(lfd, ret)
    total, parts = fee(days, f.rate_tiers.value)
    risk = risk_level(total, lfd, as_of)
    src = f"{f.planned_return.source} \"{planned.raw}\""
    if days == 0:
        tail = f"İade {ret:%d.%m} ({src}) serbest süre içinde, ücret yok."
    else:
        tail = (f"İade {ret:%d.%m} ({src}): {lfd + timedelta(days=1):%d.%m}–{ret:%d.%m} "
                f"{days} gün, {' + '.join(parts)} = ${money(total)}.")
    return ContainerResult(
        son_serbest_gun=lfd, kalan_gun=kalan, planlanan_iade=ret,
        tahmini_ucret_usd=total, risk=risk, gerekce=f"{head} {tail}",
        detention_gun=days, **base,
    )
