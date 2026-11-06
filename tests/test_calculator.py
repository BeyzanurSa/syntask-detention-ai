from datetime import date
from decimal import Decimal

import pytest

from detention_ai.calculator import Risk, detention_days, fee, last_free_day, risk_level
from detention_ai.schema import RateTier


def test_calendar_free_time_starts_day_after_gate_out():
    assert last_free_day(date(2026, 9, 28), 5, business=False) == date(2026, 10, 3)


def test_business_days_skip_weekend():
    # gate-out Thu 01.10 -> Fri 02, Mon 05, Tue 06, Wed 07
    assert last_free_day(date(2026, 10, 1), 4, business=True) == date(2026, 10, 7)


def test_business_days_gate_out_on_friday():
    assert last_free_day(date(2026, 10, 2), 1, business=True) == date(2026, 10, 5)


def test_zero_free_days():
    assert last_free_day(date(2026, 10, 2), 0, business=False) == date(2026, 10, 2)


@pytest.mark.parametrize("ret,days", [
    (date(2026, 10, 3), 0),   # returned on last free day
    (date(2026, 10, 1), 0),   # early
    (date(2026, 10, 4), 1),   # day after
    (date(2026, 10, 8), 5),   # MSCU
])
def test_detention_days_include_return_day(ret, days):
    assert detention_days(date(2026, 10, 3), ret) == days


def test_flat_fee():
    assert fee(5, [RateTier(usd_per_day=150, first_n_days=None)])[0] == Decimal("750")


def test_tiered_fee():
    tiers = [RateTier(usd_per_day=120, first_n_days=3), RateTier(usd_per_day=200, first_n_days=None)]
    total, parts = fee(5, tiers)
    assert total == Decimal("760") and parts == ["3 x $120", "2 x $200"]
    assert fee(2, tiers)[0] == Decimal("240")


def test_last_fixed_tier_extends():
    assert fee(5, [RateTier(usd_per_day=100, first_n_days=3)])[0] == Decimal("500")


@pytest.mark.parametrize("total,lfd,as_of,risk", [
    (Decimal("0"), date(2026, 10, 3), date(2026, 10, 5), Risk.LOW),
    (Decimal("200"), date(2026, 10, 7), date(2026, 10, 5), Risk.MEDIUM),
    (Decimal("200"), date(2026, 10, 3), date(2026, 10, 5), Risk.HIGH),   # passed
    (Decimal("200"), date(2026, 10, 5), date(2026, 10, 5), Risk.MEDIUM), # today = not passed
    (Decimal("500"), date(2026, 10, 7), date(2026, 10, 5), Risk.HIGH),   # >= 500 boundary
    (Decimal("499"), date(2026, 10, 7), date(2026, 10, 5), Risk.MEDIUM),
])
def test_risk_levels(total, lfd, as_of, risk):
    assert risk_level(total, lfd, as_of) == risk
