from datetime import date
from decimal import Decimal

from detention_ai.alerts import alert
from detention_ai.calculator import ContainerResult, Risk


def result(**kw):
    base = dict(konteyner_no="MSCU4829153", son_serbest_gun=date(2026, 10, 3), kalan_gun=-2,
                planlanan_iade=date(2026, 10, 8), tahmini_ucret_usd=Decimal(750),
                risk=Risk.HIGH, gerekce="", detention_gun=5)
    return ContainerResult(**{**base, **kw})


def test_high_alert_uses_calculated_numbers():
    text = alert(result())
    assert "MSCU4829153" in text and "$750" in text and "2 gün önce geçti" in text
    assert "5 gün detention" in text


def test_medium_alert():
    text = alert(result(risk=Risk.MEDIUM, kalan_gun=2, son_serbest_gun=date(2026, 10, 7),
                        planlanan_iade=date(2026, 10, 9), tahmini_ucret_usd=Decimal(200),
                        detention_gun=2))
    assert "2 gün kaldı" in text and "$200" in text and "07.10" in text


def test_unknown_alert_asks_for_confirmation():
    text = alert(result(risk=Risk.UNKNOWN, planlanan_iade=None, tahmini_ucret_usd=None,
                        kalan_gun=2, son_serbest_gun=date(2026, 10, 7)))
    assert "manuel" in text and "07.10" in text


def test_no_alert_for_low_and_out_of_scope():
    assert alert(result(risk=Risk.LOW)) is None
    assert alert(result(risk=Risk.OUT_OF_SCOPE)) is None
