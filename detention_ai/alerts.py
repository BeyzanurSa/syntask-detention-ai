"""Turkish alert text for the ops team. Template-based: every number comes from
the calculator, none from the model."""

from __future__ import annotations

from .calculator import ContainerResult, Risk


def alert(r: ContainerResult) -> str | None:
    if r.risk == Risk.HIGH:
        if r.kalan_gun is not None and r.kalan_gun < 0:
            timing = f"son serbest gün ({r.son_serbest_gun:%d.%m}) {-r.kalan_gun} gün önce geçti"
        else:
            timing = f"son serbest gün {r.son_serbest_gun:%d.%m} ({r.kalan_gun} gün kaldı)"
        return (
            f"⚠️ {r.konteyner_no}: {timing}; planlanan iade {r.planlanan_iade:%d.%m} ile "
            f"{r.detention_gun} gün detention, tahmini ${r.tahmini_ucret_usd:,.0f}. "
            f"İadeyi öne çekin veya hat ile serbest süre uzatımı görüşün."
        )
    if r.risk == Risk.MEDIUM:
        return (
            f"{r.konteyner_no}: son serbest gün {r.son_serbest_gun:%d.%m} ({r.kalan_gun} gün kaldı), "
            f"ama iade {r.planlanan_iade:%d.%m} planlı; {r.detention_gun} gün için tahmini "
            f"${r.tahmini_ucret_usd:,.0f}. İadeyi {r.son_serbest_gun:%d.%m} veya öncesine almak ücreti sıfırlar."
        )
    if r.risk == Risk.UNKNOWN:
        deadline = (f" Son serbest gün {r.son_serbest_gun:%d.%m} ({r.kalan_gun} gün kaldı)."
                    if r.son_serbest_gun else "")
        return f"❓ {r.konteyner_no}: iade tarihi net değil, manuel takip gerekli.{deadline} Kesin iade tarihini teyit edin."
    return None
