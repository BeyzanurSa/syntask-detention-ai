from datetime import date

from detention_ai.merge import merge
from detention_ai.messages import Message
from detention_ai.schema import ContainerMention, DateMention, FreeTime, RateTier, Rates


def msg(id, d, seq):
    return Message(id=id, seq=seq, date=d, sender="x", text="")


def mention(no="HLCU5520187", gate_out=None, planned_return=None, free_days=None,
            rate_tiers=None, at_terminal=None):
    return ContainerMention(
        container_no=no,
        gate_out=gate_out and DateMention(evidence=f"gate-out {gate_out}", raw=gate_out),
        at_terminal_evidence=at_terminal,
        free_time=free_days and FreeTime(evidence=f"{free_days} days free", days=free_days,
                                         basis="unspecified"),
        rates=rate_tiers and Rates(evidence="USD", tiers=rate_tiers),
        planned_return=planned_return and DateMention(evidence=f"iade {planned_return}",
                                                      raw=planned_return),
    )


def test_latest_message_wins_per_field_and_formats_unify():
    m3 = (msg("M3", date(2026, 9, 30), 2), [mention(gate_out="30/09/2026", free_days=7,
          rate_tiers=[RateTier(usd_per_day=200, first_n_days=None)])])
    m4 = (msg("M4", date(2026, 10, 2), 3), [mention(planned_return="6 Ekim")])
    m7 = (msg("M7", date(2026, 10, 4), 6), [mention(no="HLCU 552018-7", planned_return="12 Ekim")])
    f = merge([m7, m3, m4])["HLCU5520187"]  # input order must not matter
    assert f.planned_return.value.value == date(2026, 10, 12)
    assert f.planned_return.source == "M7"
    assert f.gate_out.source == "M3" and f.free_days.value == 7  # untouched fields survive
    assert f.sources == ["M3", "M4", "M7"]


def test_same_day_tie_broken_by_feed_order():
    a = (msg("A", date(2026, 10, 5), 1), [mention(planned_return="8 Ekim")])
    b = (msg("B", date(2026, 10, 5), 2), [mention(planned_return="9 Ekim")])
    assert merge([b, a])["HLCU5520187"].planned_return.source == "B"


def test_later_vague_mention_overrides_exact_date():
    a = (msg("A", date(2026, 10, 2), 1), [mention(planned_return="6 Ekim")])
    b = (msg("B", date(2026, 10, 5), 2), [mention(planned_return="haftaya bir ara")])
    pr = merge([a, b])["HLCU5520187"].planned_return
    assert pr.source == "B" and pr.value.value is None


def test_invalid_container_number_dropped():
    assert merge([(msg("A", date(2026, 10, 5), 1), [mention(no="XX12")])]) == {}


def test_later_terminal_status_after_gate_out_is_recorded():
    a = (msg("A", date(2026, 10, 1), 1), [mention(gate_out="01 Oct")])
    b = (msg("B", date(2026, 10, 2), 2), [mention(at_terminal="hâlâ terminalde")])
    f = merge([a, b])["HLCU5520187"]
    assert f.still_at_terminal.source == "B" and f.gate_out.source == "A"
