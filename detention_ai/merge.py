"""Field-by-field merge per container: the latest message that mentions a field wins.

"Latest" = message date, then feed order for same-day messages.
A later *vague* mention (e.g. "haftaya bir ara") overrides an earlier exact date:
the newest information is that the plan is no longer certain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .container import ContainerNo, parse
from .dates import ResolvedDate, resolve
from .messages import Message
from .schema import ContainerMention, RateTier


@dataclass
class Sourced:
    value: Any
    source: str  # message id
    order: tuple[date, int]


@dataclass
class ContainerFacts:
    container: ContainerNo
    gate_out: Sourced | None = None  # value: ResolvedDate
    still_at_terminal: Sourced | None = None  # value: evidence quote
    free_days: Sourced | None = None
    free_days_type: Sourced | None = None
    rate_tiers: Sourced | None = None  # value: list[RateTier]
    planned_return: Sourced | None = None  # value: ResolvedDate
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def set(self, name: str, value: Any, msg: Message) -> None:
        current: Sourced | None = getattr(self, name)
        if current is None or msg.order_key >= current.order:
            setattr(self, name, Sourced(value, msg.id, msg.order_key))


def merge(
    extractions: list[tuple[Message, list[ContainerMention]]],
) -> dict[str, ContainerFacts]:
    facts: dict[str, ContainerFacts] = {}
    for msg, mentions in sorted(extractions, key=lambda e: e[0].order_key):
        for mention in mentions:
            cno = parse(mention.container_no)
            if cno is None:
                continue  # extract.py already filtered these; defensive
            f = facts.setdefault(cno.value, ContainerFacts(container=cno))
            if msg.id not in f.sources:
                f.sources.append(msg.id)

            if mention.gate_out is not None:
                f.set("gate_out", resolve(mention.gate_out.raw, msg.date), msg)
            if mention.at_terminal_evidence is not None:
                f.set("still_at_terminal", mention.at_terminal_evidence, msg)
            if mention.free_time is not None:
                f.set("free_days", mention.free_time.days, msg)
                f.set("free_days_type", mention.free_time.basis, msg)
            if mention.rates is not None:
                f.set("rate_tiers", list(mention.rates.tiers), msg)
            if mention.planned_return is not None:
                f.set("planned_return", resolve(mention.planned_return.raw, msg.date), msg)
    return facts


def resolved(s: Sourced | None) -> ResolvedDate | None:
    return s.value if s else None


__all__ = ["ContainerFacts", "Sourced", "merge", "resolved", "RateTier"]
