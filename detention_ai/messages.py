"""Parse the raw message feed ("[M1] 28.09.2026 - Hat şirketi" + body)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

_HEADER = re.compile(r"^\[(M\d+)\]\s+(\d{2}\.\d{2}\.\d{4})\s+-\s+(.+?)\s*$")


@dataclass(frozen=True)
class Message:
    id: str  # "M1"
    seq: int  # position in the feed, tie-breaker for same-day messages
    date: date
    sender: str
    text: str

    @property
    def order_key(self) -> tuple[date, int]:
        return (self.date, self.seq)


def parse_feed(raw: str) -> list[Message]:
    messages: list[Message] = []
    current: dict | None = None
    body: list[str] = []

    def flush() -> None:
        if current is not None:
            text = " ".join(line.strip() for line in body if line.strip())
            messages.append(Message(text=text, seq=len(messages), **current))

    for line in raw.splitlines():
        m = _HEADER.match(line.strip())
        if m:
            flush()
            current = {
                "id": m.group(1),
                "date": datetime.strptime(m.group(2), "%d.%m.%Y").date(),
                "sender": m.group(3),
            }
            body = []
        elif current is not None:
            body.append(line)
    flush()
    return messages


def load_feed(path: str | Path) -> list[Message]:
    return parse_feed(Path(path).read_text(encoding="utf-8"))
