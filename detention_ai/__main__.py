"""CLI: python -m detention_ai data/messages.txt [--as-of 2026-10-05] [--fixture F] [--json]

Output does not depend on terminal width: the main table has six short columns and
never truncates or wraps a cell. It tries one-line headers, then two-line headers
(split at the last underscore), and below that prints each container as a small card.
Long explanations, alerts and notes are free text below and wrap naturally."""

from __future__ import annotations

import argparse
import io
import json
import statistics
from datetime import date
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from .calculator import ContainerResult, Risk, money
from .extract import FixtureExtractor, LlamaExtractor
from .messages import load_feed
from .pipeline import Report, run

DEFAULT_MODEL = Path(__file__).parent.parent / "models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"
RISK_STYLE = {Risk.HIGH: "bold red", Risk.MEDIUM: "yellow", Risk.LOW: "green",
              Risk.UNKNOWN: "magenta", Risk.OUT_OF_SCOPE: "dim"}
COLUMNS = ("container_no", "last_free_day", "remaining_days", "planned_return",
           "estimated_fee", "risk")


def _cells(r: ContainerResult) -> tuple[str, ...]:
    d = lambda x: f"{x:%d.%m.%Y}" if x else "—"
    return (r.konteyner_no, d(r.son_serbest_gun),
            "—" if r.kalan_gun is None else str(r.kalan_gun), d(r.planlanan_iade),
            "—" if r.tahmini_ucret_usd is None else f"${money(r.tahmini_ucret_usd)}",
            r.risk.value)


def _header(name: str, two_line: bool) -> str:
    # "last_free_day" -> "last_free\n_day": same field name, half the width
    if not two_line or "_" not in name:
        return name
    head, tail = name.rsplit("_", 1)
    return f"{head}\n_{tail}"


def _table(results: list[ContainerResult], two_line_header: bool = False) -> Table:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, show_edge=False)
    for name in COLUMNS:
        right = name in ("remaining_days", "estimated_fee")
        t.add_column(_header(name, two_line_header), no_wrap=True,
                     justify="right" if right else "left")
    for r in results:
        cells = list(_cells(r))
        cells[-1] = f"[{RISK_STYLE[r.risk]}]{cells[-1]}[/]"
        t.add_row(*cells)
    return t


def _cards(results: list[ContainerResult]) -> Group:
    parts = []
    for r in results:
        cells = _cells(r)
        parts.append(Text.assemble((cells[0], "bold"), "  ", (cells[-1], RISK_STYLE[r.risk])))
        for name, value in zip(COLUMNS[1:-1], cells[1:-1]):
            parts.append(Text(f"  {name:<15}{value}"))
    return Group(*parts)


def _natural_width(renderable) -> int:
    """Width the renderable needs without truncation (rendered off-screen)."""
    probe = Console(file=io.StringIO(), width=1000, color_system=None)
    probe.print(renderable)
    return max(len(line.rstrip()) for line in probe.file.getvalue().splitlines())


def render(report: Report, console: Console) -> None:
    n_ok = sum(e.status == "ok" for e in report.extractions)
    console.print(Text.assemble(
        ("Detention riski", "bold"),
        f"  as_of {report.as_of:%d.%m.%Y} · {len(report.extractions)} mesaj, "
        f"{n_ok} işlendi, {len(report.ignored)} yok sayıldı, "
        f"{len(report.needs_review)} manuel inceleme"))
    console.print()

    for two_line in (False, True):
        table = _table(report.results, two_line)
        if _natural_width(table) <= console.width:
            console.print(table)
            break
    else:
        console.print(_cards(report.results))

    console.print("\n[bold]Açıklamalar[/]")
    for r in report.results:
        console.print(Text.assemble("• ", (r.konteyner_no, "bold"), f": {r.gerekce}"))

    if report.alerts:
        console.print("\n[bold]Operasyon uyarıları[/]")
        for text in report.alerts.values():
            console.print(f"• {text}")

    console.print("\n[bold]Notlar[/]")
    bad_iso = [r.konteyner_no for r in report.results if not r.check_digit_ok]
    if bad_iso:
        console.print(f"[dim]• ISO 6346: {len(bad_iso)}/{len(report.results)} numaranın kontrol "
                      f"hanesi tutmuyor ({', '.join(bad_iso)}). Format geçerli olduğu için "
                      f"reddedilmedi; numaralar kaynağından doğrulanmalı.[/]")
    for e in report.extractions:
        if e.status == "ignored":
            console.print(f"[dim]• {e.message.id} yok sayıldı: konteyner numarası yok.[/]")
    checks = [w for e in report.extractions if e.status != "ignored" for w in e.warnings]
    checks += [w for r in report.results for w in r.warnings]
    if checks:
        console.print("[dim]• Doğrulama katmanının düzelttikleri:[/]")
        for w in checks:
            console.print(f"[dim]  - {w}[/]")
    else:
        console.print("[dim]• Doğrulama katmanı model çıktısında düzeltme gerektirmedi.[/]")
    if report.needs_review:
        console.print(f"[red]• Manuel inceleme gereken mesajlar: {', '.join(report.needs_review)}[/]")


def to_json(report: Report) -> str:
    rows = [{
        "konteyner_no": r.konteyner_no,
        "son_serbest_gun": r.son_serbest_gun and r.son_serbest_gun.isoformat(),
        "kalan_gun": r.kalan_gun,
        "planlanan_iade": r.planlanan_iade and r.planlanan_iade.isoformat(),
        "tahmini_ucret_usd": None if r.tahmini_ucret_usd is None else float(r.tahmini_ucret_usd),
        "risk": r.risk.value, "gerekce": r.gerekce, "uyari": report.alerts.get(r.konteyner_no),
        "iso6346_kontrol_hanesi_gecerli": r.check_digit_ok, "kaynak_mesajlar": r.sources,
    } for r in report.results]
    return json.dumps({"as_of": report.as_of.isoformat(), "containers": rows,
                       "ignored_messages": report.ignored,
                       "needs_review": report.needs_review}, ensure_ascii=False, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(prog="detention_ai")
    p.add_argument("feed", type=Path)
    p.add_argument("--as-of", type=date.fromisoformat, default=None,
                   help="Reference date (default: latest message date)")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--fixture", type=Path, help="Replay recorded extractions instead of the LLM")
    p.add_argument("--json", action="store_true")
    p.add_argument("--dump-extractions", type=Path, help="Write extractions to this JSON file")
    a = p.parse_args()

    console = Console(stderr=a.json)  # keep stdout clean for --json
    messages = load_feed(a.feed)
    if a.fixture:
        extractor = FixtureExtractor(a.fixture)
    else:
        with console.status("Model yükleniyor..."):
            extractor = LlamaExtractor(a.model)
        console.print(f"[dim]Model yüklendi: {extractor.load_s:.1f} s[/]")

    with console.status("Mesajlar okunuyor...") as status:
        def progress(e):
            status.update(f"Mesajlar okunuyor... {e.message.seq + 1}/{len(messages)}")
        report = run(messages, extractor, a.as_of, on_message=progress)
    lats = [e.latency_s for e in report.extractions if e.latency_s is not None]
    if lats and not a.fixture:
        console.print(f"[dim]Çıkarım: {len(lats)} LLM çağrısı, medyan {statistics.median(lats):.1f} s, "
                      f"toplam {sum(lats):.0f} s[/]")

    if a.dump_extractions:
        a.dump_extractions.write_text(json.dumps(
            {e.message.id: e.extraction.model_dump() for e in report.extractions if e.extraction},
            ensure_ascii=False, indent=2), encoding="utf-8")
    if a.json:
        print(to_json(report))
    else:
        render(report, console)


if __name__ == "__main__":
    main()
