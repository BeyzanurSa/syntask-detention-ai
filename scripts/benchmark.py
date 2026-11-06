"""Measure model load and per-message inference separately on this machine.

    /usr/bin/time -l .venv/bin/python scripts/benchmark.py --label warm

Load ("cold" vs "warm") depends on whether the GGUF file is in the OS page cache.
A true cold load needs the cache emptied first (macOS: `sudo purge`, or a reboot):

    sudo purge && /usr/bin/time -l .venv/bin/python scripts/benchmark.py --label cold

Inference is measured in two modes so the effect of llama.cpp prompt-prefix reuse is
measured instead of assumed:
  - reuse:    default behaviour; consecutive calls share the system-prompt prefix
  - no_reuse: llm.reset() before each call, the full prompt is evaluated every time
plus prompt-only timings (max_tokens=1) and the very first call of the process.
Results append to benchmark.json (one entry per run).
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detention_ai.extract import LlamaExtractor  # noqa: E402
from detention_ai.messages import load_feed  # noqa: E402

MODEL = ROOT / "models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"


def timed(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def summary(xs: list[float]) -> dict:
    return {"median_s": round(statistics.median(xs), 2), "min_s": round(min(xs), 2),
            "max_s": round(max(xs), 2), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="unlabelled", help="e.g. cold (after sudo purge) / warm")
    a = ap.parse_args()

    feed = [m for m in load_feed(ROOT / "data/messages.txt") if m.id != "M13"]  # M13 never hits the LLM
    rss0 = psutil.Process().memory_info().rss
    ex = LlamaExtractor(MODEL)
    rss_loaded = psutil.Process().memory_info().rss
    llm = ex.llm

    first_call = timed(lambda: ex.raw_extract(feed[0]))           # first inference of the process
    m1_reuse = timed(lambda: ex.raw_extract(feed[0]))             # identical prompt, prefix reused
    llm.reset()
    m1_no_reuse = timed(lambda: ex.raw_extract(feed[0]))          # warm, full prompt re-evaluated

    llm.reset()
    prompt_full = timed(lambda: ex.raw_extract(feed[1], max_tokens=1))   # whole prompt
    prompt_suffix = timed(lambda: ex.raw_extract(feed[2], max_tokens=1)) # only message part differs

    reuse, no_reuse, tokens = [], [], []
    for m in feed:
        reuse.append(timed(lambda: ex.raw_extract(m)))
        tokens.append(ex.last_usage)
    for m in feed:
        llm.reset()
        no_reuse.append(timed(lambda: ex.raw_extract(m)))

    completion = [u["completion_tokens"] for u in tokens]
    gen_rate = sum(completion) / sum(reuse)
    entry = {
        "label": a.label,
        "time": datetime.now().isoformat(timespec="seconds"),
        "machine": f"{platform.machine()} / {psutil.virtual_memory().total / 2**30:.0f} GB RAM",
        "model": MODEL.name, "model_file_gib": round(MODEL.stat().st_size / 2**30, 2),
        "load_s": round(ex.load_s, 2),
        "first_call_s": round(first_call, 2),
        "m1_repeat_prefix_reuse_s": round(m1_reuse, 2),
        "m1_repeat_no_reuse_s": round(m1_no_reuse, 2),
        "prompt_only_full_s": round(prompt_full, 2),
        "prompt_only_prefix_reused_s": round(prompt_suffix, 2),
        "per_message_prefix_reuse": summary(reuse),
        "per_message_no_reuse": summary(no_reuse),
        "prompt_tokens_median": statistics.median(u["prompt_tokens"] for u in tokens),
        "completion_tokens_median": statistics.median(completion),
        "approx_tokens_per_s_incl_prompt": round(gen_rate, 1),
        "rss_mb": {"before_load": round(rss0 / 2**20), "after_load": round(rss_loaded / 2**20),
                   "end": round(psutil.Process().memory_info().rss / 2**20)},
    }
    path = ROOT / "benchmark.json"
    try:
        runs = json.loads(path.read_text())
        runs = runs if isinstance(runs, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        runs = []
    path.write_text(json.dumps(runs + [entry], ensure_ascii=False, indent=2))
    print(json.dumps(entry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
