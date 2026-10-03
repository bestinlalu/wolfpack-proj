"""Replay a prepared participant through the agent without the app, and print what it found.

    python scripts/replay.py --pid S01            # rule-based investigator
    python scripts/replay.py --pid S01 --llm      # Gemini agent (needs GEMINI_API_KEY)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from bodylab.agent.investigator import make_investigator  # noqa: E402
from bodylab.engine import Engine  # noqa: E402
from bodylab.store import LocalStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", required=True)
    ap.add_argument("--step-hours", type=float, default=6.0)
    ap.add_argument("--llm", action="store_true", help="use the Gemini investigator")
    ap.add_argument("--no-wait", action="store_true", help="with --llm, fall back to rules on rate limits instead of waiting")
    args = ap.parse_args()

    store = LocalStore()
    minute, meals = store.read_inputs(args.pid)
    eng = Engine(args.pid, minute, meals, investigator=make_investigator(prefer_llm=args.llm, wait_on_rate_limit=not args.no_wait))
    print(f"Replaying {args.pid} with the {eng.investigator.name} investigator, {eng.start:%b %d} to {eng.end:%b %d}")
    t0 = time.time()
    t = eng.start + pd.Timedelta(hours=args.step_hours)
    while t <= eng.end + pd.Timedelta(hours=args.step_hours):
        out = eng.step(t)
        if out["messages"]:
            for m in eng.notebook.messages[-out["messages"]:]:
                print(f"  {pd.Timestamp(m['ts']):%a %b %d %H:%M}  [{m['kind']}] {m['title']}")
        t += pd.Timedelta(hours=args.step_hours)
    store.write_state(args.pid, eng.features, eng.notebook, eng.until, eng.investigator.name)

    nb = eng.notebook
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Funnel: {nb.funnel()}")
    usage = getattr(eng.investigator, "usage", None)
    if usage and usage["requests"]:
        from bodylab.config import SETTINGS
        cost = usage["input_tokens"] / 1e6 * SETTINGS.gemini_price_input_per_m + usage["output_tokens"] / 1e6 * SETTINGS.gemini_price_output_per_m
        agents = pd.Series([e["agent"] for e in nb.events]).value_counts().to_dict()
        print(f"Gemini: {usage['requests']} requests ({usage['requests'] / max(elapsed / 60, 1e-9):.0f}/min), "
              f"{usage['input_tokens']:,} input + {usage['output_tokens']:,} output tokens, about ${cost:.3f} at the configured prices")
        print("Cases by agent:", agents)
    print("Rank:", nb.rank())
    for d in nb.discoveries:
        print(f"  {d['card_id']} {d['title']} [{d['rarity']}, {d['status']}] {d['claim']} ({d['evidence']})")


if __name__ == "__main__":
    main()
