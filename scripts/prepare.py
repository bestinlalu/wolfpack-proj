"""Turn raw BIG IDEAs files (or synthetic data) into per-minute inputs in the local lakehouse.

    python scripts/prepare.py --synthetic            # demo participant S01 with planted effects
    python scripts/prepare.py --pid 001 --pid 002    # real participants from data/raw/<pid>/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bodylab.data import loader, synthetic  # noqa: E402
from bodylab.store import LocalStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true", help="create the synthetic demo participant S01")
    ap.add_argument("--days", type=int, default=14, help="days of synthetic data")
    ap.add_argument("--pid", action="append", default=[], help="real participant id, e.g. 001 (repeatable)")
    ap.add_argument("--raw-dir", type=Path, default=None, help="folder that holds <pid>/ subfolders")
    ap.add_argument("--wrist-offset-hours", type=float, default=None, help="shift wristband timestamps to local time")
    args = ap.parse_args()
    store = LocalStore()

    if args.synthetic:
        minute, meals = synthetic.generate("S01", days=args.days)
        store.write_inputs("S01", minute, meals)
        store.reset_state("S01")
        print(f"S01: {len(minute):,} minutes, {len(meals)} meals")

    for pid in args.pid:
        t = time.time()
        minute, meals = loader.load_participant(pid, args.raw_dir, args.wrist_offset_hours)
        store.write_inputs(pid, minute, meals)
        store.reset_state(pid)
        cover = minute["glucose"].notna().sum()
        print(f"{pid}: {len(minute):,} minutes ({minute.ts.min():%b %d} to {minute.ts.max():%b %d}), "
              f"{cover:,} glucose readings, {len(meals)} meals, worn {minute.worn.mean():.0%}  [{time.time() - t:.0f}s]")

    if not args.synthetic and not args.pid:
        ap.print_help()


if __name__ == "__main__":
    main()
