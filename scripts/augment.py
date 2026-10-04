"""Plant clear personal patterns into prepared real participants, so the demo produces more cards.

    python scripts/augment.py --all            # everyone in bodylab/users.json
    python scripts/augment.py --pid 001        # one participant (repeatable)
    python scripts/augment.py --all --undo     # put the real data back

The untouched data is kept as lakehouse/<pid>/minute.original.parquet, and every run starts from it, so running
this twice doesn't stack the patterns. Saved agent results are cleared; afterwards stream again with
`python scripts/stream_to_databricks.py --all` (or replay locally). See bodylab/data/augment.py for the patterns.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from bodylab.data.augment import augment  # noqa: E402
from bodylab.store import LocalStore  # noqa: E402
from bodylab.users import load_users  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--pid", action="append", help="participant id, e.g. 001 (repeatable)")
    who.add_argument("--all", action="store_true", help="every participant assigned to a user in bodylab/users.json")
    ap.add_argument("--undo", action="store_true", help="restore the original real data")
    args = ap.parse_args()

    store = LocalStore()
    pids = [u.pid for u in load_users()] if args.all else args.pid
    for pid in pids:
        current = store.root / pid / "minute.parquet"
        original = store.root / pid / "minute.original.parquet"
        if not current.exists():
            print(f"{pid}: not prepared, skipped (run scripts/prepare.py --pid {pid} first)")
            continue
        if args.undo:
            if original.exists():
                shutil.move(original, current)
                store.reset_state(pid)
                print(f"{pid}: real data restored")
            else:
                print(f"{pid}: already the real data")
            continue
        if not original.exists():
            shutil.copy2(current, original)
        minute = pd.read_parquet(original)
        meals = pd.read_parquet(store.root / pid / "meals.parquet")
        augment(minute, meals, seed=int(pid) if pid.isdigit() else 0).to_parquet(current, index=False)
        store.reset_state(pid)
        print(f"{pid}: patterns planted ({len(minute):,} minutes; original kept in {original.name})")


if __name__ == "__main__":
    main()
