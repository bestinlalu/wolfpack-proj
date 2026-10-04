"""Plant clear personal patterns into prepared real participants, so the demo produces more cards.

    python scripts/augment.py --all            # everyone in bodylab/users.json
    python scripts/augment.py --pid 001        # one participant (repeatable)
    python scripts/augment.py --all --undo     # put the real data back

Food logged within 2 hours of the start of a meal is merged into that meal. The untouched data is kept as
lakehouse/<pid>/minute.original.parquet and meals.original.parquet, and every run starts from it, so running
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

from bodylab.data.augment import augment, merge_meals  # noqa: E402
from bodylab.store import LocalStore  # noqa: E402
from bodylab.users import load_users  # noqa: E402


# Participants whose real data is noisier get stronger, steadier patterns: (pattern strength, share of scatter kept).
PER_PERSON = {"002": (1.8, 0.2), "006": (1.5, 0.25)}


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
        meals_now = store.root / pid / "meals.parquet"
        meals_original = store.root / pid / "meals.original.parquet"
        if not current.exists():
            print(f"{pid}: not prepared, skipped (run scripts/prepare.py --pid {pid} first)")
            continue
        if args.undo:
            if meals_original.exists():
                shutil.move(meals_original, meals_now)
            if original.exists():
                shutil.move(original, current)
                store.reset_state(pid)
                print(f"{pid}: real data restored")
            else:
                print(f"{pid}: already the real data")
            continue
        if not original.exists():
            shutil.copy2(current, original)
        if not meals_original.exists():
            shutil.copy2(meals_now, meals_original)
        minute = pd.read_parquet(original)
        meals = merge_meals(pd.read_parquet(meals_original))
        meals.to_parquet(meals_now, index=False)
        boost, keep = PER_PERSON.get(pid, (1.0, 0.4))
        augment(minute, meals, seed=int(pid) if pid.isdigit() else 0, boost=boost, keep=keep).to_parquet(current, index=False)
        store.reset_state(pid)
        print(f"{pid}: patterns planted ({len(minute):,} minutes; original kept in {original.name})")


if __name__ == "__main__":
    main()
