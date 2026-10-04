"""Stream participants' data from this laptop into Databricks, while 04_run_agent runs there.

Replaces notebooks 02_replayer and 03_stream_ingest, like a phone uploading wristband data: every few seconds the
next chunk of per-minute data and meals is inserted into `live_minute` and `live_meals` through the SQL warehouse,
and 04_run_agent picks it up. Needs DATABRICKS_HOST, DATABRICKS_TOKEN and DATABRICKS_WAREHOUSE_ID in .env.

Data source (--source):
  local       (default) the participant prepared on this laptop, in lakehouse/<pid>/:
              scripts/download_data.sh 001   then   scripts/prepare.py --pid 001
  databricks  the participant's `minute_signals` / `meals` prepared by notebooks/01_prepare_minute

    python scripts/stream_to_databricks.py --pid 001
    python scripts/stream_to_databricks.py --all                     # everyone in bodylab/users.json, side by side
    python scripts/stream_to_databricks.py --pid 001 002 --chunk-minutes 120 --no-reset   # faster, keep old results
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bodylab.store import DatabricksSqlStore, LocalStore  # noqa: E402
from bodylab.users import load_users  # noqa: E402

RESULT_TABLES = ["features_fuel", "features_stress", "features_sleep", "features_movement", "nb_events",
                 "nb_hypotheses", "nb_evidence", "nb_discoveries", "nb_quests", "nb_messages", "nb_processed",
                 "agent_state"]


def lit(v) -> str:
    """One value as a SQL literal."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NaT:
        return "NULL"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "NULL" if not np.isfinite(v) else repr(float(v))
    if isinstance(v, pd.Timestamp):
        return "NULL" if pd.isna(v) else f"TIMESTAMP '{v:%Y-%m-%d %H:%M:%S}'"
    text = str(v).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def sql_type(dtype) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "STRING"


def create_sql(table: str, df: pd.DataFrame) -> str:
    cols = ", ".join(f"`{c}` {sql_type(t)}" for c, t in df.dtypes.items())
    return f"CREATE TABLE IF NOT EXISTS {table} ({cols})"


def insert_sql(table: str, df: pd.DataFrame) -> str:
    cols = ", ".join(f"`{c}`" for c in df.columns)
    rows = ",\n".join("(" + ", ".join(lit(v) for v in row) + ")" for row in df.itertuples(index=False, name=None))
    return f"INSERT INTO {table} ({cols}) VALUES\n{rows}"


def naive(df: pd.DataFrame) -> pd.DataFrame:
    """The warehouse returns timestamps tagged UTC; the pipeline uses plain times, so drop the tag."""
    df = df.copy()
    df.attrs = {}
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_localize(None)
    return df


class Warehouse:
    def __init__(self):
        self.store = DatabricksSqlStore()
        self.prefix = f"{self.store.catalog}.{self.store.schema}"

    def read(self, table: str, pid: str) -> pd.DataFrame:
        return naive(self.store._q(table, pid))

    def run(self, sql: str) -> None:
        for attempt in range(2):
            try:
                with self.store.conn.cursor() as cur:
                    cur.execute(sql)
                return
            except Exception:
                if attempt:
                    raise
                self.store.conn = self.store._connect()  # warehouse may have dropped the session

    def exists(self, table: str) -> bool:
        try:
            self.store._q(f"{table} LIMIT 0")
            return True
        except Exception:
            return False


def load(source: str, pid: str, wh: "Warehouse | None", local: "LocalStore | None") -> tuple[pd.DataFrame, pd.DataFrame]:
    if source == "local":
        minute, meals = (naive(df) for df in local.read_inputs(pid))
    else:
        minute, meals = wh.read("minute_signals", pid), wh.read("meals", pid)
    return minute.sort_values("ts").reset_index(drop=True), meals.sort_values("ts").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--pid", nargs="+", help="participants to stream, e.g. --pid 001 002")
    who.add_argument("--all", action="store_true", help="every participant assigned to a user in bodylab/users.json")
    ap.add_argument("--source", choices=["local", "databricks"], default="local", help="where the prepared data is")
    ap.add_argument("--chunk-minutes", type=int, default=60, help="minutes of data per insert")
    ap.add_argument("--seconds", type=float, default=1.0, help="pause between inserts")
    ap.add_argument("--no-reset", action="store_true", help="keep these participants' earlier live data and results")
    args = ap.parse_args()
    pids = [u.pid for u in load_users()] if args.all else args.pid
    if not all(pid.isalnum() for pid in pids):
        sys.exit("participant ids must be letters and digits, e.g. 001")

    local = LocalStore() if args.source == "local" else None
    if local is not None:
        missing = [pid for pid in pids if pid not in local.pids()]
        if missing:
            sys.exit(f"{', '.join(missing)} isn't prepared on this laptop. Run scripts/download_data.sh "
                     f"{' '.join(missing)} and then scripts/prepare.py " + " ".join(f"--pid {m}" for m in missing) + " first.")

    print("Connecting to the Databricks SQL warehouse (it may take up to a minute to wake up)...")
    wh = Warehouse()
    p = wh.prefix
    data = {}
    for pid in pids:
        minute, meals = load(args.source, pid, wh, local)
        if minute.empty:
            print(f"{pid}: no prepared data in Databricks; skipped (run 01_prepare_minute for it)")
            continue
        data[pid] = (minute, meals)
        print(f"{pid} ({args.source}): {len(minute):,} minutes ({minute.ts.min():%b %d} to {minute.ts.max():%b %d}), "
              f"{len(meals)} meals")
    if not data:
        sys.exit("Nothing to stream.")
    pids = list(data)
    pid_list = ", ".join(f"'{pid}'" for pid in pids)

    first_minute, first_meals = next(iter(data.values()))
    wh.run(create_sql(f"{p}.live_minute", first_minute))
    wh.run(create_sql(f"{p}.live_meals", first_meals))
    if not args.no_reset:
        for table in ["live_minute", "live_meals"] + RESULT_TABLES:
            if wh.exists(table):
                wh.run(f"DELETE FROM {p}.{table} WHERE pid IN ({pid_list})")
        print(f"Cleared earlier live data and results for {', '.join(pids)}; their labs start empty.")

    # Participants were recorded on different dates, so each one advances by the same amount from its own start.
    chunk = pd.Timedelta(minutes=args.chunk_minutes)
    starts = {pid: d[0]["ts"].min() for pid, d in data.items()}
    longest = max(d[0]["ts"].max() - starts[pid] for pid, d in data.items())
    offset, last_block, started = pd.Timedelta(0), -1, time.time()
    while offset <= longest:
        minute_parts, meal_parts = [], []
        for pid, (minute, meals) in data.items():
            lo, hi = starts[pid] + offset, starts[pid] + offset + chunk
            minute_parts.append(minute[(minute["ts"] >= lo) & (minute["ts"] < hi)])
            meal_parts.append(meals[(meals["ts"] >= lo) & (meals["ts"] < hi)])
        part = pd.concat(minute_parts, ignore_index=True)
        if len(part):
            wh.run(insert_sql(f"{p}.live_minute", part))
        m = pd.concat(meal_parts, ignore_index=True)
        if len(m):
            wh.run(insert_sql(f"{p}.live_meals", m))
        block = int(offset / pd.Timedelta(hours=6))  # progress every 6 hours of data
        if block != last_block:
            last_block = block
            hours = (offset + chunk) / pd.Timedelta(hours=1)
            print(f"streamed {hours:.0f} h of data (day {int(hours // 24) + 1}) for {len(pids)} participant(s)  "
                  f"({time.time() - started:.0f}s)")
        offset += chunk
        time.sleep(args.seconds)
    print(f"Done: streamed {', '.join(pids)} in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
