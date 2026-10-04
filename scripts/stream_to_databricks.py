"""Stream one participant's data from this laptop into Databricks, while 04_run_agent runs there.

Replaces notebooks 02_replayer and 03_stream_ingest, like a phone uploading wristband data: every few seconds the
next chunk of per-minute data and meals is inserted into `live_minute` and `live_meals` through the SQL warehouse,
and 04_run_agent picks it up. Needs DATABRICKS_HOST, DATABRICKS_TOKEN and DATABRICKS_WAREHOUSE_ID in .env.

Data source (--source):
  local       (default) the participant prepared on this laptop, in lakehouse/<pid>/:
              scripts/download_data.sh 001   then   scripts/prepare.py --pid 001
  databricks  the participant's `minute_signals` / `meals` prepared by notebooks/01_prepare_minute

    python scripts/stream_to_databricks.py --pid 001
    python scripts/stream_to_databricks.py --pid 001 --chunk-minutes 120 --no-reset   # faster, keep old results
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", required=True, help="participant to stream, e.g. 001")
    ap.add_argument("--source", choices=["local", "databricks"], default="local", help="where the prepared data is")
    ap.add_argument("--chunk-minutes", type=int, default=60, help="minutes of data per insert")
    ap.add_argument("--seconds", type=float, default=1.0, help="pause between inserts")
    ap.add_argument("--no-reset", action="store_true", help="keep this participant's earlier live data and results")
    args = ap.parse_args()
    if not args.pid.isalnum():
        sys.exit("--pid must be letters and digits, e.g. 001")

    if args.source == "local":
        local = LocalStore()
        if args.pid not in local.pids():
            sys.exit(f"{args.pid} isn't prepared on this laptop. Run scripts/download_data.sh {args.pid} "
                     f"and then scripts/prepare.py --pid {args.pid} first.")
        minute, meals = (naive(df) for df in local.read_inputs(args.pid))

    print("Connecting to the Databricks SQL warehouse (it may take up to a minute to wake up)...")
    wh = Warehouse()
    p = wh.prefix
    if args.source == "databricks":
        minute, meals = wh.read("minute_signals", args.pid), wh.read("meals", args.pid)
        if minute.empty:
            sys.exit(f"No prepared data for {args.pid} in {p}.minute_signals; run 01_prepare_minute first.")
    minute = minute.sort_values("ts").reset_index(drop=True)
    meals = meals.sort_values("ts").reset_index(drop=True)
    print(f"{args.pid} ({args.source}): {len(minute):,} minutes ({minute.ts.min():%b %d} to {minute.ts.max():%b %d}), "
          f"{len(meals)} meals")

    wh.run(create_sql(f"{p}.live_minute", minute))
    wh.run(create_sql(f"{p}.live_meals", meals))
    if not args.no_reset:
        for table in ["live_minute", "live_meals"] + RESULT_TABLES:
            if wh.exists(table):
                wh.run(f"DELETE FROM {p}.{table} WHERE pid = '{args.pid}'")
        print(f"Cleared {args.pid}'s earlier live data and results; the lab starts empty.")

    chunk = pd.Timedelta(minutes=args.chunk_minutes)
    t, end, last_block = minute["ts"].min(), minute["ts"].max(), -1
    started = time.time()
    while t <= end:
        part = minute[(minute["ts"] >= t) & (minute["ts"] < t + chunk)]
        if len(part):
            wh.run(insert_sql(f"{p}.live_minute", part))
        m = meals[(meals["ts"] >= t) & (meals["ts"] < t + chunk)]
        if len(m):
            wh.run(insert_sql(f"{p}.live_meals", m))
        block = int((t - minute["ts"].min()) / pd.Timedelta(hours=6))  # progress every 6 hours of data
        if block != last_block:
            last_block = block
            print(f"streamed through {t + chunk:%a %b %d %H:%M}  ({time.time() - started:.0f}s)")
        t += chunk
        time.sleep(args.seconds)
    print(f"Done: streamed {args.pid} in {(time.time() - started) / 60:.1f} min.")


if __name__ == "__main__":
    main()
