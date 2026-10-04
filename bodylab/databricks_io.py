"""Delta table helpers for the Databricks notebooks (one row set per participant, replaced on write)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bodylab.agent.notebook import TABLES, Notebook
from bodylab.labs import LABS


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.attrs = {}
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else (v if isinstance(v, (str, bool)) else str(v)))
    return df


def write_pid_table(spark, df: pd.DataFrame, table: str, pid: str) -> None:
    if df is None or df.empty:
        if spark.catalog.tableExists(table):
            spark.sql(f"DELETE FROM {table} WHERE pid = '{pid}'")
        return
    sdf = spark.createDataFrame(_sanitize(df))
    (sdf.write.format("delta").mode("overwrite").option("replaceWhere", f"pid = '{pid}'")
        .option("mergeSchema", "true").saveAsTable(table))


def read_pid_table(spark, table: str, pid: str) -> pd.DataFrame:
    if not spark.catalog.tableExists(table):
        return pd.DataFrame()
    pdf = spark.table(table).where(f"pid = '{pid}'").toPandas()
    # On serverless, toPandas() puts PlanMetrics objects in pdf.attrs; pandas then fails writing parquet
    # ("Object of type PlanMetrics is not JSON serializable"), so drop them.
    pdf.attrs = {}
    return pdf


def save_state(spark, prefix: str, pid: str, features: dict[str, pd.DataFrame], nb: Notebook, until, agent: str) -> None:
    for lab, df in features.items():
        write_pid_table(spark, df, f"{prefix}.features_{lab}", pid)
    for name, df in nb.to_frames().items():
        if not df.empty and "pid" not in df.columns:
            df.insert(0, "pid", pid)
        write_pid_table(spark, df, f"{prefix}.nb_{name}", pid)
    state = pd.DataFrame([{"pid": pid, "until": pd.Timestamp(until), "agent": agent, "updated_at": pd.Timestamp.utcnow().tz_localize(None)}])
    write_pid_table(spark, state, f"{prefix}.agent_state", pid)


def load_notebook(spark, prefix: str, pid: str) -> Notebook:
    return Notebook.from_frames(pid, {name: read_pid_table(spark, f"{prefix}.nb_{name}", pid) for name in TABLES})


def load_features(spark, prefix: str, pid: str) -> dict[str, pd.DataFrame]:
    return {lab: read_pid_table(spark, f"{prefix}.features_{lab}", pid) for lab in LABS}
