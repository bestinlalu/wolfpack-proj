# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Raw files to per-minute signals
# MAGIC Spark reads each participant's wristband CSVs (accelerometer is about 800 MB per person), downsamples them
# MAGIC to one row per minute, and joins glucose. Writes `minute_signals` and `meals` (one replaceable slice per participant).
# MAGIC `pids` defaults to the five participants assigned to users in `bodylab/users.json`. Participants whose files are
# MAGIC missing, or whose food log doesn't have the standard columns (003 has no header, 007 uses `time_of_day`), are
# MAGIC skipped with a message. Add `synthetic` to the list to also load the synthetic participant S01
# MAGIC (planted effects, useful for checking the pipeline).

# COMMAND ----------

import os
import sys

def _repo_root() -> str:
    """Find the folder that contains bodylab/, starting from this notebook's location."""
    starts = [os.getcwd()]
    try:
        nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
        starts.append(os.path.dirname("/Workspace" + nb_path))
    except Exception:
        pass
    for start in starts:
        p = os.path.abspath(start)
        while p != os.path.dirname(p):
            if os.path.isfile(os.path.join(p, "bodylab", "__init__.py")):
                return p
            p = os.path.dirname(p)
    raise ModuleNotFoundError(
        "Couldn't find the bodylab folder. Add the whole GitHub repo as a Git folder "
        "(Workspace > Create > Git folder) and open this notebook from its notebooks/ folder.")


sys.path.insert(0, _repo_root())

import pandas as pd
from pyspark.sql import functions as F

from bodylab.data import loader, synthetic
from bodylab.databricks_io import write_pid_table

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
dbutils.widgets.text("pids", "001,002,004,005,006")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
prefix = f"{catalog}.{schema}"
raw = f"/Volumes/{catalog}/{schema}/raw"
pids = [p.strip() for p in dbutils.widgets.get("pids").split(",") if p.strip()]

# COMMAND ----------


def read_signal(pid: str, name: str):
    df = (spark.read.option("header", True).option("ignoreLeadingWhiteSpace", True)
          .option("ignoreTrailingWhiteSpace", True).csv(f"{raw}/{pid}/{name}_{pid}.csv"))
    for c in df.columns:
        df = df.withColumnRenamed(c, c.strip().lstrip("﻿").lower())
    # Serverless runs in ANSI mode, where a failed parse is an error; try_ versions return NULL so each format gets a turn.
    ts = F.coalesce(F.expr("try_to_timestamp(datetime)"), F.expr("try_to_timestamp(datetime, 'M/d/yy H:mm')"),
                    F.expr("try_to_timestamp(datetime, 'M/d/yy H:mm:ss')"))
    return df.withColumn("datetime", ts).where(F.col("datetime").isNotNull())


def per_minute_mean(pid: str, name: str, col: str) -> pd.DataFrame:
    df = read_signal(pid, name.upper())
    return (df.groupBy(F.date_trunc("minute", "datetime").alias("ts"))
              .agg(F.avg(F.expr(f"try_cast(`{col}` AS DOUBLE)")).alias(col)).toPandas())


def acc_per_minute(pid: str) -> pd.DataFrame:
    df = read_signal(pid, "ACC")
    for c in ("acc_x", "acc_y", "acc_z"):
        df = df.withColumn(c, F.expr(f"try_cast(`{c}` AS DOUBLE)"))

    def per_hour(pdf: pd.DataFrame) -> pd.DataFrame:
        # Runs on Spark workers, which can't import bodylab, so this is a self-contained copy of
        # bodylab.data.loader.acc_minute using only pandas and numpy. Keep the two in sync.
        import numpy as np
        import pandas as pd

        pdf = pdf.sort_values("datetime")
        mag = np.sqrt(pdf["acc_x"] ** 2 + pdf["acc_y"] ** 2 + pdf["acc_z"] ** 2) / 64.0  # E4: 1/64 g per unit
        enmo = np.clip(mag - 1.0, 0, None) * 1000.0
        hp = mag - mag.rolling(32, center=True, min_periods=1).mean()  # remove the 1-second mean (32 Hz)
        is_peak = (hp > 0.12) & (hp == hp.rolling(9, center=True, min_periods=1).max())
        out = pd.DataFrame({"ts": pdf["datetime"].dt.floor("min"), "enmo_sum": enmo, "n": 1, "steps": is_peak.astype(int)})
        return out.groupby("ts", as_index=False).sum()[["ts", "enmo_sum", "n", "steps"]]

    out = (df.withColumn("hour", F.date_trunc("hour", "datetime"))
             .groupBy("hour").applyInPandas(per_hour, schema="ts timestamp, enmo_sum double, n long, steps long")
             .groupBy("ts").agg(F.sum("enmo_sum").alias("enmo_sum"), F.sum("n").alias("n"), F.sum("steps").alias("steps"))
             .toPandas())
    out["enmo_mg"] = out["enmo_sum"] / out["n"]
    return out[["ts", "enmo_mg", "steps"]]


# COMMAND ----------

NEEDED = ["ACC", "EDA", "TEMP", "HR", "Dexcom", "Food_Log"]
done, skipped = [], []
for pid in pids:
    if pid == "synthetic":
        minute, meals = synthetic.generate("S01", days=14)
        pid = "S01"
    else:
        missing = [n for n in NEEDED if not os.path.exists(f"{raw}/{pid}/{n}_{pid}.csv")]
        if missing:
            print(f"{pid}: skipped, missing {', '.join(missing)} in {raw}/{pid}/ (run the download cell in 00_setup)")
            skipped.append(pid)
            continue
        if not loader.food_log_is_standard(f"{raw}/{pid}/Food_Log_{pid}.csv"):
            print(f"{pid}: skipped, food log doesn't have the standard columns")
            skipped.append(pid)
            continue
        wrist = acc_per_minute(pid)
        for name, col in (("hr", "hr"), ("eda", "eda"), ("temp", "temp")):
            wrist = wrist.merge(per_minute_mean(pid, name, col), on="ts", how="outer")
        wrist["ts"] = wrist["ts"] + pd.Timedelta(hours=loader.SETTINGS.wrist_offset_hours)
        glucose = loader.load_glucose(f"{raw}/{pid}/Dexcom_{pid}.csv")
        glucose["ts"] = glucose["ts"].dt.floor("min")
        glucose = glucose.groupby("ts", as_index=False)["glucose"].mean()
        minute = loader.build_minute_frame(pid, wrist, glucose)
        meals = loader.group_meals(loader.load_food_log(f"{raw}/{pid}/Food_Log_{pid}.csv"), pid)
    write_pid_table(spark, minute, f"{prefix}.minute_signals", pid)
    write_pid_table(spark, meals, f"{prefix}.meals", pid)
    done.append(pid)
    print(f"{pid}: {len(minute):,} minutes, {minute['glucose'].notna().sum():,} glucose readings, {len(meals)} meals, worn {minute['worn'].mean():.0%}")
print(f"Prepared {len(done)} participant(s): {', '.join(done) or 'none'}" + (f"; skipped {', '.join(skipped)}" if skipped else ""))

# COMMAND ----------

display(spark.sql(f"SELECT pid, count(*) AS minutes, min(ts) AS start, max(ts) AS end FROM {prefix}.minute_signals GROUP BY pid ORDER BY pid"))
