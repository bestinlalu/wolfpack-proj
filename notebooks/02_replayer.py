# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Replayer
# MAGIC Stands in for a phone app uploading wearable data. Writes the participant's recorded data into the stream
# MAGIC Volume in timed chunks: `chunk_minutes` of data every `seconds_per_chunk` seconds (30 minutes every 2 s plays
# MAGIC a day in about 1.6 minutes). Run it alongside `03_stream_ingest` and `04_run_agent`.

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.abspath(".."))

import pandas as pd

from bodylab.databricks_io import read_pid_table

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
dbutils.widgets.text("pid", "S01")
dbutils.widgets.text("chunk_minutes", "30")
dbutils.widgets.text("seconds_per_chunk", "2")
dbutils.widgets.dropdown("reset", "yes", ["yes", "no"])
catalog, schema, pid = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"), dbutils.widgets.get("pid")
prefix = f"{catalog}.{schema}"
stream = f"/Volumes/{catalog}/{schema}/stream"
chunk = pd.Timedelta(minutes=int(dbutils.widgets.get("chunk_minutes")))
pause = float(dbutils.widgets.get("seconds_per_chunk"))

# COMMAND ----------

if dbutils.widgets.get("reset") == "yes":
    for sub in ("minute", "meals"):
        dbutils.fs.rm(f"{stream}/{sub}/{pid}", recurse=True)
    for table in ["live_minute", "live_meals"] + [t for t in ("features_fuel", "features_stress", "features_sleep", "features_movement",
                  "nb_events", "nb_hypotheses", "nb_evidence", "nb_discoveries", "nb_quests", "nb_messages", "nb_processed", "agent_state")]:
        if spark.catalog.tableExists(f"{prefix}.{table}"):
            spark.sql(f"DELETE FROM {prefix}.{table} WHERE pid = '{pid}'")
    print("Cleared earlier replay for", pid)

minute = read_pid_table(spark, f"{prefix}.minute_signals", pid).sort_values("ts")
meals = read_pid_table(spark, f"{prefix}.meals", pid).sort_values("ts")
for sub in ("minute", "meals"):
    os.makedirs(f"{stream}/{sub}/{pid}", exist_ok=True)

# COMMAND ----------

t = minute["ts"].min()
end = minute["ts"].max()
i = 0
while t <= end:
    part = minute[(minute["ts"] >= t) & (minute["ts"] < t + chunk)]
    if len(part):
        part.to_parquet(f"{stream}/minute/{pid}/part_{i:05d}.parquet", index=False)
    m = meals[(meals["ts"] >= t) & (meals["ts"] < t + chunk)]
    if len(m):
        m.to_parquet(f"{stream}/meals/{pid}/part_{i:05d}.parquet", index=False)
    if i % 48 == 0:
        print(f"replayed through {t + chunk:%a %b %d %H:%M}")
    t += chunk
    i += 1
    time.sleep(pause)
print("Replay finished")
