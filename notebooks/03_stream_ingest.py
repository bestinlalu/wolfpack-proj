# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Streaming ingest (Auto Loader)
# MAGIC Picks up each chunk the replayer writes and appends it to `live_minute` and `live_meals`.
# MAGIC Serverless compute supports the `availableNow` trigger, so it runs in a short loop; on classic
# MAGIC compute you can switch to `processingTime="5 seconds"` and a single long-running stream.

# COMMAND ----------

import time

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
dbutils.widgets.text("minutes_to_run", "30")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
prefix = f"{catalog}.{schema}"
stream = f"/Volumes/{catalog}/{schema}/stream"
checkpoints = f"/Volumes/{catalog}/{schema}/checkpoints"
deadline = time.time() + 60 * float(dbutils.widgets.get("minutes_to_run"))


def ingest(name: str, table: str):
    return (spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "parquet")
            .option("cloudFiles.schemaLocation", f"{checkpoints}/{name}_schema")
            .option("recursiveFileLookup", "true")
            .load(f"{stream}/{name}")
            .writeStream
            .option("checkpointLocation", f"{checkpoints}/{name}")
            .option("mergeSchema", "true")
            .trigger(availableNow=True)
            .toTable(f"{prefix}.{table}"))


# COMMAND ----------

import os


def has_files(name: str) -> bool:
    """Auto Loader can't work out the columns from an empty folder, so each stream waits for its first file."""
    for _, _, files in os.walk(f"{stream}/{name}"):
        if any(f.endswith(".parquet") for f in files):
            return True
    return False


waiting_shown = False
while time.time() < deadline:
    ready = [(name, table) for name, table in (("minute", "live_minute"), ("meals", "live_meals")) if has_files(name)]
    if not ready:
        if not waiting_shown:
            print("Waiting for the replayer (02_replayer) to write its first files...")
            waiting_shown = True
        time.sleep(3)
        continue
    for name, table in ready:
        ingest(name, table).awaitTermination()
    time.sleep(3)
print("Stopped ingest loop")
