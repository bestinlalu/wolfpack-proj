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

while time.time() < deadline:
    queries = [ingest("minute", "live_minute"), ingest("meals", "live_meals")]
    for q in queries:
        q.awaitTermination()
    time.sleep(3)
print("Stopped ingest loop")
