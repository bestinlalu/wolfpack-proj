# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the schema and Volumes Body Lab uses.
# MAGIC
# MAGIC After running this:
# MAGIC 1. Upload each participant's CSVs to `/Volumes/<catalog>/<schema>/raw/<pid>/` (for example `.../raw/001/Dexcom_001.csv`),
# MAGIC    or run `scripts/download_data.sh` locally and upload the `data/raw/<pid>` folders. BVP and IBI files are not needed.
# MAGIC 2. Store API keys as secrets (from a terminal with the Databricks CLI):
# MAGIC    `databricks secrets create-scope body-lab`, then
# MAGIC    `databricks secrets put-secret body-lab gemini_api_key` and `databricks secrets put-secret body-lab elevenlabs_api_key`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
for volume in ("raw", "stream", "checkpoints"):
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{volume}")

print(f"Upload raw files to /Volumes/{catalog}/{schema}/raw/<pid>/")
display(spark.sql(f"SHOW VOLUMES IN {catalog}.{schema}"))
