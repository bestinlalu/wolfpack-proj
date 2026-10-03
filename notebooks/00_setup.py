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

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download participants into the raw Volume
# MAGIC `download_pids` defaults to the five participants assigned to users in `bodylab/users.json` (`001`–`005`); leave it empty
# MAGIC to skip. About 5 minutes per participant.
# MAGIC The large wristband files (ACC, EDA, TEMP, about 1 GB) come from PhysioNet's fast Amazon S3 mirror, whose 1.0.0 copies
# MAGIC are identical in size to version 1.1.3 for all 16 participants. HR, Dexcom and the food log come from PhysioNet 1.1.3,
# MAGIC because the 1.0.0 copies have wrong HR dates and no food logs. Finished files are skipped. BVP and IBI are not used.
# MAGIC If this fails with a connection error, the workspace can't reach outside websites: run `scripts/download_data.sh`
# MAGIC on your laptop and upload the `data/raw/<pid>` folder to this Volume instead.

# COMMAND ----------

import os
import time

import requests

dbutils.widgets.text("download_pids", "001,002,003,004,005")
pids = [p.strip() for p in dbutils.widgets.get("download_pids").split(",") if p.strip()]
PHYSIONET = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
MIRROR = "https://physionet-open.s3.amazonaws.com/big-ideas-glycemic-wearable/1.0.0"
SOURCES = {"ACC": [MIRROR, PHYSIONET], "EDA": [MIRROR, PHYSIONET], "TEMP": [MIRROR, PHYSIONET],
           "HR": [PHYSIONET], "Dexcom": [PHYSIONET], "Food_Log": [PHYSIONET]}
raw = f"/Volumes/{catalog}/{schema}/raw"


def download(url: str, dest: str) -> None:
    expected = int(requests.head(url, timeout=60, allow_redirects=True).headers.get("Content-Length", 0))
    if os.path.exists(dest) and expected and os.path.getsize(dest) == expected:
        print(f"  {os.path.basename(dest)}: already downloaded")
        return
    t = time.time()
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8 << 20):
                f.write(chunk)
    size = os.path.getsize(dest)
    shown = f"{size / 1e6:.1f} MB" if size >= 1e6 else f"{size / 1e3:.0f} KB"
    print(f"  {os.path.basename(dest)}: {shown} in {time.time() - t:.0f}s")


for pid in pids:
    os.makedirs(f"{raw}/{pid}", exist_ok=True)
    print(pid)
    for name, bases in SOURCES.items():
        dest = f"{raw}/{pid}/{name}_{pid}.csv"
        for i, base in enumerate(bases):
            try:
                download(f"{base}/{pid}/{name}_{pid}.csv", dest)
                break
            except Exception as exc:
                if i == len(bases) - 1:
                    raise
                print(f"  {name}: fast mirror failed ({exc}); trying PhysioNet")
print("Done. Next: run 01_prepare_minute with pids =", ",".join(pids) or "<pid>")
