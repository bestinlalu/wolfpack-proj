# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Run the agent
# MAGIC Every few seconds: read what has streamed in so far, run the engine up to the latest timestamp
# MAGIC (feature tables, hypothesis tests, investigations of new surprises), and write the lab notebook back to Delta.
# MAGIC The Body Lab app reads these tables. Uses Gemini when the `body-lab/gemini_api_key` secret exists.

# COMMAND ----------

# MAGIC %pip install -q google-genai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import time

sys.path.append(os.path.abspath(".."))

import pandas as pd

from bodylab.agent.investigator import make_investigator
from bodylab.databricks_io import load_features, load_notebook, read_pid_table, save_state
from bodylab.engine import Engine

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
dbutils.widgets.text("pid", "S01")
dbutils.widgets.text("minutes_to_run", "30")
dbutils.widgets.dropdown("use_gemini", "yes", ["yes", "no"])
catalog, schema, pid = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"), dbutils.widgets.get("pid")
prefix = f"{catalog}.{schema}"
deadline = time.time() + 60 * float(dbutils.widgets.get("minutes_to_run"))

try:
    os.environ["GEMINI_API_KEY"] = dbutils.secrets.get("body-lab", "gemini_api_key")
except Exception:
    print("No Gemini secret found; using the rule-based investigator.")
investigator = make_investigator(prefer_llm=dbutils.widgets.get("use_gemini") == "yes")
print("Investigator:", investigator.name)

# COMMAND ----------

last = None
while time.time() < deadline:
    minute = read_pid_table(spark, f"{prefix}.live_minute", pid)
    meals = read_pid_table(spark, f"{prefix}.live_meals", pid)
    if minute.empty:
        time.sleep(5)
        continue
    minute = minute.drop_duplicates("ts").sort_values("ts")
    meals = meals.drop_duplicates("meal_id").sort_values("ts") if not meals.empty else meals
    until = minute["ts"].max()
    if last is not None and until <= last:
        time.sleep(5)
        continue
    engine = Engine(pid, minute, meals, load_notebook(spark, prefix, pid), investigator)
    engine.features = load_features(spark, prefix, pid)
    out = engine.step(until)
    save_state(spark, prefix, pid, engine.features, engine.notebook, engine.until, investigator.name)
    for m in engine.notebook.messages[len(engine.notebook.messages) - out["messages"]:]:
        print(f"{pd.Timestamp(m['ts']):%a %H:%M} [{m['kind']}] {m['title']}")
    last = until
    time.sleep(5)
print("Agent loop stopped at", last)

# COMMAND ----------

display(spark.sql(f"SELECT kind, title, body, ts FROM {prefix}.nb_messages WHERE pid = '{pid}' ORDER BY ts DESC"))
