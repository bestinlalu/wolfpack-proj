# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Batch run (no streaming)
# MAGIC Runs the whole investigation for every prepared participant in one go, in 6-hour steps.
# MAGIC Useful for checking results across all 16 people, or as a fallback if streaming misbehaves during a demo.

# COMMAND ----------

# MAGIC %pip install -q google-genai

# COMMAND ----------

dbutils.library.restartPython()

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

from bodylab.agent.investigator import make_investigator
from bodylab.databricks_io import read_pid_table, save_state, write_pid_table
from bodylab.engine import Engine

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
dbutils.widgets.dropdown("use_gemini", "no", ["yes", "no"])
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
prefix = f"{catalog}.{schema}"
if dbutils.widgets.get("use_gemini") == "yes":
    os.environ["GEMINI_API_KEY"] = dbutils.secrets.get("body-lab", "gemini_api_key")

# COMMAND ----------

pids = [r.pid for r in spark.sql(f"SELECT DISTINCT pid FROM {prefix}.minute_signals ORDER BY pid").collect()]
summary = []
for pid in pids:
    minute = read_pid_table(spark, f"{prefix}.minute_signals", pid)
    meals = read_pid_table(spark, f"{prefix}.meals", pid)
    engine = Engine(pid, minute, meals, investigator=make_investigator(dbutils.widgets.get("use_gemini") == "yes"))
    engine.run(step_hours=6)
    write_pid_table(spark, minute, f"{prefix}.live_minute", pid)
    write_pid_table(spark, meals, f"{prefix}.live_meals", pid)
    save_state(spark, prefix, pid, engine.features, engine.notebook, engine.until, engine.investigator.name)
    summary.append({"pid": pid, **engine.notebook.funnel(), "rank": engine.notebook.rank()[0]})
    print(pid, summary[-1])

display(pd.DataFrame(summary))
