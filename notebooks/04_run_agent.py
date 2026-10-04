# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Run the agent
# MAGIC Every few seconds: read what has streamed in so far, run the engine up to the latest timestamp
# MAGIC (feature tables, hypothesis tests, investigations of new surprises), and write the lab notebook back to Delta.
# MAGIC The Body Lab app reads these tables. Uses Gemini when the `body-lab/gemini_api_key` secret exists.
# MAGIC
# MAGIC The live data can come from `02_replayer` + `03_stream_ingest`, or from a laptop with
# MAGIC `python scripts/stream_to_databricks.py --pid 001`, which inserts straight into `live_minute` and `live_meals`
# MAGIC through the SQL warehouse (no other notebook needed).

# COMMAND ----------

# MAGIC %pip install -q google-genai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
import time

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
from bodylab.agent.notebook import Notebook
from bodylab.databricks_io import load_notebook, read_pids_table, save_states
from bodylab.engine import Engine
from bodylab.meal_log import combine_meals, historical_meals_changed, meal_fingerprint

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "body_lab")
dbutils.widgets.text("pids", "001,002,004,005,006")
dbutils.widgets.text("minutes_to_run", "30")
dbutils.widgets.dropdown("use_gemini", "no", ["yes", "no"])
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
pids = [p.strip() for p in dbutils.widgets.get("pids").split(",") if p.strip().isalnum()]
prefix = f"{catalog}.{schema}"
deadline = time.time() + 60 * float(dbutils.widgets.get("minutes_to_run"))

if dbutils.widgets.get("use_gemini") == "yes":
    try:
        os.environ["GEMINI_API_KEY"] = dbutils.secrets.get("body-lab", "gemini_api_key")
    except Exception:
        print("No Gemini secret found; using the rule-based investigator.")
investigator = make_investigator(prefer_llm=dbutils.widgets.get("use_gemini") == "yes")
print("Investigator:", investigator.name, "| participants:", ", ".join(pids))

# COMMAND ----------

# Each participant's lab notebook stays in memory between loops (this notebook is its only writer during a run);
# Delta is read once at the start and written once per loop for everyone who changed.
notebooks = {pid: load_notebook(spark, prefix, pid) for pid in pids}
last: dict[str, pd.Timestamp] = {}
last_meals: dict[str, pd.DataFrame] = {}
written: dict = {}  # fingerprints of what's already in Delta, so unchanged tables aren't rewritten
waiting_shown = False
while time.time() < deadline:
    live = read_pids_table(spark, f"{prefix}.live_minute", pids)
    live_meals = read_pids_table(spark, f"{prefix}.live_meals", pids)
    photo_meals = read_pids_table(spark, f"{prefix}.photo_meals", pids)
    changed = []
    for pid in pids:
        minute = live[live["pid"] == pid] if not live.empty else live
        if minute.empty:
            if pid in last:  # stream was reset: start this participant over
                notebooks[pid] = Notebook(pid)
                del last[pid]
                last_meals.pop(pid, None)
            continue
        minute = minute.drop_duplicates("ts").sort_values("ts")
        meals = live_meals[live_meals["pid"] == pid] if not live_meals.empty else live_meals
        meals = meals.drop_duplicates("meal_id").sort_values("ts") if not meals.empty else meals
        photos = photo_meals[photo_meals["pid"] == pid] if not photo_meals.empty else photo_meals
        meals = combine_meals(meals, photos)
        until = minute["ts"].max()
        meals_changed = pid not in last_meals or meal_fingerprint(meals) != meal_fingerprint(last_meals[pid])
        if pid in last and until < last[pid]:  # stream restarted from the beginning
            print(f"{pid}: stream restarted, starting this participant's lab over")
            notebooks[pid] = Notebook(pid)
        elif ((pid not in last and not photos.empty)
              or (pid in last_meals and historical_meals_changed(last_meals[pid], meals, last[pid]))):
            # A corrected/backdated meal changes historical baselines and evidence.
            # Rebuild this participant instead of testing old situations twice.
            notebooks[pid] = Notebook(pid)
        elif pid in last and until == last[pid] and not meals_changed:
            continue
        engine = Engine(pid, minute, meals, notebooks[pid], investigator)
        before = len(engine.notebook.messages)
        engine.step(until)
        notebooks[pid] = engine.notebook
        last[pid] = until
        last_meals[pid] = meals.copy()
        changed.append((pid, engine.features, engine.notebook, engine.until, investigator.name))
        for m in engine.notebook.messages[before:]:
            print(f"{pid} {pd.Timestamp(m['ts']):%a %b %d %H:%M} [{m['kind']}] {m['title']}")
    if changed:
        save_states(spark, prefix, changed, written)
    elif not last and not waiting_shown:
        print("Waiting for streamed data (scripts/stream_to_databricks.py or 02/03)...")
        waiting_shown = True
    time.sleep(2)
print("Agent loop stopped at", {pid: str(t) for pid, t in last.items()})

# COMMAND ----------

display(spark.sql(f"SELECT pid, kind, title, body, ts FROM {prefix}.nb_messages "
                  f"WHERE pid IN ({', '.join(repr(p) for p in pids)}) ORDER BY ts DESC"))
