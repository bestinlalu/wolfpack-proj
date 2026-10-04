"""Where inputs, features and the lab notebook live.

LocalStore: parquet files under ./lakehouse (development and offline demo).
DatabricksSqlStore: reads the Delta tables written by the Databricks notebooks through a SQL warehouse
(used by the Databricks App).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from bodylab.agent.notebook import TABLES, Notebook
from bodylab.config import SETTINGS
from bodylab.labs import LABS
from bodylab.meal_log import PHOTO_COLUMNS, PHOTO_MEALS_SCHEMA, combine_meals, validate_photo_meal


class LocalStore:
    read_only = False

    def __init__(self, root: Path | None = None):
        self.root = Path(root or SETTINGS.lakehouse_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, pid: str, name: str) -> Path:
        d = self.root / pid
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.parquet"

    def pids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if (p / "minute.parquet").exists())

    def users(self) -> list:
        """Users from bodylab/users.json whose participant is prepared on this laptop."""
        from bodylab.users import load_users

        prepared = set(self.pids())
        return [u for u in load_users() if u.pid in prepared]

    def write_inputs(self, pid: str, minute: pd.DataFrame, meals: pd.DataFrame) -> None:
        minute.to_parquet(self._p(pid, "minute"), index=False)
        meals.to_parquet(self._p(pid, "meals"), index=False)

    def read_inputs(self, pid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        meals = pd.read_parquet(self._p(pid, "meals"))
        photo_path = self._p(pid, "photo_meals")
        photos = pd.read_parquet(photo_path) if photo_path.exists() else pd.DataFrame()
        return pd.read_parquet(self._p(pid, "minute")), combine_meals(meals, photos)

    def write_meal(self, pid: str, row: dict) -> None:
        row = validate_photo_meal(pid, row)
        path = self._p(pid, "photo_meals")
        photos = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        photos = pd.concat([photos, pd.DataFrame([row])], ignore_index=True)
        photos.drop_duplicates("meal_id", keep="last").to_parquet(path, index=False)

    def write_state(self, pid: str, features: dict[str, pd.DataFrame], notebook: Notebook, until: pd.Timestamp | None, agent: str) -> None:
        for lab, df in features.items():
            df.to_parquet(self._p(pid, f"features_{lab}"), index=False)
        for name, df in notebook.to_frames().items():
            df.to_parquet(self._p(pid, f"nb_{name}"), index=False)
        (self.root / pid / "state.json").write_text(json.dumps({"until": str(until) if until is not None else None, "agent": agent}))

    def read_state(self, pid: str) -> tuple[dict[str, pd.DataFrame], Notebook, pd.Timestamp | None, str]:
        features = {}
        for lab in LABS:
            p = self._p(pid, f"features_{lab}")
            features[lab] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        frames = {}
        for name in TABLES:
            p = self._p(pid, f"nb_{name}")
            frames[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        nb = Notebook.from_frames(pid, frames)
        state_path = self.root / pid / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        until = pd.Timestamp(state["until"]) if state.get("until") else None
        return features, nb, until, state.get("agent", "")

    def reset_state(self, pid: str) -> None:
        for p in (self.root / pid).glob("*.parquet"):
            if p.stem.startswith(("features_", "nb_")):
                p.unlink()
        (self.root / pid / "state.json").unlink(missing_ok=True)


def _plain_times(df: pd.DataFrame) -> pd.DataFrame:
    """The warehouse returns some timestamps tagged UTC and others untagged, depending on how each table was made.
    The pipeline uses plain times throughout (as in local mode), and pandas can't compare the two kinds, so drop
    the tag (values are UTC wall-clock times, which is how they were written)."""
    for c in df.columns:
        if isinstance(df[c].dtype, pd.DatetimeTZDtype):
            df[c] = df[c].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


class DatabricksSqlStore:
    """Read-only pipeline results plus reviewed photo-meal writes, for the Databricks App.

    Needs DATABRICKS_WAREHOUSE_ID (set as an app resource) and BODYLAB_CATALOG / BODYLAB_SCHEMA.
    Authentication uses the app's service principal through the Databricks SDK config.
    """

    read_only = True

    PARALLEL_READS = 4  # connections used to fetch one user's ~13 tables at once

    def __init__(self):
        import threading

        self.catalog = os.getenv("BODYLAB_CATALOG", "workspace")
        self.schema = os.getenv("BODYLAB_SCHEMA", "body_lab")
        self._lock = threading.Lock()
        self._idle = [self._connect()]  # small pool, so parallel reads each get their own connection
        self._executor = None
        self._photo_table_ready = False

    @staticmethod
    def _connect():
        from databricks import sql
        from databricks.sdk.core import Config

        cfg = Config()
        return sql.connect(
            server_hostname=cfg.host.replace("https://", ""),
            http_path=f"/sql/1.0/warehouses/{os.environ['DATABRICKS_WAREHOUSE_ID']}",
            credentials_provider=lambda: cfg.authenticate,
        )

    def _execute(self, statement: str, fetch: bool, parameters: dict | None = None):
        """Run one statement on a pooled connection; a dropped session (warehouse slept) gets one reconnect."""
        with self._lock:
            conn = self._idle.pop() if self._idle else None
        conn = conn or self._connect()
        try:
            for attempt in range(2):
                try:
                    with conn.cursor() as cur:
                        if parameters is None:
                            cur.execute(statement)
                        else:
                            cur.execute(statement, parameters=parameters)
                        return _plain_times(cur.fetchall_arrow().to_pandas()) if fetch else None
                except Exception as exc:
                    if attempt or "TABLE_OR_VIEW_NOT_FOUND" in str(exc):  # a missing table is a real answer
                        raise
                    try:
                        conn.close()
                    except Exception:
                        pass
                    conn = self._connect()
        finally:
            with self._lock:
                self._idle.append(conn)

    def _q(self, table: str, pid: str | None = None) -> pd.DataFrame:
        where = f" WHERE pid = '{pid}'" if pid and pid.isalnum() else ""
        return self._execute(f"SELECT * FROM {self.catalog}.{self.schema}.{table}{where}", fetch=True)

    def _parallel(self, fn, items: list) -> list:
        from concurrent.futures import ThreadPoolExecutor

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.PARALLEL_READS)
        return list(self._executor.map(fn, items))

    def pids(self) -> list[str]:
        return sorted(self._q("agent_state")["pid"].unique().tolist())

    def users(self) -> list:
        """Sign-in list from the `users` table, independent of agent results (falls back to bodylab/users.json)."""
        from bodylab.users import User, load_users

        try:
            df = self._q("users")
        except Exception:
            return load_users()
        if df.empty:
            return load_users()
        return [User(username=r.username, name=r.name, pid=r.pid, email=r.email or "")
                for r in df.sort_values("pid").itertuples()]

    def sync_users(self, users: list) -> None:
        """Create `users` if needed and upsert every user from bodylab/users.json (no passwords are stored)."""
        def lit(v: str) -> str:
            return "'" + str(v or "").replace("\\", "\\\\").replace("'", "\\'") + "'"

        table = f"{self.catalog}.{self.schema}.users"
        values = ", ".join(f"({lit(u.username)}, {lit(u.name)}, {lit(u.pid)}, {lit(u.email)})" for u in users)
        self._run(f"CREATE TABLE IF NOT EXISTS {table} (username STRING, name STRING, pid STRING, email STRING)")
        self._run(f"MERGE INTO {table} t USING (SELECT * FROM VALUES {values} AS s(username, name, pid, email)) s "
                  "ON t.username = s.username WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *")

    def _run(self, statement: str) -> None:
        self._execute(statement, fetch=False)

    def signature(self) -> str:
        """One small query that changes whenever the agent saves new results for anyone."""
        try:
            df = self._q("agent_state")
        except Exception:
            return ""
        if df.empty:
            return "empty"
        stamp = df["updated_at"] if "updated_at" in df else df["until"]
        return "|".join(f"{p}@{s}" for p, s in sorted(zip(df["pid"], stamp.astype(str))))

    def read_inputs(self, pid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        minute, meals, photos = self._parallel(
            lambda t: self._photo_meals(pid) if t == "photo_meals" else self._q(t, pid),
            ["live_minute", "live_meals", "photo_meals"])
        return minute, combine_meals(meals, photos)

    def _photo_meals(self, pid: str) -> pd.DataFrame:
        try:
            return self._q("photo_meals", pid)
        except Exception as exc:
            if "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                return pd.DataFrame()
            raise

    def write_meal(self, pid: str, row: dict) -> None:
        row = validate_photo_meal(pid, row)
        table = f"{self.catalog}.{self.schema}.photo_meals"
        if not self._photo_table_ready:
            try:
                self._execute(f"SELECT meal_id FROM {table} LIMIT 0", fetch=True)
            except Exception as exc:
                if "TABLE_OR_VIEW_NOT_FOUND" not in str(exc):
                    raise
                self._run(f"CREATE TABLE IF NOT EXISTS {table} ({PHOTO_MEALS_SCHEMA}) USING DELTA")
            self._photo_table_ready = True
        select = ", ".join(f"CAST(:{name} AS {('TIMESTAMP' if name in ('ts', 'uploaded_at') else 'BOOLEAN' if name == 'carbs_missing' else 'DOUBLE' if name in ('carbs', 'sugar', 'fiber', 'protein', 'fat', 'calories') else 'STRING')}) AS {name}" for name in PHOTO_COLUMNS)
        self._execute(f"MERGE INTO {table} t USING (SELECT {select}) s "
                      "ON t.pid = s.pid AND t.meal_id = s.meal_id "
                      "WHEN MATCHED THEN UPDATE SET * WHEN NOT MATCHED THEN INSERT *", fetch=False, parameters=row)

    def read_state(self, pid: str):
        tables = [f"features_{lab}" for lab in LABS] + [f"nb_{name}" for name in TABLES] + ["agent_state"]
        got = dict(zip(tables, self._parallel(lambda t: self._safe(t, pid), tables)))  # ~12 tables, 4 at a time
        features = {lab: got[f"features_{lab}"] for lab in LABS}
        frames = {name: got[f"nb_{name}"] for name in TABLES}
        state = got["agent_state"]
        until = pd.Timestamp(state["until"].iloc[0]) if len(state) else None
        agent = state["agent"].iloc[0] if len(state) else ""
        return features, Notebook.from_frames(pid, frames), until, agent

    def _safe(self, table: str, pid: str) -> pd.DataFrame:
        try:
            return self._q(table, pid)
        except Exception:
            return pd.DataFrame()


def open_store():
    if os.getenv("BODYLAB_SOURCE", "local").lower() == "databricks":
        return DatabricksSqlStore()
    return LocalStore()
