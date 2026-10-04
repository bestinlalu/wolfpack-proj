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

    def write_inputs(self, pid: str, minute: pd.DataFrame, meals: pd.DataFrame) -> None:
        minute.to_parquet(self._p(pid, "minute"), index=False)
        meals.to_parquet(self._p(pid, "meals"), index=False)

    def read_inputs(self, pid: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.read_parquet(self._p(pid, "minute")), pd.read_parquet(self._p(pid, "meals"))

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


class DatabricksSqlStore:
    """Read-only view of the Delta tables, for the Databricks App.

    Needs DATABRICKS_WAREHOUSE_ID (set as an app resource) and BODYLAB_CATALOG / BODYLAB_SCHEMA.
    Authentication uses the app's service principal through the Databricks SDK config.
    """

    read_only = True

    def __init__(self):
        self.catalog = os.getenv("BODYLAB_CATALOG", "workspace")
        self.schema = os.getenv("BODYLAB_SCHEMA", "body_lab")
        self.conn = self._connect()

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

    def _q(self, table: str, pid: str | None = None) -> pd.DataFrame:
        where = f" WHERE pid = '{pid}'" if pid and pid.isalnum() else ""
        query = f"SELECT * FROM {self.catalog}.{self.schema}.{table}{where}"
        for attempt in range(2):
            try:
                with self.conn.cursor() as cur:
                    cur.execute(query)
                    return cur.fetchall_arrow().to_pandas()
            except Exception as exc:
                # A missing table is a real answer; a dropped session (warehouse slept) gets one reconnect.
                if attempt or "TABLE_OR_VIEW_NOT_FOUND" in str(exc):
                    raise
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = self._connect()

    def pids(self) -> list[str]:
        return sorted(self._q("agent_state")["pid"].unique().tolist())

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
        return self._q("live_minute", pid), self._q("live_meals", pid)

    def read_state(self, pid: str):
        features = {lab: self._safe(f"features_{lab}", pid) for lab in LABS}
        frames = {name: self._safe(f"nb_{name}", pid) for name in TABLES}
        state = self._safe("agent_state", pid)
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
