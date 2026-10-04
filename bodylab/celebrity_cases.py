"""Curated celebrity-inspired leads and local quest tracking for the demo.

These scenarios do not describe the athletes' actual habits. Completing a quest
counts observations; it does not confirm a health claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bodylab.config import SETTINGS

CASES = (
    ("messi", "Lionel Messi", "fuel", "steps_after", ">=", "meal-time walking", "Do meals with more walking afterward go with a smaller glucose rise?"),
    ("serena", "Serena Williams", "movement", "prev_sleep_h", ">=", "sleep and movement", "Does a longer night's sleep go with a lower walk heart rate?"),
    ("ronaldo", "Cristiano Ronaldo", "sleep", "late_steps", "<=", "evening activity", "Do quieter evenings go with earlier sleep?"),
    ("biles", "Simone Biles", "stress", "prev_sleep_h", ">=", "rest and stress", "Does a longer night's sleep go with less daytime stress?"),
    ("phelps", "Michael Phelps", "movement", "prev_sleep_h", ">=", "recovery", "Does a longer night's sleep go with a lower walk heart rate?"),
    ("federer", "Roger Federer", "fuel", "steps_before", ">=", "pre-meal movement", "Do meals with more walking beforehand go with a smaller glucose rise?"),
    ("nadal", "Rafael Nadal", "movement", "hour", ">=", "walk timing", "Do later walks go with a lower walk heart rate?"),
    ("kohli", "Virat Kohli", "fuel", "steps_before", ">=", "meal-time movement", "Do meals with more walking beforehand go with a smaller glucose rise?"),
)


def suggest(features: dict[str, pd.DataFrame], now: pd.Timestamp) -> list[dict]:
    """Use recorded situations only; no guessed distance or universal target."""
    leads = []
    for key, celebrity, lab, feature, op, theme, question in CASES:
        df = features.get(lab, pd.DataFrame())
        if df.empty or feature not in df or "end_ts" not in df:
            continue
        history = df[pd.to_datetime(df["end_ts"]) <= now]
        if "good_data" in history:
            history = history[history["good_data"].fillna(False).astype(bool)]
        values = pd.to_numeric(history[feature], errors="coerce")
        values = values[np.isfinite(values)]
        if feature in ("steps_before", "steps_after", "prev_sleep_h"):
            values = values[values > 0]
        if len(values) < 3:
            continue
        threshold = float(values.median())
        if feature.startswith("steps") or feature == "late_steps":
            threshold = float(round(threshold / 10) * 10)
            target = f'{"At least" if op == ">=" else "At most"} {threshold:,.0f} steps '
            target += {"steps_before": "in the hour before a meal", "steps_after": "in the hour after a meal", "late_steps": "after 21:00"}[feature]
        elif feature == "prev_sleep_h":
            target = f"Observe situations after at least {threshold:.1f} hours of recorded sleep"
        else:
            hour, minute = divmod(round(threshold * 60), 60)
            target = f"Observe walks starting at or after {hour:02d}:{minute:02d}"
        leads.append(dict(case_id=key, celebrity=celebrity, lab=lab, feature=feature, op=op,
                          threshold=threshold, theme=theme, question=question, target_text=target,
                          basis=f"Based on the median of {len(values)} recorded situations."))
    return leads


def accept(quests: list[dict], lead: dict, now: pd.Timestamp) -> bool:
    if any(q["case_id"] == lead["case_id"] for q in quests):
        return False
    quests.append({**lead, "started_at": now.isoformat(), "target": 3, "progress": 0,
                   "done": False, "observations": []})
    return True


def update(quests: list[dict], features: dict[str, pd.DataFrame], now: pd.Timestamp) -> None:
    for q in quests:
        df = features.get(q["lab"], pd.DataFrame())
        if now < pd.Timestamp(q["started_at"]):
            q["started_at"] = now.isoformat()
            q["observations"] = []
        # A rewind must not retain observations from the future.
        q["observations"] = [o for o in q["observations"] if pd.Timestamp(o["ts"]) <= now]
        seen = {o["sid"] for o in q["observations"]}
        if not df.empty:
            rows = df[(pd.to_datetime(df["end_ts"]) > pd.Timestamp(q["started_at"])) & (pd.to_datetime(df["end_ts"]) <= now)]
            for _, row in rows.sort_values("end_ts").iterrows():
                value = row.get(q["feature"], np.nan)
                response = row.get("response", np.nan)
                if row["sid"] in seen or not bool(row.get("good_data", True)) or not np.isfinite(value) or not np.isfinite(response):
                    continue
                matches = value >= q["threshold"] if q["op"] == ">=" else value <= q["threshold"]
                if matches and len(q["observations"]) < q["target"]:
                    q["observations"].append(dict(sid=str(row["sid"]), ts=pd.Timestamp(row["end_ts"]).isoformat(), response=float(response)))
                    seen.add(row["sid"])
        q["progress"] = len(q["observations"])
        q["done"] = q["progress"] >= q["target"]


def _path(pid: str, root: Path | None) -> Path:
    if not pid.isalnum():
        raise ValueError("Invalid participant ID")
    return (root or SETTINGS.lakehouse_dir) / "celebrity_quests" / f"{pid}.json"


def load(pid: str, root: Path | None = None) -> list[dict]:
    path = _path(pid, root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save(pid: str, quests: list[dict], root: Path | None = None) -> None:
    path = _path(pid, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(quests, indent=2), encoding="utf-8")
    temporary.replace(path)
