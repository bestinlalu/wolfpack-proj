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
    ("phelps", "Michael Phelps", "fuel", "prev_sleep_h", ">=", "sleep and meals", "Does a longer night's sleep go with a smaller glucose rise?"),
    ("federer", "Roger Federer", "fuel", "steps_before", ">=", "pre-meal movement", "Do meals with more walking beforehand go with a smaller glucose rise?"),
    ("nadal", "Rafael Nadal", "movement", "hour", ">=", "walk timing", "Do later walks go with a lower walk heart rate?"),
    ("kohli", "Virat Kohli", "sleep", "dinner_gap_h", ">=", "dinner timing", "Does a longer gap between dinner and bed go with earlier sleep?"),
    ("bolt", "Usain Bolt", "movement", "stress_before", "<=", "calm before moving", "Does a calmer few hours before a walk go with a lower walk heart rate?"),
    ("kipchoge", "Eliud Kipchoge", "stress", "steps_earlier", ">=", "active days", "Do more active days go with a calmer stress signal?"),
    ("djokovic", "Novak Djokovic", "movement", "hour", "<=", "walk timing", "Do earlier walks go with a lower walk heart rate?"),
    ("lebron", "LeBron James", "movement", "since_last_meal_h", ">=", "meal and movement timing", "Does leaving more time after a meal go with a lower walk heart rate?"),
    ("osaka", "Naomi Osaka", "movement", "temp_before", "<=", "cool starts", "Does cooler skin before a walk go with a lower walk heart rate?"),
    ("curry", "Stephen Curry", "fuel", "hour", "<=", "meal timing", "Do earlier meals go with a smaller glucose rise?"),
    ("mbappe", "Kylian Mbappé", "fuel", "since_last_meal_h", "<=", "regular meals", "Do shorter gaps between meals go with a smaller glucose rise?"),
)

# Which way the matching hypothesis points: +1 means more of the factor goes with a higher response.
CASE_DIRECTIONS = {key: -1 for key, *_ in CASES}
CASE_DIRECTIONS.update(ronaldo=1, bolt=1, djokovic=1, osaka=1, curry=1, mbappe=1)



def _plain(value) -> pd.Timestamp:
    """Quests saved earlier may hold UTC-tagged times; the pipeline compares plain times."""
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts


def _plain_col(values: pd.Series) -> pd.Series:
    ts = pd.to_datetime(values)
    return ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts

def _same_investigation(q: dict, h: dict) -> bool:
    if q.get("hyp_id") == h["hyp_id"]:
        return True
    # Also recognize quests saved before hypotheses were attached to leads.
    return (q.get("lab") == h["lab"] and q.get("feature") == h["factor"]
            and q.get("direction", CASE_DIRECTIONS.get(q.get("case_id"))) == h["direction"])


def suggest(features: dict[str, pd.DataFrame], now: pd.Timestamp,
            hypotheses: list[dict], quests: list[dict] = ()) -> list[dict]:
    """Use recorded situations only; no guessed distance or universal target."""
    leads = []
    for key, celebrity, lab, feature, op, theme, question in CASES:
        matching = [h for h in hypotheses if h["status"] in ("testing", "fading")
                    and h["lab"] == lab and h["factor"] == feature
                    and h["direction"] == CASE_DIRECTIONS[key]
                    and (pd.isna(h.get("opened_at")) or _plain(h["opened_at"]) <= now)]
        if not matching or any(_same_investigation(q, h) for q in quests for h in matching):
            continue
        hypothesis = matching[0]
        df = features.get(lab, pd.DataFrame())
        if df.empty or feature not in df or "end_ts" not in df:
            continue
        history = df[_plain_col(df["end_ts"]) <= now]
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
            target += {"steps_before": "in the hour before a meal", "steps_after": "in the hour after a meal", "late_steps": "after 21:00",
                       "steps_earlier": "earlier in the day"}[feature]
        elif feature == "prev_sleep_h":
            target = f"Observe situations after at least {threshold:.1f} hours of recorded sleep"
        elif feature == "dinner_gap_h":
            target = f"Observe nights with at least {threshold:.1f} hours between dinner and sleep"
        elif feature == "stress_before":
            target = "Observe walks after a calmer-than-usual few hours (stress signal at or below your typical level)"
        elif feature == "temp_before":
            target = f"Observe walks that start with skin at or below {threshold:.1f} °C"
        elif feature == "since_last_meal_h":
            target = (f"Observe walks at least {threshold:.1f} hours after a meal" if lab == "movement"
                      else f"Observe meals at most {threshold:.1f} hours after the previous one")
        else:
            hour, minute = divmod(round(threshold * 60), 60)
            what = "walks" if lab == "movement" else "meals"
            target = f"Observe {what} starting at or {'after' if op == '>=' else 'before'} {hour:02d}:{minute:02d}"
        leads.append(dict(case_id=key, celebrity=celebrity, lab=lab, feature=feature, op=op,
                          hyp_id=hypothesis["hyp_id"], direction=CASE_DIRECTIONS[key],
                          threshold=threshold, theme=theme, question=question, target_text=target,
                          basis=f"Based on the median of {len(values)} recorded situations."))
    return leads


def accept(quests: list[dict], lead: dict, now: pd.Timestamp) -> bool:
    hypothesis = dict(hyp_id=lead["hyp_id"], lab=lead["lab"], factor=lead["feature"], direction=lead["direction"])
    if any(_same_investigation(q, hypothesis) for q in quests):
        return False
    quests.append({**lead, "started_at": now.isoformat(), "target": 3, "progress": 0,
                   "done": False, "observations": []})
    return True


def update(quests: list[dict], features: dict[str, pd.DataFrame], now: pd.Timestamp) -> None:
    for q in quests:
        df = features.get(q["lab"], pd.DataFrame())
        if now < _plain(q["started_at"]):
            q["started_at"] = now.isoformat()
            q["observations"] = []
        # A rewind must not retain observations from the future.
        q["observations"] = [o for o in q["observations"] if _plain(o["ts"]) <= now]
        seen = {o["sid"] for o in q["observations"]}
        if not df.empty:
            rows = df[(_plain_col(df["end_ts"]) > _plain(q["started_at"])) & (_plain_col(df["end_ts"]) <= now)]
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
