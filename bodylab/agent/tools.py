"""Tools the investigating agent can call. Every tool returns JSON-safe dicts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bodylab.agent import checks
from bodylab.agent.notebook import Notebook
from bodylab.config import SETTINGS
from bodylab.formatting import fmt
from bodylab.labs import LABS, cause
from bodylab.pipeline.features import Signals, robust_scale

FLOORS = {"steps": 500.0, "z": 0.3, "h": 0.75, "clock": 0.75, "mg/dL": 5.0, "°C": 0.3, "flag": 0.5}


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return None if not np.isfinite(f) else round(f, 2)


@dataclass
class ToolContext:
    pid: str
    features: dict[str, pd.DataFrame]
    sig: Signals
    meals: pd.DataFrame
    notebook: Notebook
    now: pd.Timestamp

    def situation(self, sid: str) -> tuple[str, pd.Series]:
        lab = sid.split("-")[1]
        df = self.features[lab]
        return lab, df[df["sid"] == sid].iloc[0]

    def history(self, lab: str, before: pd.Timestamp) -> pd.DataFrame:
        df = self.features[lab]
        return df[df["end_ts"] <= before]


def describe(lab: str, row: pd.Series) -> dict:
    base = {"sid": row["sid"], "lab": lab, "start": str(row["ts"]), "weekday": row["ts"].day_name(),
            "response": _num(row["response"]), "expected": _num(row.get("expected")), "z": _num(row.get("z")),
            "response_label": LABS[lab].response_label, "unit": LABS[lab].response_unit,
            "response_text": fmt(_num(row["response"]), LABS[lab].response_unit),
            "usual_text": fmt(_num(row.get("expected")), LABS[lab].response_unit)}
    if lab == "fuel":
        base.update(items=row["items"], carbs=_num(row["carbs"]), slot=row["slot"])
    if lab == "movement":
        base.update(duration_min=int(row["duration_min"]), cadence=_num(row["cadence"]))
    if lab == "sleep":
        base.update(duration_h=_num(row["duration_h"]), wake=str(row["end_ts"]))
    if lab == "stress":
        base.update(hour=int(row["hour"]), after_meal=bool(row["after_meal"]))
    return base


def check_data_quality(ctx: ToolContext, situation_id: str) -> dict:
    lab, row = ctx.situation(situation_id)
    result = checks.check_situation(ctx.sig, lab, row, ctx.meals)
    return {"situation": situation_id, "passed": checks.passed(result), "checks": result}


def _similarity(lab: str, ev: pd.Series, cand: pd.DataFrame) -> pd.Series:
    if lab == "fuel":
        d = (cand["carbs"] - ev["carbs"]).abs()
        score = d / 20 + (cand["slot"] != ev["slot"]) * 1.0 + (cand["hour"] - ev["hour"]).abs() / 6
        return score.where(d <= 25)
    if lab == "movement":
        d = (cand["cadence"] - ev["cadence"]).abs()
        score = d / 10 + (cand["duration_min"] - ev["duration_min"]).abs() / 15
        return score.where(d <= 15)
    if lab == "stress":
        same = (cand["hour"] == ev["hour"]) & (cand["ts"].dt.normalize() < ev["ts"].normalize())
        score = (cand["after_meal"] != ev["after_meal"]) * 2 + (cand["weekend"] != ev["weekend"]) * 0.5
        return score.where(same)
    return pd.Series(0.0, index=cand.index)


def find_similar_situations(ctx: ToolContext, situation_id: str, limit: int = 5) -> dict:
    lab, ev = ctx.situation(situation_id)
    hist = ctx.history(lab, ev["ts"]).dropna(subset=["response"])
    hist = hist[hist["sid"] != situation_id]
    if hist.empty:
        return {"situation": situation_id, "similar": []}
    hist = hist.assign(score=_similarity(lab, ev, hist)).dropna(subset=["score"]).sort_values("score")
    if lab == "fuel":
        hist = hist[hist["slot"] == ev["slot"]]
    good = []
    for _, r in hist.iterrows():
        if checks.passed(checks.check_situation(ctx.sig, lab, r, ctx.meals)):
            good.append(describe(lab, r) | {"similarity": _num(r["score"])})
        if len(good) >= limit:
            break
    return {"situation": situation_id, "similar": good}


def personal_normal(ctx: ToolContext, lab: str, column: str, before: pd.Timestamp, row: pd.Series | None = None) -> tuple[float, float]:
    """Typical value and spread of a column, within the same context when there is enough of it."""
    if column not in ctx.features[lab]:
        hist = np.array([])
    else:
        df = ctx.history(lab, before)
        if row is not None and lab == "fuel" and (df["slot"] == row["slot"]).sum() >= SETTINGS.detection.min_same_slot:
            df = df[df["slot"] == row["slot"]]
        if row is not None and lab == "stress" and (df["hour"] == row["hour"]).sum() >= 3:
            df = df[df["hour"] == row["hour"]]
        hist = df[column].to_numpy(float)
    hist = hist[np.isfinite(hist)]
    unit = cause(lab, column).unit if column != "response" else "z"
    if not len(hist):
        return np.nan, FLOORS.get(unit, 1.0)
    return float(np.median(hist)), robust_scale(hist, FLOORS.get(unit, 1.0))


def get_personal_normal(ctx: ToolContext, lab: str, signal: str) -> dict:
    med, scale = personal_normal(ctx, lab, signal, ctx.now)
    return {"lab": lab, "signal": signal, "median": _num(med), "typical_spread": _num(scale)}


def compare_situations(ctx: ToolContext, situation_id: str, similar_ids: list[str]) -> dict:
    """Rank what was different in this situation versus similar past ones, in units of personal spread."""
    lab, ev = ctx.situation(situation_id)
    sims = ctx.features[lab][ctx.features[lab]["sid"].isin(similar_ids)]
    if sims.empty:
        return {"situation": situation_id, "error": "no similar situations given"}
    # Contrast with similar situations that went the other way (or stayed normal).
    sign = np.sign(ev["z"]) if np.isfinite(ev["z"]) and ev["z"] != 0 else 1
    z = sims["z"].fillna(0)
    contrast = sims[z * sign < 0]
    if len(contrast) < 2:
        contrast = sims[z.abs() < 0.8] if (z.abs() < 0.8).sum() >= 2 else sims
    sims = contrast
    typical = sims.iloc[0]
    cfg = SETTINGS.detection
    diffs = []
    for c in LABS[lab].causes:
        if c.key not in ev or not np.isfinite(ev[c.key]):
            continue
        group = sims[c.key].to_numpy(float)
        group = group[np.isfinite(group)]
        if not len(group):
            continue
        _, scale = personal_normal(ctx, lab, c.key, ev["ts"], ev)
        dz = (ev[c.key] - float(np.median(group))) / scale
        level = "very unusual" if abs(dz) >= cfg.very_unusual_z else "somewhat" if abs(dz) >= cfg.somewhat_z else "normal"
        diffs.append({"factor": c.key, "label": c.label, "unit": c.unit, "this_time": _num(ev[c.key]),
                      "comparison": _num(typical[c.key]), "similar_median": _num(np.median(group)),
                      "this_time_text": fmt(_num(ev[c.key]), c.unit), "typical_text": fmt(_num(np.median(group)), c.unit),
                      "difference_z": _num(dz), "level": level, "meal_related": c.meal_related})
    diffs.sort(key=lambda d: -abs(d["difference_z"] or 0))
    return {"situation": describe(lab, ev), "comparison": describe(lab, typical), "n_similar": int(len(sims)),
            "response_direction": "higher" if (ev["z"] or 0) > 0 else "lower", "differences": diffs}


def open_hypothesis(ctx: ToolContext, situation_id: str, factor: str, direction: int) -> dict:
    """Open (or support) a hypothesis that `factor` moves the response. direction=+1: higher factor, higher response."""
    lab, ev = ctx.situation(situation_id)
    valid = {c.key for c in LABS[lab].causes}
    if factor not in valid:
        return {"error": f"'{factor}' is not a candidate cause for the {lab} lab", "allowed": sorted(valid)}
    event = next((e for e in ctx.notebook.events if e["sid"] == situation_id), None)
    if event is None:
        return {"error": "open a hypothesis only from an investigated surprise"}
    direction = 1 if int(direction) >= 0 else -1
    opposite = [h for h in ctx.notebook.active(lab) if h["factor"] == factor and h["direction"] == -direction]
    if opposite:
        h = opposite[0]
        ctx.notebook._add_evidence(h, situation_id, "contradicts", ctx.now, event.get("z", np.nan), np.nan)
        return {"conflict": h["hyp_id"], "claim": h["claim"],
                "note": "This points the opposite way to an existing hypothesis, so it was counted as evidence against it. Close the case as unexplained."}
    med, _ = personal_normal(ctx, lab, factor, ev["ts"], ev)
    side = int(np.sign(ev[factor] - med)) if np.isfinite(med) and np.isfinite(ev[factor]) and ev[factor] != med else 1
    h, note = ctx.notebook.open_hypothesis(event, lab, factor, direction, ctx.now, side)
    event["hyp_id"] = h["hyp_id"]
    return {"hypothesis": h["hyp_id"], "claim": h["claim"], "note": note, "supports": h["supports"],
            "status": h["status"], "needed": SETTINGS.hypothesis.confirm_supports,
            "more_tests_needed": max(SETTINGS.hypothesis.confirm_supports - h["supports"], 0) if h["status"] == "testing" else 0}


def list_hypotheses(ctx: ToolContext, lab: str | None = None) -> dict:
    return {"hypotheses": [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in h.items() if k in ("hyp_id", "lab", "factor", "direction", "claim", "status", "supports", "contradicts")}
                           for h in ctx.notebook.active(lab)]}


TOOL_SPECS = [
    {"name": "check_data_quality", "description": "Run the data checks (sensor gaps, warm-up, impossible jumps, false lows, overlapping meals, wrong meal time, wristband off) for a situation. Always call this first; if it fails, close the case as bad_data.",
     "parameters": {"type": "object", "properties": {"situation_id": {"type": "string"}}, "required": ["situation_id"]}},
    {"name": "find_similar_situations", "description": "Find this person's earlier situations in the same lab that are most similar (similar carbs and time for meals, similar pace for walks, same hour for stress). Bad-data situations are excluded.",
     "parameters": {"type": "object", "properties": {"situation_id": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["situation_id"]}},
    {"name": "compare_situations", "description": "Rank what was different about this situation versus the similar ones, scored against the person's own day-to-day spread (difference_z). |difference_z| >= 1.5 is a clear difference.",
     "parameters": {"type": "object", "properties": {"situation_id": {"type": "string"}, "similar_ids": {"type": "array", "items": {"type": "string"}}}, "required": ["situation_id", "similar_ids"]}},
    {"name": "get_personal_normal", "description": "The person's typical value (median and spread) for a lab's response or one of its candidate causes.",
     "parameters": {"type": "object", "properties": {"lab": {"type": "string", "enum": list(LABS)}, "signal": {"type": "string"}}, "required": ["lab", "signal"]}},
    {"name": "list_hypotheses", "description": "Hypotheses already open or confirmed, so you can add support instead of duplicating.",
     "parameters": {"type": "object", "properties": {"lab": {"type": "string", "enum": list(LABS)}}}},
    {"name": "open_hypothesis", "description": "Record the top clear difference as a hypothesis (a lead, not a finding). direction=+1 if a higher factor goes with a higher response, -1 if a higher factor goes with a lower response.",
     "parameters": {"type": "object", "properties": {"situation_id": {"type": "string"}, "factor": {"type": "string"}, "direction": {"type": "integer", "enum": [-1, 1]}}, "required": ["situation_id", "factor", "direction"]}},
    {"name": "close_case", "description": "Finish the investigation. verdict: bad_data, unexplained, or lead. title: for a lead, a headline under 14 words that leads with the why (e.g. \"Thursday's lunch spiked higher with fewer steps before eating\"). message: 1-2 sentences using only numbers returned by the tools, ending with how many tests are still needed. Never claim causation or give medical advice.",
     "parameters": {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["bad_data", "unexplained", "lead"]}, "title": {"type": "string"}, "message": {"type": "string"}}, "required": ["verdict", "message"]}},
]

TOOL_FUNCS = {
    "check_data_quality": check_data_quality,
    "find_similar_situations": find_similar_situations,
    "compare_situations": compare_situations,
    "get_personal_normal": get_personal_normal,
    "list_hypotheses": list_hypotheses,
    "open_hypothesis": open_hypothesis,
}
