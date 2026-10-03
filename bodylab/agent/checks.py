"""Data checks that run before any comparison. A failed check closes the case as bad data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bodylab.config import SETTINGS
from bodylab.pipeline.features import Signals


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "passed": bool(ok), "detail": detail}


def glucose_checks(sig: Signals, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    cfg = SETTINGS.data_check
    out = []
    a, b = max(sig.pos(start), 0), min(sig.pos(end), len(sig.glucose))
    g = sig.glucose[a:b]
    t = np.arange(a, b)[~np.isnan(g)]
    vals = g[~np.isnan(g)]
    if len(t) < 2:
        return [_check("Glucose gaps", False, "Almost no glucose readings in this window")]
    gaps = np.diff(np.concatenate([[a], t, [b]]))
    worst = int(gaps.max()) - 5
    out.append(_check("Glucose gaps", worst <= cfg.max_glucose_gap_min, f"Longest gap {max(worst, 0)} min (limit {cfg.max_glucose_gap_min})"))

    first = np.where(~np.isnan(sig.glucose))[0]
    if len(first):
        all_t = first
        new_sensor = all_t[np.concatenate([[True], np.diff(all_t) > 120])]
        warm = any(0 <= a - s < cfg.sensor_warmup_hours * 60 for s in new_sensor)
        out.append(_check("Sensor warm-up", not warm, "Within 24 h of a new sensor" if warm else "Sensor settled"))

    step = np.abs(np.diff(vals)) / np.maximum(np.diff(t) / 5, 1)
    jump = float(step.max()) if len(step) else 0.0
    out.append(_check("Impossible jumps", jump <= cfg.max_jump_mg_dl_per_5min, f"Largest change {jump:.0f} mg/dL per 5 min"))

    drop = False
    for i in range(1, len(vals) - 2):
        if vals[i - 1] - vals[i] >= cfg.false_low_drop_mg_dl and vals[min(i + 3, len(vals) - 1)] - vals[i] >= cfg.false_low_drop_mg_dl * 0.8:
            drop = True
            break
    out.append(_check("False lows", not drop, "Sharp drop with quick rebound" if drop else "No sensor-compression pattern"))
    return out


def worn_check(sig: Signals, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    w = sig.window(sig.worn.astype(float), start, end)
    frac = float(w.mean()) if len(w) else 0.0
    return _check("Wristband worn", frac >= 0.8, f"Worn {frac:.0%} of the time")


def check_situation(sig: Signals, lab: str, row: pd.Series, meals: pd.DataFrame) -> list[dict]:
    cfg = SETTINGS.data_check
    if lab == "fuel":
        t = row["ts"]
        checks = glucose_checks(sig, t - pd.Timedelta(minutes=15), row["end_ts"])
        others = meals[(meals["ts"] != t) & ((meals["ts"] - t).abs() <= pd.Timedelta(minutes=cfg.overlap_window_min)) & (meals["carbs"].fillna(0) >= 10)]
        checks.append(_check("Overlapping meals", others.empty, "No other meal within 2 h" if others.empty else f"{len(others)} other meal(s) within 2 h"))
        before = sig.mean(sig.glucose, t - pd.Timedelta(minutes=35), t - pd.Timedelta(minutes=25))
        at = sig.mean(sig.glucose, t - pd.Timedelta(minutes=5), t + pd.Timedelta(minutes=5))
        early = np.isfinite(before) and np.isfinite(at) and at - before >= cfg.early_rise_mg_dl
        checks.append(_check("Meal time", not early, "Glucose rose before the logged time" if early else "Rise starts after the logged time"))
        checks.append(_check("Meal log complete", not row.get("carbs_missing", False) and np.isfinite(row.get("carbs", np.nan)), "Carbs logged" if np.isfinite(row.get("carbs", np.nan)) else "Carbs missing"))
        return checks
    checks = [worn_check(sig, row["ts"], row["end_ts"])]
    hr = sig.window(sig.hr, row["ts"], row["end_ts"])
    cover = float(np.isfinite(hr).mean()) if len(hr) else 0.0
    checks.append(_check("Heart rate coverage", cover >= 0.7, f"Heart rate present {cover:.0%} of the time"))
    return checks


def passed(checks: list[dict]) -> bool:
    return all(c["passed"] for c in checks)
