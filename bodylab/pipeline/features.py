"""Situations per lab (meals, still hours, nights, walks) with responses, causes and expectations.

Only data up to `until` is used, and a situation appears only once its window has finished.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bodylab.config import SETTINGS

MIN = pd.Timedelta(minutes=1)


def robust_scale(x: np.ndarray, floor: float) -> float:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return floor
    return max(1.4826 * float(np.median(np.abs(x - np.median(x)))), floor)


@dataclass
class Signals:
    """Per-minute arrays for fast window lookups."""

    minute: pd.DataFrame
    until: pd.Timestamp

    def __post_init__(self) -> None:
        m = self.minute[self.minute["ts"] <= self.until].reset_index(drop=True)
        self.df = m
        self.t0 = m["ts"].iloc[0] if len(m) else self.until
        self.ts = m["ts"].values
        self.steps = m["steps"].to_numpy(float)
        self.enmo = m["enmo_mg"].to_numpy(float)
        self.hr = m["hr"].to_numpy(float)
        self.eda = m["eda"].to_numpy(float)
        self.temp = m["temp"].to_numpy(float)
        self.glucose = m["glucose"].to_numpy(float)
        self.worn = m["worn"].to_numpy(bool)
        self.steps_cum = np.concatenate([[0.0], np.cumsum(self.steps)])
        cfg = SETTINGS.signals
        self.still = self.worn & (self.enmo < cfg.still_enmo_mg) & (self.steps < 10)
        awake_still = self.still & (pd.Series(self.ts).dt.hour.between(8, 21).values)
        hr_ref = self.hr[awake_still & ~np.isnan(self.hr)]
        eda_ref = self.eda[awake_still & ~np.isnan(self.eda)]
        self.rest_hr = float(np.median(hr_ref)) if len(hr_ref) else 65.0
        hr_scale = robust_scale(hr_ref, 2.0)
        eda_med = float(np.median(eda_ref)) if len(eda_ref) else 0.3
        eda_scale = robust_scale(eda_ref, 0.1)
        idx = ((self.eda - eda_med) / eda_scale + (self.hr - self.rest_hr) / hr_scale) / 2
        self.stress = np.where(self.still, idx, np.nan)
        enmo_roll = pd.Series(self.enmo).rolling(30, center=True, min_periods=10).median().to_numpy()
        hr_roll = pd.Series(self.hr).rolling(30, center=True, min_periods=10).mean().to_numpy()
        self.sleep_like = (enmo_roll < cfg.sleep_enmo_mg) & ((hr_roll < self.rest_hr - cfg.sleep_hr_drop) | np.isnan(hr_roll))

    def pos(self, t: pd.Timestamp) -> int:
        return int((t - self.t0) / MIN)

    def window(self, arr: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        a, b = max(self.pos(start), 0), min(self.pos(end), len(arr))
        return arr[a:b] if b > a else np.array([])

    def steps_between(self, start: pd.Timestamp, end: pd.Timestamp) -> float:
        a, b = max(self.pos(start), 0), min(self.pos(end), len(self.steps))
        return float(self.steps_cum[b] - self.steps_cum[a]) if b > a else 0.0

    def mean(self, arr: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> float:
        w = self.window(arr, start, end)
        w = w[~np.isnan(w)]
        return float(w.mean()) if len(w) else np.nan


def _runs(mask: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    """Start/end index pairs of True runs, bridging gaps of up to `max_gap` False values."""
    runs: list[tuple[int, int]] = []
    start = None
    gap = 0
    for i, v in enumerate(mask):
        if v:
            if start is None:
                start = i
            gap = 0
            end = i + 1
        elif start is not None:
            gap += 1
            if gap > max_gap:
                runs.append((start, end))
                start = None
                gap = 0
    if start is not None:
        runs.append((start, end))
    return runs


def build_nights(sig: Signals, meals: pd.DataFrame) -> pd.DataFrame:
    cfg = SETTINGS.signals
    rows = []
    if not len(sig.ts):
        return pd.DataFrame()
    first_day = pd.Timestamp(sig.ts[0]).normalize()
    last_day = sig.until.normalize()
    for day in pd.date_range(first_day, last_day, freq="D"):
        start = day + pd.Timedelta(hours=cfg.night_start_hour)
        end = day + pd.Timedelta(days=1, hours=cfg.night_end_hour)
        if end > sig.until or start < pd.Timestamp(sig.ts[0]):
            continue
        a, b = sig.pos(start), sig.pos(end)
        runs = [r for r in _runs(sig.sleep_like[a:b], cfg.sleep_max_interrupt_min) if r[1] - r[0] >= cfg.sleep_min_minutes]
        if not runs:
            continue
        s, e = max(runs, key=lambda r: r[1] - r[0])
        onset, wake = start + s * MIN, start + e * MIN
        dinner = meals[(meals["ts"] < onset) & (meals["ts"] > onset - pd.Timedelta(hours=8))]
        rows.append({
            "sid": f"{sig.df['pid'].iloc[0]}-sleep-{day:%Y%m%d}",
            "lab": "sleep", "ts": onset, "end_ts": wake, "night_of": day,
            "response": (onset - start) / MIN,
            "duration_h": (wake - onset) / pd.Timedelta(hours=1),
            "restless_min": int((sig.enmo[a + s:a + e] >= cfg.still_enmo_mg).sum()),
            "sleep_hr": sig.mean(sig.hr, onset, wake),
            "late_steps": sig.steps_between(day + pd.Timedelta(hours=21), onset),
            "dinner_gap_h": (onset - dinner["ts"].max()) / pd.Timedelta(hours=1) if len(dinner) else np.nan,
            "evening_glucose": sig.mean(sig.glucose, onset - pd.Timedelta(hours=3), onset),
            "day_stress": sig.mean(sig.stress, day + pd.Timedelta(hours=12), day + pd.Timedelta(hours=18)),
            "night_temp": sig.mean(sig.temp, onset, wake),
        })
    return pd.DataFrame(rows)


def prev_sleep_hours(nights: pd.DataFrame, t: pd.Timestamp) -> float:
    if nights.empty:
        return np.nan
    done = nights[nights["end_ts"] <= t]
    done = done[done["end_ts"] > t - pd.Timedelta(hours=20)]
    return float(done.iloc[-1]["duration_h"]) if len(done) else np.nan


def _slot(t: pd.Timestamp) -> str:
    h = t.hour + t.minute / 60
    if h < 10.5:
        return "breakfast"
    if h < 15:
        return "lunch"
    if h < 17:
        return "snack"
    return "dinner"


def build_meals(sig: Signals, meals: pd.DataFrame, nights: pd.DataFrame) -> pd.DataFrame:
    win = pd.Timedelta(minutes=SETTINGS.signals.fuel_window_min)
    rows = []
    meals = meals.sort_values("ts").reset_index(drop=True)
    for i, m in meals.iterrows():
        t = m["ts"]
        if not len(sig.ts) or t + win > sig.until or t - pd.Timedelta(hours=1) < pd.Timestamp(sig.ts[0]):
            continue
        start_g = sig.mean(sig.glucose, t - pd.Timedelta(minutes=15), t + MIN)
        after = sig.window(sig.glucose, t, t + win)
        peak = float(np.nanmax(after)) if np.isfinite(after).any() else np.nan
        prev = meals.loc[: i - 1, "ts"] if i > 0 else pd.Series([], dtype="datetime64[ns]")
        rows.append({
            "sid": f"{m['pid']}-fuel-{t:%Y%m%d%H%M}",
            "lab": "fuel", "ts": t, "end_ts": t + win, "meal_id": m["meal_id"], "items": m["items"],
            "carbs": float(m["carbs"]) if pd.notna(m["carbs"]) else np.nan,
            "carbs_missing": bool(m.get("carbs_missing", False)),
            "slot": _slot(t),
            "start_glucose": start_g, "peak_glucose": peak,
            "response": peak - start_g if np.isfinite(peak) and np.isfinite(start_g) else np.nan,
            "steps_before": sig.steps_between(t - pd.Timedelta(hours=1), t),
            "steps_after": sig.steps_between(t, t + pd.Timedelta(hours=1)),
            "stress_before": sig.mean(sig.stress, t - pd.Timedelta(minutes=45), t),
            "hour": t.hour + t.minute / 60,
            "since_last_meal_h": (t - prev.iloc[-1]) / pd.Timedelta(hours=1) if len(prev) else np.nan,
            "prev_sleep_h": prev_sleep_hours(nights, t),
        })
    return pd.DataFrame(rows)


def build_walks(sig: Signals, meals: pd.DataFrame, nights: pd.DataFrame) -> pd.DataFrame:
    cfg = SETTINGS.signals
    walking = sig.steps >= cfg.walk_steps_per_min
    rows = []
    for s, e in _runs(walking, cfg.walk_max_gap_min):
        if e - s < cfg.walk_min_minutes:
            continue
        start, end = sig.t0 + s * MIN, sig.t0 + e * MIN
        if end + pd.Timedelta(hours=1) > sig.until:
            continue
        hr_during = sig.hr[s + 3:e]
        hr_during = hr_during[~np.isnan(hr_during)]
        if len(hr_during) < 5:
            continue
        after = sig.hr[e:e + 60]
        settled = np.where(after <= sig.rest_hr + 5)[0]
        before_meals = meals[meals["ts"] < start]
        rows.append({
            "sid": f"{sig.df['pid'].iloc[0]}-movement-{start:%Y%m%d%H%M}",
            "lab": "movement", "ts": start, "end_ts": end,
            "duration_min": e - s, "cadence": float(sig.steps[s:e].mean()),
            "response": float(hr_during.mean()),
            "recovery_min": int(settled[0]) if len(settled) else 60,
            "prev_sleep_h": prev_sleep_hours(nights, start),
            "stress_before": sig.mean(sig.stress, start - pd.Timedelta(hours=3), start),
            "hour": start.hour + start.minute / 60,
            "since_last_meal_h": (start - before_meals["ts"].max()) / pd.Timedelta(hours=1) if len(before_meals) else np.nan,
            "temp_before": sig.mean(sig.temp, start - pd.Timedelta(minutes=30), start),
        })
    return pd.DataFrame(rows)


def build_stress_hours(sig: Signals, meals: pd.DataFrame, nights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not len(sig.ts):
        return pd.DataFrame()
    pid = sig.df["pid"].iloc[0]
    block = pd.Timedelta(hours=2)
    first = pd.Timestamp(sig.ts[0]).normalize()
    blocks = [d + pd.Timedelta(hours=hh) for d in pd.date_range(first, sig.until.normalize(), freq="D") for hh in range(8, 22, 2)]
    for h in blocks:
        if h < pd.Timestamp(sig.ts[0]) or h + block > sig.until:
            continue
        w = sig.window(sig.stress, h, h + block)
        if np.isfinite(w).sum() < 30:
            continue
        recent = meals[(meals["ts"] <= h) & (meals["ts"] > h - pd.Timedelta(hours=2))]
        day = h.normalize()
        rows.append({
            "sid": f"{pid}-stress-{h:%Y%m%d%H}",
            "lab": "stress", "ts": h, "end_ts": h + block,
            "response": float(np.nanmean(w)),
            "hour": h.hour, "weekend": int(h.dayofweek >= 5), "after_meal": int(len(recent) > 0),
            "prev_sleep_h": prev_sleep_hours(nights, h),
            "steps_earlier": sig.steps_between(day + pd.Timedelta(hours=5), h),
        })
    return pd.DataFrame(rows)


def add_expectations(df: pd.DataFrame, lab: str) -> pd.DataFrame:
    """Expected response from the person's own earlier situations, then a robust z-score."""
    if df.empty:
        return df.assign(expected=np.nan, resid=np.nan, z=np.nan, history=0)
    df = df.sort_values("ts").reset_index(drop=True)
    min_hist = SETTINGS.detection.history_needed(lab)
    expected, history = [], []
    for i, row in df.iterrows():
        hist = df.iloc[:i]
        hist = hist[hist["end_ts"] <= row["ts"]].dropna(subset=["response"])
        history.append(len(hist))
        expected.append(_expected(hist, row, lab) if len(hist) >= min_hist else np.nan)
    df["expected"] = expected
    df["history"] = history
    df["resid"] = df["response"] - df["expected"]
    floors = {"fuel": 6.0, "stress": 0.5, "sleep": 15.0, "movement": 3.0}
    z = []
    for i, row in df.iterrows():
        prior = df.iloc[:i]
        prior = prior[prior["end_ts"] <= row["ts"]]["resid"].to_numpy(float)
        z.append(row["resid"] / robust_scale(prior, floors[lab]) if np.isfinite(row["resid"]) else np.nan)
    df["z"] = z
    return df


def _expected(hist: pd.DataFrame, row: pd.Series, lab: str) -> float:
    if lab == "fuel":
        # Compare within the same meal slot: a lunch is judged against this person's lunches.
        hist = hist.dropna(subset=["carbs"])
        hist = hist[hist["slot"] == row["slot"]]
        if len(hist) < SETTINGS.detection.min_same_slot:
            return np.nan
        if np.isfinite(row["carbs"]) and hist["carbs"].std() > 3:
            slope, intercept = np.polyfit(hist["carbs"], hist["response"], 1)
            return float(intercept + slope * row["carbs"])
        return float(hist["response"].median())
    if lab == "movement":
        # Known effect kept in the baseline: digestion raises heart rate for about 90 minutes after eating.
        recent = (hist["since_last_meal_h"].fillna(9) < 1.5).astype(float)
        if len(hist) >= 4 and hist["cadence"].std() > 0:
            X = np.column_stack([np.ones(len(hist)), hist["cadence"], recent])
            coef, *_ = np.linalg.lstsq(X, hist["response"].to_numpy(float), rcond=None)
            r = float(np.nan_to_num(row["since_last_meal_h"], nan=9) < 1.5)
            return float(coef[0] + coef[1] * row["cadence"] + coef[2] * r)
        return float(hist["response"].median())
    if lab == "stress":
        same = hist[(hist["hour"] == row["hour"]) & (hist["ts"].dt.normalize() < row["ts"].normalize())]
        if same["ts"].dt.normalize().nunique() < 3:
            return np.nan
        matched = same[same["after_meal"] == row["after_meal"]]
        same = matched if len(matched) >= 2 else same
        return float(same["response"].median())
    return float(hist["response"].median())


def build_all(minute: pd.DataFrame, meals: pd.DataFrame, until: pd.Timestamp) -> dict[str, pd.DataFrame]:
    sig = Signals(minute, until)
    meals = meals[meals["ts"] <= until]
    nights = build_nights(sig, meals)
    out = {
        "fuel": build_meals(sig, meals, nights),
        "stress": build_stress_hours(sig, meals, nights),
        "sleep": nights,
        "movement": build_walks(sig, meals, nights),
    }
    for lab, df in out.items():
        if not df.empty:
            df.insert(0, "pid", minute["pid"].iloc[0])
        out[lab] = add_expectations(df, lab)
    return out
