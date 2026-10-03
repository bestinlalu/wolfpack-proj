"""Read BIG IDEAs participant files into tidy per-minute signals and grouped meals.

Raw layout (PhysioNet big-ideas-glycemic-wearable 1.1.3):
    <raw_dir>/<pid>/Dexcom_<pid>.csv, ACC_<pid>.csv, EDA_<pid>.csv, HR_<pid>.csv,
    TEMP_<pid>.csv, Food_Log_<pid>.csv ; <raw_dir>/Demographics.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from bodylab.config import SETTINGS

CHUNK_ROWS = 2_000_000
ACC_UNITS_PER_G = 64.0  # Empatica E4 accelerometer: 1/64 g per unit
ACC_HZ = 32


def parse_ts(values: pd.Series) -> pd.Series:
    sample = str(values.dropna().iloc[0]) if values.notna().any() else ""
    if "/" in sample:
        fmt = "%m/%d/%y %H:%M:%S" if sample.count(":") == 2 else "%m/%d/%y %H:%M"
        return pd.to_datetime(values, format=fmt, errors="coerce")
    return pd.to_datetime(values, format="ISO8601", errors="coerce")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lstrip("﻿").lower() for c in df.columns]
    return df


def _read_chunks(path: Path) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, chunksize=CHUNK_ROWS, encoding="utf-8-sig", skipinitialspace=True):
        chunk = _clean_columns(chunk)
        chunk["datetime"] = parse_ts(chunk["datetime"])
        yield chunk.dropna(subset=["datetime"])


def acc_minute(chunk: pd.DataFrame) -> pd.DataFrame:
    """Per-minute movement from raw accelerometer samples.

    enmo_mg: mean Euclidean norm minus one g (milli-g), a standard movement intensity measure.
    steps: estimated steps, counted as acceleration peaks after removing the 1-second mean.
    Returns partial sums so chunks can be combined.
    """
    mag = np.sqrt(chunk["acc_x"] ** 2 + chunk["acc_y"] ** 2 + chunk["acc_z"] ** 2) / ACC_UNITS_PER_G
    enmo = np.clip(mag - 1.0, 0, None) * 1000.0
    hp = mag - mag.rolling(ACC_HZ, center=True, min_periods=1).mean()
    is_peak = (hp > 0.12) & (hp == hp.rolling(9, center=True, min_periods=1).max())
    minute = chunk["datetime"].dt.floor("min")
    out = pd.DataFrame({"ts": minute, "enmo_sum": enmo, "n": 1, "steps": is_peak.astype(int)})
    return out.groupby("ts", as_index=False).sum()


def _mean_minute(chunk: pd.DataFrame, col: str) -> pd.DataFrame:
    out = pd.DataFrame({"ts": chunk["datetime"].dt.floor("min"), f"{col}_sum": pd.to_numeric(chunk[col], errors="coerce")})
    out["n"] = out[f"{col}_sum"].notna().astype(int)
    return out.groupby("ts", as_index=False).sum()


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["ts"])
    return pd.concat(parts).groupby("ts", as_index=False).sum()


def _per_minute(path: Path, fn: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
    return _combine([fn(c) for c in _read_chunks(path)])


def load_glucose(path: Path) -> pd.DataFrame:
    df = _clean_columns(pd.read_csv(path, encoding="utf-8-sig"))
    ts_col = next(c for c in df.columns if c.startswith("timestamp"))
    val_col = next(c for c in df.columns if c.startswith("glucose value"))
    type_col = next(c for c in df.columns if c.startswith("event type"))
    df = df[df[type_col] == "EGV"]
    out = pd.DataFrame({"ts": parse_ts(df[ts_col]), "glucose": pd.to_numeric(df[val_col], errors="coerce")})
    return out.dropna().sort_values("ts").reset_index(drop=True)


def load_food_log(path: Path) -> pd.DataFrame:
    df = _clean_columns(pd.read_csv(path, encoding="utf-8-sig"))
    df["ts"] = parse_ts(df["time_begin"].fillna(df["date"].astype(str) + " " + df["time"].astype(str)))
    for c in ["calorie", "total_carb", "dietary_fiber", "sugar", "protein", "total_fat"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def group_meals(food: pd.DataFrame, pid: str, group_min: int | None = None) -> pd.DataFrame:
    """Food log rows within `group_min` of the previous item form one meal."""
    group_min = group_min or SETTINGS.signals.meal_group_min
    if food.empty:
        return pd.DataFrame(columns=["pid", "meal_id", "ts", "carbs", "sugar", "fiber", "protein", "fat", "calories", "items", "carbs_missing"])
    food = food.sort_values("ts")
    new_meal = food["ts"].diff().dt.total_seconds().fillna(1e9) > group_min * 60
    food = food.assign(meal_no=new_meal.cumsum())
    meals = food.groupby("meal_no").agg(
        ts=("ts", "min"),
        carbs=("total_carb", "sum"),
        carbs_missing=("total_carb", lambda s: bool(s.isna().all())),
        sugar=("sugar", "sum"),
        fiber=("dietary_fiber", "sum"),
        protein=("protein", "sum"),
        fat=("total_fat", "sum"),
        calories=("calorie", "sum"),
        items=("logged_food", lambda s: ", ".join(str(x) for x in s if pd.notna(x))),
    ).reset_index(drop=True)
    meals.insert(0, "pid", pid)
    meals.insert(1, "meal_id", [f"{pid}-m{i:03d}" for i in range(len(meals))])
    return meals


def load_participant(pid: str, raw_dir: Path | None = None, offset_hours: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (minute signals, meals) for one participant from raw files."""
    raw_dir = Path(raw_dir or SETTINGS.raw_dir) / pid
    offset = pd.Timedelta(hours=SETTINGS.wrist_offset_hours if offset_hours is None else offset_hours)

    acc = _per_minute(raw_dir / f"ACC_{pid}.csv", acc_minute)
    acc["enmo_mg"] = acc["enmo_sum"] / acc["n"]
    acc = acc[["ts", "enmo_mg", "steps"]]

    frames = [acc]
    for name in ["hr", "eda", "temp"]:
        part = _per_minute(raw_dir / f"{name.upper()}_{pid}.csv", lambda c, col=name: _mean_minute(c, col))
        part[name] = part[f"{name}_sum"] / part["n"].replace(0, np.nan)
        frames.append(part[["ts", name]])

    wrist = frames[0]
    for f in frames[1:]:
        wrist = wrist.merge(f, on="ts", how="outer")
    wrist["ts"] = wrist["ts"] + offset

    glucose = load_glucose(raw_dir / f"Dexcom_{pid}.csv")
    glucose["ts"] = glucose["ts"].dt.floor("min")
    glucose = glucose.groupby("ts", as_index=False)["glucose"].mean()

    minute = build_minute_frame(pid, wrist, glucose)
    meals = group_meals(load_food_log(raw_dir / f"Food_Log_{pid}.csv"), pid)
    return minute, meals


def build_minute_frame(pid: str, wrist: pd.DataFrame, glucose: pd.DataFrame) -> pd.DataFrame:
    """Regular one-minute grid from first to last sample; glucose stays sparse (every 5 minutes)."""
    start = min(wrist["ts"].min(), glucose["ts"].min())
    end = max(wrist["ts"].max(), glucose["ts"].max())
    grid = pd.DataFrame({"ts": pd.date_range(start.floor("min"), end.floor("min"), freq="min")})
    out = grid.merge(wrist, on="ts", how="left").merge(glucose, on="ts", how="left")
    out.insert(0, "pid", pid)
    return finalize_minute(out)


def finalize_minute(df: pd.DataFrame) -> pd.DataFrame:
    cfg = SETTINGS.data_check
    df = df.sort_values("ts").reset_index(drop=True)
    df["steps"] = df["steps"].fillna(0).clip(upper=200)
    df["worn"] = (df["temp"] >= cfg.min_worn_temp_c) & (df["eda"] >= cfg.min_worn_eda_us)
    return df


def load_demographics(raw_dir: Path | None = None) -> pd.DataFrame:
    path = Path(raw_dir or SETTINGS.raw_dir) / "Demographics.csv"
    df = _clean_columns(pd.read_csv(path, encoding="utf-8-sig"))
    df["pid"] = df["id"].astype(int).map(lambda i: f"{i:03d}")
    return df
