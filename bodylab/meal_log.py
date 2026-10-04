"""Photo meal storage contract and combination with replay meal logs."""
from __future__ import annotations

import math

import pandas as pd

PHOTO_MEALS_SCHEMA = """pid STRING, meal_id STRING, ts TIMESTAMP, carbs DOUBLE,
    carbs_missing BOOLEAN, sugar DOUBLE, fiber DOUBLE, protein DOUBLE, fat DOUBLE,
    calories DOUBLE, items STRING, source STRING, confidence STRING, uploaded_at TIMESTAMP"""
PHOTO_COLUMNS = ["pid", "meal_id", "ts", "carbs", "carbs_missing", "sugar", "fiber",
                 "protein", "fat", "calories", "items", "source", "confidence", "uploaded_at"]


def validate_photo_meal(pid: str, row: dict) -> dict:
    if not pid.isalnum() or row.get("pid") != pid:
        raise ValueError("Meal must belong to the signed-in participant.")
    result = {key: row[key] for key in PHOTO_COLUMNS}
    if not str(result["meal_id"]).startswith(f"{pid}-photo-"):
        raise ValueError("Invalid photo meal ID.")
    if result["source"] != "photo" or result["confidence"] not in ("low", "medium", "high"):
        raise ValueError("Photo source and confidence are required.")
    for name in ("ts", "uploaded_at"):
        stamp = pd.Timestamp(result[name])
        if pd.isna(stamp) or stamp.tzinfo is not None:
            raise ValueError("Meal timestamps must be valid local replay times.")
        result[name] = stamp.to_pydatetime()
    for name in ("carbs", "sugar", "fiber", "protein", "fat", "calories"):
        value = float(result[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Nutrition estimates must be finite, non-negative numbers.")
        result[name] = value
    result["carbs_missing"] = False
    result["items"] = str(result["items"])
    return result


def combine_meals(replay: pd.DataFrame, photos: pd.DataFrame) -> pd.DataFrame:
    """A reviewed photo replaces a replay meal at the same participant/minute.

    One meal per minute matches the engine's situation IDs. The latest upload
    wins if a meal is re-photographed; the original rows remain in storage.
    """
    if photos.empty:
        return replay.copy()
    photos = photos.sort_values("uploaded_at")
    combined = pd.concat([replay, photos], ignore_index=True)
    combined["ts"] = pd.to_datetime(combined["ts"]).dt.floor("min")
    return combined.drop_duplicates(["pid", "ts"], keep="last").sort_values("ts").reset_index(drop=True)


def meal_fingerprint(meals: pd.DataFrame) -> int:
    if meals.empty:
        return 0
    ordered = meals.reindex(columns=PHOTO_COLUMNS).sort_values(["ts", "meal_id"]).reset_index(drop=True)
    ordered["source"] = ordered["source"].fillna("log")
    ordered["confidence"] = ordered["confidence"].fillna("")
    normalized = ordered.astype(object).where(pd.notna(ordered), None).astype(str)
    return int(pd.util.hash_pandas_object(normalized, index=False).sum())


def historical_meals_changed(previous: pd.DataFrame, current: pd.DataFrame, through: pd.Timestamp) -> bool:
    def history(df):
        return df[pd.to_datetime(df["ts"]) <= through] if not df.empty else df
    return meal_fingerprint(history(previous)) != meal_fingerprint(history(current))
