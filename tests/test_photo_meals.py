import threading

import pandas as pd
import pytest

from bodylab.agent.tools import _similarity
from bodylab.data.synthetic import generate
from bodylab.meal_log import combine_meals, historical_meals_changed, validate_photo_meal
from bodylab.pipeline.features import Signals, build_meals, build_nights
from bodylab.store import DatabricksSqlStore, LocalStore


def photo_row():
    return dict(pid="S01", meal_id="S01-photo-abc", ts=pd.Timestamp("2020-02-13 12:00"),
                carbs=42, carbs_missing=False, sugar=5, fiber=3, protein=12, fat=7,
                calories=300, items="Chef's salad", source="photo", confidence="low",
                uploaded_at=pd.Timestamp("2026-10-04 10:00"))


def test_local_photo_save_is_idempotent_and_preserves_replay(tmp_path):
    minute, meals = generate(days=3)
    store = LocalStore(tmp_path)
    store.write_inputs("S01", minute, meals)
    row = photo_row()
    store.write_meal("S01", row)
    store.write_meal("S01", row)
    _, combined = store.read_inputs("S01")
    saved = combined[combined.meal_id == row["meal_id"]]
    assert len(saved) == 1 and saved.iloc[0]["confidence"] == "low"
    assert len(pd.read_parquet(tmp_path / "S01" / "meals.parquet")) == len(meals)
    assert len(pd.read_parquet(tmp_path / "S01" / "photo_meals.parquet")) == 1


def test_sql_save_uses_bound_values_and_does_not_write_pipeline_tables():
    calls = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, statement, parameters=None): calls.append((statement, parameters))
    class Connection:
        def cursor(self): return Cursor()
    store = object.__new__(DatabricksSqlStore)
    store.catalog, store.schema = "workspace", "body_lab"
    store._photo_table_ready = True
    store._lock = threading.Lock()
    store._idle = [Connection()]
    store.write_meal("S01", photo_row())
    statement, params = calls[0]
    assert "MERGE INTO workspace.body_lab.photo_meals" in statement
    assert ":items" in statement and "Chef's salad" not in statement
    assert params["items"] == "Chef's salad" and params["confidence"] == "low"
    assert "live_meals" not in statement and "live_minute" not in statement


def test_photo_validation_rejects_invalid_participant_and_nutrition():
    with pytest.raises(ValueError): validate_photo_meal("S02", photo_row())
    with pytest.raises(ValueError): validate_photo_meal("S01", {**photo_row(), "carbs": float("nan")})
    with pytest.raises(ValueError): validate_photo_meal("S01", {**photo_row(), "carbs": -1})


def test_photo_replaces_same_minute_and_triggers_historical_reanalysis():
    replay = pd.DataFrame([{**photo_row(), "meal_id": "S01-original", "carbs": 20, "source": "log"}])
    combined = combine_meals(replay, pd.DataFrame([photo_row()]))
    assert len(combined) == 1 and combined.iloc[0].carbs == 42
    through = pd.Timestamp("2020-02-13 16:00")
    assert historical_meals_changed(replay, combined, through)
    assert not historical_meals_changed(combined, combined, through)
    future = pd.DataFrame([{**photo_row(), "meal_id": "S01-photo-next", "ts": through + pd.Timedelta(hours=1)}])
    assert not historical_meals_changed(combined, combine_meals(combined, future), through)


def test_features_keep_photo_metadata_and_wait_for_glucose_window():
    minute, meals = generate(days=3)
    row = photo_row()
    meals = combine_meals(meals, pd.DataFrame([row]))
    before = Signals(minute, row["ts"] + pd.Timedelta(minutes=119))
    assert row["meal_id"] not in build_meals(before, meals, build_nights(before, meals)).meal_id.values
    after = Signals(minute, row["ts"] + pd.Timedelta(minutes=120))
    feature = build_meals(after, meals, build_nights(after, meals))
    photo = feature[feature.meal_id == row["meal_id"]].iloc[0]
    assert photo.source == "photo" and photo.confidence == "low"
    assert pd.notna(photo.response)


def test_photo_comparisons_allow_estimation_uncertainty():
    ev = pd.Series(dict(carbs=40, slot="lunch", hour=12, source="log"))
    candidates = pd.DataFrame([dict(carbs=70, slot="lunch", hour=12, source=source) for source in ("log", "photo")])
    scores = _similarity("fuel", ev, candidates)
    assert pd.isna(scores.iloc[0]) and pd.notna(scores.iloc[1])
