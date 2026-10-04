"""The loader handles the real file quirks: Dexcom metadata rows, BOM + short dates in HR, spaced headers."""
from pathlib import Path

import numpy as np
import pandas as pd

from bodylab.data import loader

DEXCOM = """Index,Timestamp (YYYY-MM-DDThh:mm:ss),Event Type,Event Subtype,Patient Info,Device Info,Source Device ID,Glucose Value (mg/dL),Insulin Value (u),Carb Value (grams),Duration (hh:mm:ss),Glucose Rate of Change (mg/dL/min),Transmitter Time (Long Integer)
1,,FirstName,,2019,,,,,,,,
5,,Device,,,Dexcom G6 Mobile App,iPhone G6,,,,,,
7,,Alert,High,,,iPhone G6,200.0,,,,,
13,2020-02-13 17:23:32,EGV,,,,iPhone G6,61.0,,,,,11101.0
14,2020-02-13 17:28:32,EGV,,,,iPhone G6,59.0,,,,,11401.0
15,2020-02-13 17:33:32,EGV,,,,iPhone G6,Low,,,,,11701.0
"""

FOOD = """date,time,time_begin,time_end,logged_food,amount,unit,searched_food,calorie,total_carb,dietary_fiber,sugar,protein,total_fat
2020-02-13,20:30:00,2020-02-13 20:30:00,,Chicken Leg,1.0,,chicken leg,475.0,0.0,0.0,0.0,62.0,23.0
2020-02-13,20:30:00,2020-02-13 20:30:00,,Asparagus,4.0,,Asparagus,13.0,2.5,1.2,0.8,1.4,0.1
2020-02-14,07:10:00,2020-02-14 07:10:00,,"Cereal, frosted",0.75,cup,x,110.0,26.0,,10.0,1.0,
"""


def _write(tmp: Path, pid: str) -> Path:
    d = tmp / pid
    d.mkdir(parents=True)
    (d / f"Dexcom_{pid}.csv").write_text(DEXCOM)
    (d / f"Food_Log_{pid}.csv").write_text(FOOD)
    (d / f"HR_{pid}.csv").write_text("﻿datetime, hr\n2/13/20 17:23,94\n2/13/20 17:23,90\n2/13/20 17:24,80\n", encoding="utf-8")
    times = pd.date_range("2020-02-13 17:23:00", periods=4 * 120, freq="250ms")
    pd.DataFrame({"datetime": times.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3], " eda": 0.3}).to_csv(d / f"EDA_{pid}.csv", index=False)
    pd.DataFrame({"datetime": times.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3], " temp": 32.0}).to_csv(d / f"TEMP_{pid}.csv", index=False)
    acc_t = pd.date_range("2020-02-13 17:23:00", periods=32 * 120, freq="31250us")
    still = pd.DataFrame({"datetime": acc_t.strftime("%Y-%m-%d %H:%M:%S.%f"), " acc_x": 0.0, " acc_y": 0.0, " acc_z": 64.0})
    still.to_csv(d / f"ACC_{pid}.csv", index=False)
    return tmp


def test_glucose_keeps_only_readings(tmp_path):
    _write(tmp_path, "001")
    g = loader.load_glucose(tmp_path / "001" / "Dexcom_001.csv")
    assert list(g["glucose"]) == [61.0, 59.0]
    assert g["ts"].iloc[0] == pd.Timestamp("2020-02-13 17:23:32")


def test_food_rows_group_into_meals(tmp_path):
    _write(tmp_path, "001")
    meals = loader.group_meals(loader.load_food_log(tmp_path / "001" / "Food_Log_001.csv"), "001")
    assert len(meals) == 2
    assert meals["carbs"].iloc[0] == 2.5
    assert "Asparagus" in meals["items"].iloc[0]


def test_participant_minute_frame(tmp_path):
    _write(tmp_path, "001")
    minute, meals = loader.load_participant("001", tmp_path, offset_hours=0)
    row = minute[minute["ts"] == pd.Timestamp("2020-02-13 17:23")].iloc[0]
    assert row["hr"] == 92.0
    assert row["glucose"] == 61.0
    assert abs(row["enmo_mg"]) < 1e-6 and row["steps"] == 0
    assert row["worn"]
    assert len(meals) == 2


def test_parse_ts_handles_both_formats():
    a = loader.parse_ts(pd.Series(["2/13/20 15:29"]))
    b = loader.parse_ts(pd.Series(["2020-02-13 15:28:50.250"]))
    assert a.iloc[0] == pd.Timestamp("2020-02-13 15:29")
    assert b.iloc[0].microsecond == 250000


def test_acc_steps_counts_peaks():
    t = pd.date_range("2020-01-01", periods=32 * 60, freq="31250us")
    wave = 64 + 20 * np.sin(2 * np.pi * 1.8 * np.arange(len(t)) / 32)  # 1.8 steps per second
    df = pd.DataFrame({"datetime": t, "acc_x": 0.0, "acc_y": 0.0, "acc_z": wave})
    out = loader.acc_minute(df)
    assert 90 <= out["steps"].sum() <= 125


def test_food_log_column_check(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text(FOOD)
    headerless = tmp_path / "Food_Log_003.csv"
    headerless.write_text("2020-02-22,10:30:00,2020-02-22 10:30:00,Chicken Nuggets,8.0,piece,x,393.0,19.0,0.1,20.0\n")
    renamed = tmp_path / "Food_Log_007.csv"
    renamed.write_text(FOOD.replace("date,time,", "date,time_of_day,", 1))
    assert loader.food_log_is_standard(good)
    assert not loader.food_log_is_standard(headerless)
    assert not loader.food_log_is_standard(renamed)
