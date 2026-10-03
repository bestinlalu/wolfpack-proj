"""Synthetic participant with planted effects, in the same shape as `loader.load_participant`.

Planted effects the agent should find:
- Fuel: skipping the usual walk before a meal makes the glucose spike about 65% bigger (a walk shrinks it by about 40%).
- Fuel (cross-lab): eating during a stress peak raises the spike by about 35%.
- Movement: after nights under 6 hours, walk heart rate runs about 12 bpm higher.
- Sleep: activity after 21:30 delays estimated sleep onset by about 50 minutes.
- Stress: the stress signal peaks on weekday afternoons (14:00-16:00).
Planted data problems: a glucose gap after one lunch, a false night-time low, daily band-off showers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bodylab.data.loader import finalize_minute

START = pd.Timestamp("2020-02-13 00:00")  # a Thursday


def _curve(t: np.ndarray, peak_min: float = 45.0) -> np.ndarray:
    t = np.clip(t, 0, None) / peak_min
    return t ** 2 * np.exp(2 * (1 - t))


def generate(pid: str = "S01", days: int = 14, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n = days * 1440
    ts = pd.date_range(START, periods=n, freq="min")
    minute_of_day = np.arange(n) % 1440
    day_idx = np.arange(n) // 1440
    weekday = ts.dayofweek.values

    asleep = np.zeros(n, bool)
    walking = np.zeros(n, bool)
    cadence = np.zeros(n)
    stress = np.zeros(n)
    band_off = np.zeros(n, bool)
    walk_hr_extra = np.zeros(n)
    meals: list[dict] = []

    def idx(day: int, hhmm: float) -> int:
        return int(day * 1440 + hhmm)

    short_nights = {3, 5, 7}  # nights starting on these days are short
    late_activity_days = {1, 4, 6, 8}
    prev_sleep_hours = 7.5
    for d in range(days):
        # walks
        if rng.random() < 0.7:
            s = idx(d, 8 * 60 + 30 + rng.integers(-5, 6))
            walking[s:s + 14] = True
        lunch = 12 * 60 + 30 + int(rng.integers(-30, 31))
        pre_lunch_walk = rng.random() < 0.6
        if pre_lunch_walk:
            s = idx(d, lunch - 40)
            walking[s:s + 22] = True
        evening_walk = rng.random() < 0.6
        if evening_walk:
            s = idx(d, 19 * 60 + rng.integers(0, 20))
            walking[s:s + 30] = True
        if d in late_activity_days:
            s = idx(d, 21 * 60 + 40)
            walking[s:s + 25] = True

        # stress: weekday afternoons plus a random episode
        if weekday[d * 1440] < 5:
            stress[idx(d, 14 * 60):idx(d, 16 * 60)] = 1.0
        ep = idx(d, int(rng.integers(9 * 60, 18 * 60)))
        stress[ep:ep + 40] = np.maximum(stress[ep:ep + 40], 0.7)

        # band off for a shower
        s = idx(d, 7 * 60 + 5)
        band_off[s:s + 20] = True

        # meals
        breakfast = 7 * 60 + 35 + int(rng.integers(-15, 16))
        meals.append({"ts": breakfast, "day": d, "carbs": float(rng.integers(35, 61)), "items": "Oatmeal with banana", "slot": "breakfast"})
        if d % 2 == 0 or rng.random() < 0.3:
            meals.append({"ts": lunch, "day": d, "carbs": float(62 + rng.integers(-3, 4)), "items": "Pasta with tomato sauce", "slot": "lunch"})
        else:
            meals.append({"ts": lunch, "day": d, "carbs": float(rng.choice([22, 45])), "items": "Turkey sandwich" if rng.random() < 0.5 else "Spinach salad", "slot": "lunch"})
        if rng.random() < 0.4:
            snack = 15 * 60 + int(rng.integers(-20, 30))
            meals.append({"ts": snack, "day": d, "carbs": 22.0, "items": "Trail mix", "slot": "snack"})
        dinner = 18 * 60 + 20 + int(rng.integers(-20, 30)) if rng.random() > 0.25 else 20 * 60 + 40
        meals.append({"ts": dinner, "day": d, "carbs": float(rng.integers(50, 81)), "items": "Chicken, rice and vegetables", "slot": "dinner"})

        # sleep for the night starting this evening
        onset = 23 * 60 + int(rng.normal(0, 15)) + (50 if d in late_activity_days else 0)
        wake = 24 * 60 + 7 * 60 + int(rng.normal(0, 12))
        if d in short_nights:  # short night from an early alarm, bedtime unchanged
            wake = 24 * 60 + 4 * 60 + 10 + int(rng.normal(0, 10))
        a, b = idx(d, onset), min(idx(d, wake), n)
        asleep[a:b] = True
        prev_sleep_hours = (b - a) / 60

    walking &= ~asleep
    for d in short_nights:  # every walk on the day after a short night runs hotter
        a, b = (d + 1) * 1440, min((d + 2) * 1440, n)
        walk_hr_extra[a:b] = np.where(walking[a:b], 12.0, 0.0)
    cadence[walking] = rng.normal(108, 6, walking.sum())

    # movement
    light = (~asleep) & (~walking) & (rng.random(n) < 0.15)
    steps = np.where(walking, cadence, np.where(light, rng.integers(5, 25, n), 0)).astype(float)
    enmo = np.where(asleep, rng.normal(3, 1, n), np.where(walking, rng.normal(120, 15, n), np.where(light, rng.normal(35, 8, n), rng.normal(8, 3, n))))
    restless = asleep & (rng.random(n) < 0.02)
    enmo[restless] = 40
    steps[restless] = 0

    # meal timing arrays and glucose
    meal_rows = []
    glucose = 92 + 4 * np.sin(2 * np.pi * minute_of_day / 1440) + rng.normal(0, 2.5, n)
    still_stress = stress
    for m in meals:
        t0 = m["day"] * 1440 + m["ts"]
        if t0 >= n:
            continue
        steps_before = steps[max(0, t0 - 60):t0].sum()
        stressed = still_stress[max(0, t0 - 30):t0].mean() > 0.5
        late = m["ts"] >= 20 * 60
        rise = 0.9 * m["carbs"]
        rise *= 0.6 if steps_before > 800 else 1.0
        rise *= 1.35 if stressed else 1.0
        rise *= 1.15 if late else 1.0
        rise *= rng.normal(1.0, 0.07)
        t = np.arange(n) - t0
        glucose += np.where(t >= 0, rise * _curve(t), 0)
        meal_rows.append((t0, m))
    glucose -= np.where(walking, 6, 0)

    # heart rate, EDA, temperature
    hr = np.where(asleep, 56, 70) + rng.normal(0, 2, n)
    hr += np.where(walking, 0.3 * cadence + walk_hr_extra, 0)
    excess = np.zeros(n)
    for i in range(1, n):  # recovery after activity
        excess[i] = excess[i - 1] * np.exp(-1 / 6)
        if walking[i - 1] and not walking[i]:
            excess[i] = hr[i - 1] - 70
    hr += np.where(~walking & ~asleep, excess, 0)
    for t0, _ in meal_rows:
        hr[t0:t0 + 90] += 5
    hr += 6 * stress * ~walking

    eda = np.where(asleep, 0.2, 0.4) + 0.5 * stress + np.where(walking, 0.3, 0) + rng.normal(0, 0.12, n)
    temp = np.where(asleep, 33.6, np.where(walking, 31.2, 31.8)) + rng.normal(0, 0.15, n)

    # planted problems
    eda[band_off] = 0.0
    temp[band_off] = 25.5
    hr[band_off] = np.nan
    steps[band_off] = 0
    lunch_day5 = [t0 for t0, m in meal_rows if m["day"] == 5 and m["slot"] == "lunch"]
    if lunch_day5:
        glucose[lunch_day5[0] + 25:lunch_day5[0] + 55] = np.nan
    fl = 3 * 1440 + 3 * 60
    glucose[fl:fl + 15] -= 45

    gl = np.full(n, np.nan)
    gl[3::5] = glucose[3::5]

    minute = pd.DataFrame({
        "pid": pid, "ts": ts, "enmo_mg": np.clip(enmo, 0, None), "steps": np.clip(steps, 0, None).round(),
        "hr": hr, "eda": np.clip(eda, 0, None), "temp": temp, "glucose": gl.round(),
    })
    minute = finalize_minute(minute)

    meal_df = pd.DataFrame([{
        "pid": pid, "meal_id": f"{pid}-m{i:03d}", "ts": ts[t0], "carbs": m["carbs"], "carbs_missing": False,
        "sugar": m["carbs"] * 0.2, "fiber": 4.0, "protein": 15.0, "fat": 10.0, "calories": m["carbs"] * 4 + 160,
        "items": m["items"],
    } for i, (t0, m) in enumerate(sorted(meal_rows, key=lambda r: r[0]))])
    return minute, meal_df
