"""Demo augmentation: plant clear, personal patterns into a real participant's prepared minute data.

Real BIG IDEAs participants only have about ten days of data, which is rarely enough for the agent to confirm much.
For the demo, this nudges the recorded signals around each participant's own real meals, walks and nights so a few
patterns become consistent enough to be discovered. Everything else (timing, meals, gaps, noise) stays real.
`scripts/augment.py` keeps the untouched data in `minute.original.parquet`.

Planted patterns (each person gets them at their own strength):
- Fuel: more steps in the hour before a meal, or in the hour after it, means a smaller glucose rise;
  a higher stress signal before eating means a bigger one.
- Movement: walks after a short night, or after a stressful few hours, run at a higher heart rate.
- Stress: the stress signal runs higher on days after a short night, and on days with less activity so far.
- Sleep: more activity after 21:00 delays sleep onset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bodylab.config import SETTINGS
from bodylab.pipeline import features as F

MIN = pd.Timedelta(minutes=1)


def _top(values: pd.Series, q: float) -> pd.Series:
    """Rows in the top (q > 0.5) or bottom (q < 0.5) part of this person's own range."""
    cut = values.quantile(q)
    return values >= cut if q > 0.5 else values <= cut


def _rank(values: pd.Series) -> pd.Series:
    """0 for this person's lowest value, 1 for the highest (missing values sit in the middle)."""
    return values.rank(pct=True).fillna(0.5)


def augment(minute: pd.DataFrame, meals: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = rng.uniform(0.85, 1.15)  # people differ a little in how strong each pattern is
    m = minute.sort_values("ts").reset_index(drop=True).copy()
    until = m["ts"].max()
    sig = F.Signals(m, until)
    t0 = sig.t0
    nights = F.build_nights(sig, meals)

    def idx(t: pd.Timestamp) -> int:
        return int((t - t0) / MIN)

    n = len(m)
    hr, eda, enmo, steps, glucose = (m[c].to_numpy(float).copy() for c in ("hr", "eda", "enmo_mg", "steps", "glucose"))

    # ---- Sleep: active late evenings push sleep onset back about 45 minutes
    if len(nights) >= 4:
        late = _top(nights["late_steps"], 0.6)
        for _, night in nights[late].iterrows():
            a = idx(night["ts"])
            b = min(a + int(45 * strength), n)
            enmo[a:b] = np.maximum(enmo[a:b], SETTINGS.signals.sleep_enmo_mg + 25)
            steps[a:b] += 6  # light moving about, well under a walk
    m["enmo_mg"], m["steps"] = enmo, steps
    sig = F.Signals(m, until)
    nights = F.build_nights(sig, meals)
    if len(nights) >= 4:
        short_nights = nights[_top(nights["duration_h"], 0.4)]
    else:
        short_nights = pd.DataFrame(columns=["ts", "end_ts", "duration_h"])

    def after_short_night(start: pd.Timestamp, end: pd.Timestamp) -> bool:
        return bool(((short_nights["end_ts"] <= start) & (short_nights["end_ts"] > start - pd.Timedelta(hours=20))).any()) \
            and end > start

    # ---- Stress: days after a short night carry a higher stress signal (via skin conductance)
    awake_still = sig.still & pd.Series(sig.ts).dt.hour.between(8, 21).to_numpy()
    eda_ref = eda[awake_still & ~np.isnan(eda)]
    eda_scale = F.robust_scale(eda_ref, 0.1)
    for _, night in short_nights.iterrows():
        a, b = idx(night["end_ts"]), min(idx(night["end_ts"] + pd.Timedelta(hours=14)), n)
        if b > a:
            eda[a:b] += 2.5 * strength * eda_scale

    # ---- Stress: low-activity days run more stressed than active ones (compared at the same time of day)
    blocks = F.build_stress_hours(sig, meals, nights)
    if len(blocks) >= 8:
        activity = blocks.groupby("hour")["steps_earlier"].transform(_rank)  # vs the same hour on other days
        for i, blk in blocks.iterrows():
            a, b = idx(blk["ts"]), min(idx(blk["end_ts"]), n)
            eda[a:b] += 3.0 * strength * eda_scale * (1 - 2 * activity[i])

    # ---- Movement: walks after a short night, or after a stressful few hours, run hotter
    m["eda"] = eda
    sig = F.Signals(m, until)
    walks = F.build_walks(sig, meals, nights)
    if len(walks) >= 6:
        tension = _rank(walks["stress_before"])
        for i, w in walks.iterrows():
            extra = (9 if after_short_night(w["ts"], w["end_ts"]) else 0) + 10 * (2 * tension[i] - 1)
            if extra:
                a, b = idx(w["ts"]), min(idx(w["end_ts"]), n)
                hr[a:b] += extra * strength

    # ---- Fuel: scale each meal's glucose rise by what happened around it
    m["eda"], m["hr"] = eda, hr
    sig = F.Signals(m, until)
    fuel = F.build_meals(sig, meals, nights).dropna(subset=["start_glucose"])
    if len(fuel) >= 6:
        walked_before, walked_after = _rank(fuel["steps_before"]), _rank(fuel["steps_after"])
        stressed = _rank(fuel["stress_before"])
        win = SETTINGS.signals.fuel_window_min
        for i, meal in fuel.iterrows():
            f = 1.0
            f *= 1.6 - 1.1 * walked_before[i]  # x1.6 with no walk, x0.5 after this person's longest walks
            f *= 1.15 - 0.3 * walked_after[i]
            f *= 0.8 + 0.5 * stressed[i]
            f = 1 + (f - 1) * strength
            a = idx(meal["ts"])
            if a < 0 or a >= n:
                continue
            base = meal["start_glucose"]
            # full effect over the meal window, then fade back to the real trace over an hour
            ramp = np.concatenate([np.full(win, f), np.linspace(f, 1.0, 60)])
            b = min(a + len(ramp), n)
            seg = glucose[a:b]
            glucose[a:b] = np.where(np.isnan(seg), seg, base + (seg - base) * ramp[: b - a])
    m["glucose"] = glucose
    return m
