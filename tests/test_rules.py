"""Hypothesis lifecycle and data checks."""
import numpy as np
import pandas as pd

from bodylab.agent import checks
from bodylab.agent.notebook import Notebook
from bodylab.data.synthetic import generate
from bodylab.pipeline.features import Signals

T0 = pd.Timestamp("2020-02-13 12:00")


def _event(nb: Notebook, sid: str = "S01-fuel-1") -> dict:
    e = {"pid": "S01", "event_id": f"E{len(nb.events) + 1:03d}", "sid": sid, "lab": "fuel", "z": 2.0}
    nb.events.append(e)
    return e


def _situation(sid: str, z: float) -> pd.Series:
    return pd.Series({"sid": sid, "z": z, "ts": T0})


def test_confirm_after_three_supports():
    nb = Notebook("S01")
    h, _ = nb.open_hypothesis(_event(nb), "fuel", "steps_before", -1, T0, side=-1)
    assert h["supports"] == 1 and h["status"] == "testing"
    for i in range(2):  # fewer steps (factor_z negative) and a higher spike support direction -1
        nb.record_evidence(h, _situation(f"s{i}", 1.0), factor_z=-2.0, ts=T0)
    assert nb.update_status(h, T0) == "confirmed"


def test_reject_when_contradicted():
    nb = Notebook("S01")
    h, _ = nb.open_hypothesis(_event(nb), "fuel", "steps_before", -1, T0)
    for i in range(3):
        nb.record_evidence(h, _situation(f"s{i}", -1.0), factor_z=-2.0, ts=T0)
    assert nb.update_status(h, T0) == "rejected"


def test_small_factor_change_is_not_a_test():
    nb = Notebook("S01")
    h, _ = nb.open_hypothesis(_event(nb), "fuel", "steps_before", -1, T0)
    assert nb.record_evidence(h, _situation("s1", 2.0), factor_z=0.1, ts=T0) is None
    assert h["chances"] == 1


def test_expires_without_chances():
    nb = Notebook("S01")
    h, _ = nb.open_hypothesis(_event(nb), "fuel", "steps_before", -1, T0)
    assert nb.update_status(h, T0 + pd.Timedelta(days=5)) == "expired"


def test_meal_cap_drops_weakest():
    nb = Notebook("S01")
    for factor in ("hour", "since_last_meal_h"):
        nb.open_hypothesis(_event(nb, f"x-{factor}"), "fuel", factor, 1, T0)
    nb.open_hypothesis(_event(nb, "x-3"), "fuel", "since_last_meal_h", -1, T0)
    open_meal = [h for h in nb.open_testing() if h["meal_related"]]
    assert len(open_meal) == 2


def test_rank_progression():
    nb = Notebook("S01")
    assert nb.rank()[0] == "Intern"
    nb.discoveries = [{"rarity": "rare", "status": "confirmed"}] * 4
    assert nb.rank()[0] == "Researcher"


def test_glucose_gap_fails_check():
    minute, meals = generate(days=8)
    sig = Signals(minute, minute["ts"].max())
    lunch = meals[(meals["ts"].dt.day == START_DAY + 5) & (meals["ts"].dt.hour.between(11, 14))].iloc[0]["ts"]
    result = checks.glucose_checks(sig, lunch - pd.Timedelta(minutes=15), lunch + pd.Timedelta(minutes=120))
    assert not next(c for c in result if c["check"] == "Glucose gaps")["passed"]


def test_band_off_fails_worn_check():
    minute, _ = generate(days=3)
    sig = Signals(minute, minute["ts"].max())
    shower = pd.Timestamp("2020-02-14 07:05")
    assert not checks.worn_check(sig, shower, shower + pd.Timedelta(minutes=20))["passed"]
    assert checks.worn_check(sig, shower + pd.Timedelta(hours=2), shower + pd.Timedelta(hours=3))["passed"]


START_DAY = 13
