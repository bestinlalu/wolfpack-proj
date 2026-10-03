"""Planted-change validation: the agent must recover effects we put into the synthetic data."""
import pandas as pd
import pytest

from bodylab.data.synthetic import generate
from bodylab.engine import Engine


@pytest.fixture(scope="module")
def notebook():
    minute, meals = generate(days=14)
    eng = Engine("S01", minute, meals)
    eng.run(step_hours=6)
    return eng.notebook


def _discovery(nb, lab, factor):
    by_h = {h["hyp_id"]: h for h in nb.hypotheses}
    return next((d for d in nb.discoveries if d["lab"] == lab and by_h[d["hyp_id"]]["factor"] == factor and d["status"] != "rejected"), None)


def test_short_night_walks_found_with_right_size(notebook):
    d = _discovery(notebook, "movement", "prev_sleep_h")
    assert d is not None
    assert 8 <= d["effect_abs"] <= 16  # planted +12 bpm


def test_pre_meal_walk_found(notebook):
    d = _discovery(notebook, "fuel", "steps_before")
    assert d is not None
    assert d["effect_pct"] > 25  # skipping the walk: planted about +65%


def test_night_owl_lag_found(notebook):
    d = _discovery(notebook, "sleep", "late_steps")
    assert d is not None
    assert 25 <= d["effect_abs"] <= 75  # planted +50 minutes


def test_bad_data_is_dismissed_quietly(notebook):
    bad = [e for e in notebook.events if e["verdict"] == "bad_data"]
    assert bad, "the planted glucose gap should be caught"
    assert not any(m["ref"] == e["event_id"] for e in bad for m in notebook.messages)


def test_every_lead_message_leads_with_the_why(notebook):
    for e in notebook.events:
        if e["verdict"] == "lead":
            assert " with " in e["title"] and "Biggest difference" in e["message"]


def test_no_lookahead():
    minute, meals = generate(days=6)
    eng = Engine("S01", minute, meals)
    cut = minute["ts"].min() + pd.Timedelta(days=3)
    eng.step(cut)
    for df in eng.features.values():
        if not df.empty:
            assert (df["end_ts"] <= cut).all()
