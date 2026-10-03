"""Personal 1-10 stress scale."""
import numpy as np

from bodylab.stress_scale import MIN_WINDOWS, band, stress_level


def test_learning_until_enough_history():
    assert stress_level(np.arange(MIN_WINDOWS - 1), 3.0) is None
    assert stress_level(np.arange(MIN_WINDOWS), 3.0) is not None


def test_levels_follow_personal_rank():
    hist = np.linspace(-2, 2, 100)
    assert stress_level(hist, -5) == 1
    assert stress_level(hist, 0) == 5
    assert stress_level(hist, 1.9) == 10
    assert stress_level(hist + 10, 0) == 1  # same reading, calmer than everything for a person with a higher baseline


def test_bands():
    assert band(2) == "calmer than usual for you"
    assert band(5) == "typical for you"
    assert band(9) == "more stressed than usual for you"
    assert band(None) == "still learning your baseline"
