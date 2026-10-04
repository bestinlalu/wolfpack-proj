"""Demo augmentation keeps the data's shape and only nudges signal values."""
import numpy as np

from bodylab.data.augment import augment
from bodylab.data.synthetic import generate


def test_augment_keeps_shape_and_timing():
    minute, meals = generate(days=8)
    out = augment(minute, meals, seed=1)
    assert list(out.columns) == list(minute.columns) and len(out) == len(minute)
    assert (out["ts"].values == minute.sort_values("ts")["ts"].values).all()
    assert out["glucose"].isna().sum() == minute["glucose"].isna().sum()  # gaps stay gaps
    assert not np.allclose(out["glucose"].fillna(0), minute["glucose"].fillna(0))


def test_augment_is_deterministic():
    minute, meals = generate(days=6)
    a, b = augment(minute, meals, seed=3), augment(minute, meals, seed=3)
    assert a.equals(b)
