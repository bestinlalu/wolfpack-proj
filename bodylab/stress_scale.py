"""Personal 1-10 stress scale: where a reading ranks among the person's own past 2-hour windows."""
from __future__ import annotations

import numpy as np

MIN_WINDOWS = 14  # about two days of waking windows before the scale means anything

EXPLAINER = ("Stress level 1–10: your skin's sweat response and heart rate while sitting still, ranked against your own "
             "past 2-hour windows. 9 means more stressed than about 80–90% of your usual windows; 2 means calmer than most. "
             "1–3 is calmer than usual for you, 4–7 typical, 8–10 more stressed. It measures arousal, so excitement or heat "
             "can raise it too.")


def stress_level(history, value) -> int | None:
    """Decile of `value` within `history` (raw stress signals from earlier windows), or None while learning."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    hist = np.asarray(history, dtype=float)
    hist = hist[np.isfinite(hist)]
    if len(hist) < MIN_WINDOWS or not np.isfinite(value):
        return None
    pct = ((hist < value).sum() + 0.5 * (hist == value).sum()) / len(hist)
    return int(min(max(np.ceil(pct * 10), 1), 10))


def band(level: int | None) -> str:
    if level is None:
        return "still learning your baseline"
    if level <= 3:
        return "calmer than usual for you"
    if level <= 7:
        return "typical for you"
    return "more stressed than usual for you"
