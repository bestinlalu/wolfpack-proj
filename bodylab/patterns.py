"""Short display labels for hypotheses; evidence and lifecycle rules stay in the notebook."""
from __future__ import annotations

import json

import pandas as pd

from bodylab.agent.notebook import TITLES
from bodylab.labs import cause


FACTORS = {
    "steps_before": ("More pre-meal walking", "Less pre-meal walking"),
    "steps_after": ("More after-meal walking", "Less after-meal walking"),
    "stress_before": ("Higher stress", "Lower stress"),
    "prev_sleep_h": ("Longer sleep", "Shorter sleep"),
    "start_glucose": ("Higher starting glucose", "Lower starting glucose"),
    "since_last_meal_h": ("Longer meal gap", "Shorter meal gap"),
    "steps_earlier": ("More daytime activity", "Less daytime activity"),
    "weekend": ("Weekends", "Weekdays"),
    "late_steps": ("More late activity", "Less late activity"),
    "dinner_gap_h": ("Earlier dinner", "Later dinner"),
    "evening_glucose": ("Higher evening glucose", "Lower evening glucose"),
    "day_stress": ("More afternoon stress", "Less afternoon stress"),
    "night_temp": ("Warmer nights", "Cooler nights"),
    "temp_before": ("Warmer skin", "Cooler skin"),
}
RESPONSES = {
    "fuel": ("bigger glucose rise", "smaller glucose rise"),
    "movement": ("higher walk heart rate", "lower walk heart rate"),
    "sleep": ("later sleep", "earlier sleep"),
    "stress": ("more stress", "less stress"),
}
STATUS = {"testing": "Possible link", "confirmed": "Confirmed", "fading": "Mixed evidence",
          "rejected": "Denied", "inconclusive": "Not enough data", "expired": "Not enough data"}


def _factor(h: dict, high: bool) -> str:
    if h["factor"] == "hour":
        return ("Later" if high else "Earlier") + (" meals" if h["lab"] == "fuel" else " walks")
    return FACTORS[h["factor"]][0 if high else 1]


def _parts(h: dict) -> tuple[str, str]:
    side = h.get("side", 1)
    side = 1 if side is None or pd.isna(side) or side >= 0 else -1
    response = RESPONSES[h["lab"]][0 if h["direction"] * side > 0 else 1]
    return _factor(h, side > 0), response


def question(h: dict) -> str:
    factor, response = _parts(h)
    verb = "Do" if h["factor"] in ("hour", "weekend", "night_temp") else "Does"
    article = "a " if h["lab"] in ("fuel", "movement") else ""
    return f"{verb} {factor.lower()} go with {article}{response}?"


def statement(h: dict) -> str:
    factor, response = _parts(h)
    article = "a " if h["lab"] in ("fuel", "movement") else ""
    return f"{factor} went with {article}{response}."


def name(h: dict) -> str:
    return TITLES.get((h["lab"], h["factor"], h["direction"]), cause(h["lab"], h["factor"]).label + " Pattern")


def origin(h: dict, events: list[dict]) -> str:
    event = next((e for e in events if e["event_id"] == h.get("trigger_event")), None)
    if event is None:
        return ""
    observed = dict(h)
    differences = event.get("differences_json")
    if isinstance(differences, str) and differences:
        top = next((d for d in json.loads(differences) if d["factor"] == h["factor"]), None)
        if top and top.get("this_time") is not None and top.get("similar_median") is not None:
            observed["side"] = 1 if top["this_time"] > top["similar_median"] else -1
            if event.get("z") is not None and pd.notna(event["z"]):
                observed["direction"] = observed["side"] * (1 if event["z"] > 0 else -1)
    ts = event.get("ts")
    day = pd.Timestamp(ts).day_name() if ts is not None and pd.notna(ts) else "First observation"
    situation = {"fuel": "meal", "movement": "walk", "sleep": "night", "stress": "stress window"}[h["lab"]]
    label = f"{day}’s {situation}" if day != "First observation" else "the first observation"
    return f"Started after {label}: {statement(observed).lower()}"
