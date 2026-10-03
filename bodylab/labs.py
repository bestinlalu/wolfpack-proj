"""What each lab compares, measures, and may blame."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cause:
    key: str
    label: str
    unit: str
    meal_related: bool = False
    phrase_high: str = ""
    phrase_low: str = ""


@dataclass(frozen=True)
class Lab:
    key: str
    name: str
    situation: str
    response_label: str
    response_unit: str
    causes: tuple[Cause, ...]
    context: tuple[str, ...] = field(default_factory=tuple)


FUEL = Lab(
    key="fuel", name="Fuel", situation="meal",
    response_label="glucose rise", response_unit="mg/dL",
    context=("carbs", "slot"),
    causes=(
        Cause("steps_before", "Steps in the hour before", "steps", phrase_high="more steps in the hour before eating", phrase_low="fewer steps in the hour before eating"),
        Cause("steps_after", "Steps in the hour after", "steps", phrase_high="more steps in the hour after eating", phrase_low="fewer steps in the hour after eating"),
        Cause("stress_before", "Stress signal before eating", "z", phrase_high="a higher stress signal before eating", phrase_low="a calmer stress signal before eating"),
        Cause("hour", "Time of day", "clock", meal_related=True, phrase_high="eating later in the day", phrase_low="eating earlier in the day"),
        Cause("start_glucose", "Glucose at the start", "mg/dL", phrase_high="higher glucose before eating", phrase_low="lower glucose before eating"),
        Cause("since_last_meal_h", "Hours since the last meal", "h", meal_related=True, phrase_high="a longer gap since the last meal", phrase_low="a shorter gap since the last meal"),
        Cause("prev_sleep_h", "Previous night's sleep", "h", phrase_high="a longer night's sleep", phrase_low="a shorter night's sleep"),
    ),
)

STRESS = Lab(
    key="stress", name="Stress", situation="2-hour window",
    response_label="stress signal", response_unit="z",
    context=("hour", "weekend", "after_meal"),
    causes=(
        Cause("prev_sleep_h", "Previous night's sleep", "h", phrase_high="a longer night's sleep", phrase_low="a shorter night's sleep"),
        Cause("steps_earlier", "Steps earlier that day", "steps", phrase_high="more activity earlier in the day", phrase_low="less activity earlier in the day"),
        Cause("weekend", "Day type", "flag", phrase_high="it being a weekend", phrase_low="it being a weekday"),
    ),
)

SLEEP = Lab(
    key="sleep", name="Sleep", situation="night",
    response_label="sleep onset", response_unit="min after 20:00",
    context=(),
    causes=(
        Cause("late_steps", "Steps after 21:00", "steps", phrase_high="more activity after 21:00", phrase_low="less activity after 21:00"),
        Cause("dinner_gap_h", "Hours from dinner to sleep", "h", meal_related=True, phrase_high="a longer gap between dinner and sleep", phrase_low="a later dinner, closer to sleep"),
        Cause("evening_glucose", "Evening glucose", "mg/dL", phrase_high="higher evening glucose", phrase_low="lower evening glucose"),
        Cause("day_stress", "Afternoon stress signal", "z", phrase_high="a more stressful afternoon", phrase_low="a calmer afternoon"),
        Cause("night_temp", "Skin temperature at night", "°C", phrase_high="a warmer night", phrase_low="a cooler night"),
    ),
)

MOVEMENT = Lab(
    key="movement", name="Movement", situation="walk",
    response_label="walk heart rate", response_unit="bpm",
    context=("cadence", "duration_min"),
    causes=(
        Cause("prev_sleep_h", "Previous night's sleep", "h", phrase_high="a longer night's sleep", phrase_low="a shorter night's sleep"),
        Cause("stress_before", "Stress signal in the 3 hours before", "z", phrase_high="a more stressful few hours before", phrase_low="a calmer few hours before"),
        Cause("hour", "Time of day", "clock", phrase_high="walking later in the day", phrase_low="walking earlier in the day"),
        Cause("since_last_meal_h", "Hours since the last meal", "h", meal_related=True, phrase_high="walking longer after a meal", phrase_low="walking soon after a meal"),
        Cause("temp_before", "Skin temperature before", "°C", phrase_high="warmer skin before the walk", phrase_low="cooler skin before the walk"),
    ),
)

LABS = {lab.key: lab for lab in (FUEL, STRESS, SLEEP, MOVEMENT)}

# Causes that belong to another lab's measurement make a cross-lab finding.
CROSS_LAB_CAUSES = {
    "fuel": {"stress_before", "prev_sleep_h"},
    "stress": {"prev_sleep_h"},
    "sleep": {"evening_glucose", "day_stress", "dinner_gap_h"},
    "movement": {"prev_sleep_h", "stress_before"},
}

# Well-known effects: confirmed patterns on these become common cards.
COMMON_PATTERNS = {
    ("fuel", "hour"), ("fuel", "steps_after"), ("sleep", "dinner_gap_h"),
    ("stress", "weekend"), ("movement", "hour"), ("movement", "since_last_meal_h"),
}


def cause(lab_key: str, cause_key: str) -> Cause:
    return next(c for c in LABS[lab_key].causes if c.key == cause_key)
