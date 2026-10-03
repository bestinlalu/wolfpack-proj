"""Thresholds and settings. Values are starting points to tune on the real data."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment. Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


@dataclass(frozen=True)
class DataCheck:
    max_glucose_gap_min: int = 15
    sensor_warmup_hours: int = 24
    max_jump_mg_dl_per_5min: float = 30.0
    false_low_drop_mg_dl: float = 25.0
    min_worn_temp_c: float = 28.0
    min_worn_eda_us: float = 0.02
    overlap_window_min: int = 120
    early_rise_min: int = 15
    early_rise_mg_dl: float = 15.0


@dataclass(frozen=True)
class Detection:
    surprise_z: float = 1.28  # about the top or bottom 10%
    min_history: int = 5
    min_history_by_lab: tuple = (("fuel", 5), ("stress", 3), ("sleep", 3), ("movement", 5))
    min_same_slot: int = 4

    def history_needed(self, lab: str) -> int:
        return dict(self.min_history_by_lab).get(lab, self.min_history)
    clear_difference_z: float = 1.5
    very_unusual_z: float = 2.0
    somewhat_z: float = 1.0


@dataclass(frozen=True)
class Hypothesis:
    confirm_supports: int = 3
    confirm_ratio: float = 0.75
    max_chances: int = 5
    expire_days: float = 4.0
    max_open: int = 10
    max_open_meal: int = 2
    factor_present_z: float = 0.5
    response_z: float = 0.3
    fading_window: int = 4
    fading_ratio: float = 0.5


@dataclass(frozen=True)
class Signals:
    still_enmo_mg: float = 15.0
    walk_steps_per_min: float = 60.0
    walk_min_minutes: int = 10
    walk_max_gap_min: int = 1
    meal_group_min: int = 30
    fuel_window_min: int = 120
    night_start_hour: int = 20
    night_end_hour: int = 12
    sleep_max_interrupt_min: int = 10
    sleep_min_minutes: int = 120
    sleep_enmo_mg: float = 6.0
    sleep_hr_drop: float = 3.0


@dataclass(frozen=True)
class Settings:
    data_check: DataCheck = field(default_factory=DataCheck)
    detection: Detection = field(default_factory=Detection)
    hypothesis: Hypothesis = field(default_factory=Hypothesis)
    signals: Signals = field(default_factory=Signals)
    # Hours to add to wristband timestamps so they line up with Dexcom and food-log local time.
    wrist_offset_hours: float = float(os.getenv("BODYLAB_WRIST_OFFSET_HOURS", "0"))
    lakehouse_dir: Path = Path(os.getenv("BODYLAB_LAKEHOUSE", str(ROOT / "lakehouse")))
    raw_dir: Path = Path(os.getenv("BODYLAB_RAW_DIR", str(ROOT / "data" / "raw")))
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    # Tried in order when the main model is overloaded (503), rate limited (429) or retired (404).
    gemini_fallback_models: tuple = tuple(m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.5-flash,gemini-flash-latest,gemini-flash-lite-latest").split(",") if m.strip())
    # Most cases to send to Gemini per run (0 = no limit); the rest use the rule-based investigator.
    gemini_max_cases: int = int(os.getenv("GEMINI_MAX_CASES", "0"))
    # Only for the cost estimate printed by scripts/replay.py; check Google's pricing page for your model.
    gemini_price_input_per_m: float = float(os.getenv("GEMINI_PRICE_INPUT_PER_M", "0.50"))
    gemini_price_output_per_m: float = float(os.getenv("GEMINI_PRICE_OUTPUT_PER_M", "3.00"))
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    elevenlabs_model: str = os.getenv("ELEVENLABS_MODEL", "eleven_multilingual_v2")


SETTINGS = Settings()


def gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def elevenlabs_api_key() -> str | None:
    return os.getenv("ELEVENLABS_API_KEY")
