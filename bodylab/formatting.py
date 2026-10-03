"""Human-readable values for messages and tool results."""
from __future__ import annotations


def fmt(v, unit: str) -> str:
    if v is None:
        return "n/a"
    if unit == "steps":
        return f"{v:,.0f} steps"
    if unit == "h":
        return f"{v:.1f} h"
    if unit in ("mg/dL", "bpm", "°C"):
        return f"{v:.0f} {unit}" if unit != "°C" else f"{v:.1f} °C"
    if unit == "z":  # stress signal, in units of the person's own spread
        return "calm" if v <= -0.5 else "typical" if v < 1 else "elevated" if v < 2.5 else "high"
    if unit == "flag":
        return "weekend" if v >= 0.5 else "weekday"
    if unit == "clock":
        return f"{int(v):02d}:{int(round((v % 1) * 60)) % 60:02d}"
    if unit == "min after 20:00":
        return f"{int(v) // 60 + 20:02d}:{int(v) % 60:02d}"
    return f"{v:.1f}"
