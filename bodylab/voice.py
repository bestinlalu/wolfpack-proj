"""Weekly recap: text from the lab notebook, optionally polished by Gemini, spoken by ElevenLabs."""
from __future__ import annotations

import logging

import pandas as pd
import requests

from bodylab.agent.notebook import Notebook
from bodylab.config import SETTINGS, elevenlabs_api_key, gemini_api_key

log = logging.getLogger(__name__)


def recap_text(nb: Notebook, until: pd.Timestamp, days: int = 7) -> str:
    since = until - pd.Timedelta(days=days)
    recent = [m for m in nb.messages if pd.Timestamp(m["ts"]) > since]
    found = [m for m in recent if m["kind"] == "discovery"]
    cases = [m for m in recent if m["kind"] == "case"]
    rejected = [m for m in recent if m["kind"] == "rejected"]
    quiet = [e for e in nb.events if pd.Timestamp(e["ts"]) > since and e["verdict"] in ("bad_data", "unexplained")]
    rank, pts, nxt = nb.rank()
    parts = [f"Here's your Body Lab week."]
    if found:
        parts.append(f"You confirmed {len(found)} new discover{'y' if len(found) == 1 else 'ies'}. " + " ".join(m["body"].split(". Held")[0] + "." for m in found[:2]))
    if cases:
        parts.append(f"I solved {len(cases)} case{'s' if len(cases) != 1 else ''}. The latest: {cases[-1]['title']}.")
    if rejected:
        parts.append(f"{len(rejected)} idea{'s' if len(rejected) != 1 else ''} didn't hold up, so I logged {'them' if len(rejected) != 1 else 'it'} as noise.")
    if quiet:
        parts.append(f"{len(quiet)} surprise{'s' if len(quiet) != 1 else ''} had no clear reason or bad sensor data, so I didn't bother you about {'them' if len(quiet) != 1 else 'it'}.")
    article = "an" if rank[0].lower() in "aeiou" else "a"
    parts.append(f"You're {article} {rank} with {pts} points" + (f", {nxt - pts} away from the next rank." if nxt else "."))
    return " ".join(parts)


def polish(text: str) -> str:
    """Optional: let Gemini turn the recap into a warm 30-second script. Facts must stay the same."""
    if not gemini_api_key():
        return text
    try:
        from google import genai

        client = genai.Client(api_key=gemini_api_key())
        prompt = ("Rewrite this as a friendly 30-second spoken recap (under 85 words). Keep every number and fact exactly; "
                  "add nothing new; no medical advice; plain sentences, no lists.\n\n" + text)
        from google.genai import types

        cfg = types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        resp = client.models.generate_content(model=SETTINGS.gemini_model, contents=prompt, config=cfg)
        return (resp.text or text).strip()
    except Exception as exc:
        log.warning("Recap polish failed: %s", exc)
        return text


def speak(text: str) -> bytes | None:
    """MP3 bytes from ElevenLabs text-to-speech, or None without a key."""
    key = elevenlabs_api_key()
    if not key:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{SETTINGS.elevenlabs_voice_id}"
    resp = requests.post(url, headers={"xi-api-key": key, "Accept": "audio/mpeg"},
                         json={"text": text, "model_id": SETTINGS.elevenlabs_model}, timeout=60)
    resp.raise_for_status()
    return resp.content
