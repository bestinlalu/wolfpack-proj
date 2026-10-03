"""Check that the API keys in .env work, without printing them.

    python scripts/check_keys.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from bodylab.config import SETTINGS, elevenlabs_api_key, gemini_api_key  # noqa: E402


def check_gemini() -> bool:
    key = gemini_api_key()
    if not key:
        print("Gemini: no GEMINI_API_KEY in .env")
        return False
    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=key)
        cfg = types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        resp = client.models.generate_content(model=SETTINGS.gemini_model, contents="Reply with the single word: ready", config=cfg)
        print(f"Gemini: OK ({SETTINGS.gemini_model} replied '{(resp.text or '').strip()[:40]}')")
        return True
    except Exception as exc:
        print(f"Gemini: FAILED ({type(exc).__name__}: {str(exc)[:200]})")
        return False


def check_elevenlabs() -> bool:
    key = elevenlabs_api_key()
    if not key:
        print("ElevenLabs: no ELEVENLABS_API_KEY in .env (optional)")
        return False
    # Tests the one permission Body Lab needs (text to speech) with a two-word request.
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{SETTINGS.elevenlabs_voice_id}"
    resp = requests.post(url, headers={"xi-api-key": key, "Accept": "audio/mpeg"},
                         json={"text": "Body Lab.", "model_id": SETTINGS.elevenlabs_model}, timeout=30)
    if resp.ok:
        print(f"ElevenLabs: OK ({len(resp.content):,} bytes of audio)")
    else:
        detail = resp.text[:200].replace("\n", " ")
        print(f"ElevenLabs: FAILED (HTTP {resp.status_code}: {detail})")
    return resp.ok


if __name__ == "__main__":
    ok = check_gemini()
    check_elevenlabs()
    sys.exit(0 if ok else 1)
