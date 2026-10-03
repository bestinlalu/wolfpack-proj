"""Meal photo to meal log with Gemini. Photo estimates are rough, so they are flagged as such."""
from __future__ import annotations

import json

from bodylab.config import SETTINGS, gemini_api_key

MEAL_SCHEMA = {
    "type": "object",
    "properties": {
        "foods": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "portion": {"type": "string"}}, "required": ["name", "portion"]}},
        "carbs_g": {"type": "number"},
        "protein_g": {"type": "number"},
        "fat_g": {"type": "number"},
        "fiber_g": {"type": "number"},
        "sugar_g": {"type": "number"},
        "calories": {"type": "number"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["foods", "carbs_g", "protein_g", "fat_g", "fiber_g", "sugar_g", "calories", "confidence"],
}

PROMPT = ("Identify the foods in this meal photo and estimate each portion. Then estimate total carbohydrates, protein, "
          "fat, fiber, sugar (grams) and calories for the whole plate. Be conservative and say low confidence when "
          "portions or hidden ingredients (oil, sauces, sugar) are unclear.")


def analyze(image: bytes, mime_type: str = "image/jpeg") -> dict:
    if not gemini_api_key():
        raise RuntimeError("Set GEMINI_API_KEY to use photo logging.")
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key())
    resp = client.models.generate_content(
        model=SETTINGS.gemini_model,
        contents=[types.Part.from_bytes(data=image, mime_type=mime_type), PROMPT],
        config=types.GenerateContentConfig(response_mime_type="application/json", response_json_schema=MEAL_SCHEMA, temperature=0.1,
                                           automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
    )
    data = json.loads(resp.text)
    data["source"] = "photo"
    return data
