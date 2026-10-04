"""Mindfulness tab helpers (no Gemini or ElevenLabs calls)."""
import json

from bodylab import meditation as md


def test_parse_cues_falls_back_for_missing_or_bad():
    p = md.PRACTICES["breathing"]
    cues = md.parse_cues('```json\n{"start": "Breathe.", "checkin": ""}\n```', p)
    assert cues["start"] == "Breathe." and cues["checkin"] == p.fallback["checkin"] and cues["finish"] == p.fallback["finish"]
    assert md.parse_cues("not json", p) == {c: p.fallback[c] for c in md.CUES}


def test_single_round_practices_have_no_checkin():
    assert md.cue_names(md.PRACTICES["sleep"]) == ["start", "finish"]
    assert md.cue_names(md.PRACTICES["breathing"], rounds=1) == ["start", "finish"]
    assert md.cue_names(md.PRACTICES["body_scan"]) == ["start", "checkin", "finish"]


def test_write_cues_without_key_uses_builtin(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    p = md.PRACTICES["pause"]
    assert md.write_cues(p, "Sam", "", 5, 1) == p.fallback


def test_timer_rounds():
    html = md.timer_html(md.PRACTICES["body_scan"], 3, 4, {"start": b"mp3"}, "001-body_scan")
    phases = json.loads(html.split("const P = ")[1].split(", S = ")[0])
    assert [p["sec"] for p in phases] == [180] * 4
    assert [p["cue"] for p in phases] == ["start", "checkin", "checkin", "checkin"]
    assert "data:audio/mpeg;base64,bXAz" in html
