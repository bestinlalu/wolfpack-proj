"""Voices tab helpers (no Gemini or ElevenLabs calls)."""
import json

from bodylab import voice_sessions as vs


def test_parse_cues_falls_back_for_missing_or_bad():
    mode = vs.MODES["focus"]
    cues = vs.parse_cues('```json\n{"start": "Go.", "break": ""}\n```', mode)
    assert cues["start"] == "Go." and cues["break"] == mode.fallback["break"] and cues["finish"] == mode.fallback["finish"]
    assert vs.parse_cues("not json", mode) == {c: mode.fallback[c] for c in vs.CUES}


def test_sleep_has_no_break_cues():
    mode = vs.MODES["sleep"]
    assert vs.cue_names(mode) == ["start", "finish"]
    assert vs.parse_cues('{"break": "x"}', mode)["break"] == ""


def test_write_cues_without_key_uses_builtin(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    mode = vs.MODES["drive"]
    assert vs.write_cues(mode, "Sam", "", 50, 10) == mode.fallback


def test_timer_phases():
    html = vs.timer_html(vs.MODES["focus"], 25, 5, 4, {"start": b"mp3"}, "001-focus")
    phases = json.loads(html.split("const P = ")[1].split(", S = ")[0])
    assert [p["sec"] for p in phases] == [1500, 300] * 3 + [1500]
    assert [p["cue"] for p in phases][:3] == ["start", "break", "resume"]
    assert "data:audio/mpeg;base64,bXAz" in html
