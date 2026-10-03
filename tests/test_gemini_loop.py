"""The Gemini tool-calling loop, driven by a scripted fake client (no network)."""
from types import SimpleNamespace

import pandas as pd
import pytest
from google.genai import types

from bodylab.agent import investigator as inv
from bodylab.data.synthetic import generate
from bodylab.engine import Engine


def _response(name, args):
    part = types.Part.from_function_call(name=name, args=args)
    return SimpleNamespace(function_calls=[SimpleNamespace(name=name, args=args)],
                           candidates=[SimpleNamespace(content=types.Content(role="model", parts=[part]))])


class ScriptedModels:
    """Plays the role of Gemini: follows the investigation order using real tool results."""

    def __init__(self):
        self.step = 0
        self.sid = None
        self.similar = []

    def generate_content(self, model, contents, config):
        last = contents[-1].parts[0]
        if self.step == 0:
            self.sid = last.text.split('"situation_id": "')[1].split('"')[0]
            self.step = 1
            return _response("check_data_quality", {"situation_id": self.sid})
        result = last.function_response.response["result"]
        if self.step == 1:
            if not result["passed"]:
                return _response("close_case", {"verdict": "bad_data", "message": "Bad data."})
            self.step = 2
            return _response("find_similar_situations", {"situation_id": self.sid})
        if self.step == 2:
            self.similar = [s["sid"] for s in result["similar"]]
            if not self.similar:
                return _response("close_case", {"verdict": "unexplained", "message": "Nothing to compare."})
            self.step = 3
            return _response("compare_situations", {"situation_id": self.sid, "similar_ids": self.similar})
        if self.step == 3:
            top = result["differences"][0]
            if abs(top["difference_z"]) < 1.5:
                return _response("close_case", {"verdict": "unexplained", "message": "Normal variation."})
            self.step = 4
            sign = 1 if (top["this_time"] - top["similar_median"]) * (1 if result["response_direction"] == "higher" else -1) > 0 else -1
            self.factor = top["label"]
            return _response("open_hypothesis", {"situation_id": self.sid, "factor": top["factor"], "direction": sign})
        return _response("close_case", {"verdict": "lead", "title": f"Lead on {self.factor}", "message": f"{self.factor} was different. 2 more tests needed."})


class FakeClient:
    def __init__(self, api_key=None, http_options=None):
        self.models = None


@pytest.fixture
def gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    import google.genai
    monkeypatch.setattr(google.genai, "Client", FakeClient)
    g = inv.GeminiInvestigator(model="fake")

    class PerCase:
        name = "gemini"

        def investigate(self, ctx, event):
            g.client.models = ScriptedModels()
            return g.investigate(ctx, event)

    return PerCase()


def test_gemini_loop_closes_cases(gemini):
    minute, meals = generate(days=9)
    eng = Engine("S01", minute, meals, investigator=gemini)
    eng.run(step_hours=12)
    events = eng.notebook.events
    assert events
    assert all(e["verdict"] in ("lead", "unexplained", "bad_data") for e in events)
    assert not any("fallback" in e["agent"] for e in events), "the scripted loop should never fall back to rules"
    leads = [e for e in events if e["verdict"] == "lead"]
    assert leads and all(e["title"].startswith("Lead on") for e in leads)
    assert eng.notebook.hypotheses
