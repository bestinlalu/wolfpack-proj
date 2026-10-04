import pandas as pd

from bodylab.agent.chat import BodyLabChat, _clean
from bodylab.agent.notebook import Notebook


def test_clean_handles_pandas_values():
    assert _clean(pd.Timestamp("2026-10-03 12:00")) == "2026-10-03T12:00:00"
    assert _clean(float("nan")) is None


def test_evidence_packet_is_grounded_and_bounded():
    nb = Notebook("demo")
    nb.hypotheses.append({
        "pid": "demo", "hyp_id": "H1", "lab": "fuel", "claim": "More walking → lower glucose rise",
        "status": "testing", "supports": 2, "contradicts": 1, "chances": 3,
        "opened_at": pd.Timestamp("2026-10-01"), "last_tested_at": pd.Timestamp("2026-10-03"),
        "closed_at": pd.NaT, "cross_lab": True, "factor": "steps_before",
    })
    nb.events.append({
        "pid": "demo", "event_id": "E1", "sid": "demo-fuel-1", "lab": "fuel",
        "ts": pd.Timestamp("2026-10-03 12:00"), "title": "Lunch spike", "message": "Higher than usual",
        "verdict": "lead", "z": 1.8, "hyp_id": "H1",
    })
    chat = object.__new__(BodyLabChat)
    chat.notebook = nb
    chat.features = {"fuel": pd.DataFrame([{"ts": pd.Timestamp("2026-10-03"), "z": 1.2}])}
    chat.until = pd.Timestamp("2026-10-03 20:00")

    packet = chat.evidence_packet()
    assert packet["summary"]["testing_hypotheses"] == 1
    assert packet["hypotheses"][0]["hyp_id"] == "H1"
    assert packet["recent_cases"][0]["title"] == "Lunch spike"
    assert packet["lab_coverage"][0]["lab"]
    assert "pid" not in str(packet)
