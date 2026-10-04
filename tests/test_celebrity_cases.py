import pandas as pd

from bodylab import celebrity_cases as cases


def hypotheses():
    return [dict(hyp_id="H1", lab="fuel", factor="steps_after", direction=-1, status="testing", opened_at=pd.Timestamp("2020-01-01"))]


def features():
    return {"fuel": pd.DataFrame([
        dict(sid=f"meal{i}", end_ts=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
             steps_after=100 * i, steps_before=100 * i, response=30 - i, good_data=i != 5)
        for i in range(1, 9)
    ])}


def test_targets_ignore_future_and_bad_data():
    leads = cases.suggest(features(), pd.Timestamp("2020-01-05"), hypotheses())
    messi = next(q for q in leads if q["case_id"] == "messi")
    assert messi["threshold"] == 250
    assert "4 recorded" in messi["basis"]
    assert not cases.suggest(features(), pd.Timestamp("2020-01-03"), hypotheses())


def test_only_related_active_hypotheses_get_leads():
    now = pd.Timestamp("2020-01-05")
    assert not cases.suggest(features(), now, [])
    for status in ("confirmed", "rejected", "inconclusive", "expired"):
        h = {**hypotheses()[0], "status": status}
        assert not cases.suggest(features(), now, [h])
    assert not cases.suggest(features(), now, [{**hypotheses()[0], "direction": 1}])
    assert not cases.suggest(features(), now, [{**hypotheses()[0], "factor": "start_glucose"}])
    assert len(cases.suggest(features(), now, [{**hypotheses()[0], "status": "fading"}])) == 1


def test_acceptance_exhausts_relevant_leads_including_legacy_quests():
    now = pd.Timestamp("2020-01-05")
    leads = cases.suggest(features(), now, hypotheses())
    assert [lead["case_id"] for lead in leads] == ["messi"]
    quests = []
    cases.accept(quests, leads[0], now)
    assert not cases.suggest(features(), now, hypotheses(), quests)
    legacy = [{k: v for k, v in quests[0].items() if k not in ("hyp_id", "direction")}]
    assert not cases.suggest(features(), now, hypotheses(), legacy)


def test_celebrity_questions_and_patterns_are_distinct():
    assert len({c[6] for c in cases.CASES}) == len(cases.CASES)
    assert len({(c[2], c[3], cases.CASE_DIRECTIONS[c[0]]) for c in cases.CASES}) == len(cases.CASES)


def test_accept_track_complete_and_rewind(tmp_path):
    start = pd.Timestamp("2020-01-05")
    lead = cases.suggest(features(), start, hypotheses())[0]
    quests = []
    assert cases.accept(quests, lead, start)
    assert not cases.accept(quests, lead, start)
    cases.update(quests, features(), pd.Timestamp("2020-01-08"))
    assert quests[0]["progress"] == 2  # Bad-data meal5 is excluded.
    cases.update(quests, features(), pd.Timestamp("2020-01-08"))
    assert quests[0]["progress"] == 2  # No double-counting on Refresh.
    cases.update(quests, features(), pd.Timestamp("2020-01-09"))
    assert quests[0]["done"] and quests[0]["progress"] == 3
    cases.save("S01", quests, tmp_path)
    assert cases.load("S01", tmp_path) == quests
    assert cases.load("S02", tmp_path) == []
    cases.update(quests, features(), pd.Timestamp("2020-01-03"))
    assert quests[0]["progress"] == 0 and not quests[0]["done"]
