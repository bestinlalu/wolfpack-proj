import pandas as pd

from bodylab import celebrity_cases as cases


def features():
    return {"fuel": pd.DataFrame([
        dict(sid=f"meal{i}", end_ts=pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
             steps_after=100 * i, steps_before=100 * i, response=30 - i, good_data=i != 5)
        for i in range(1, 9)
    ])}


def test_targets_ignore_future_and_bad_data():
    leads = cases.suggest(features(), pd.Timestamp("2020-01-05"))
    messi = next(q for q in leads if q["case_id"] == "messi")
    assert messi["threshold"] == 250
    assert "4 recorded" in messi["basis"]
    assert not cases.suggest(features(), pd.Timestamp("2020-01-03"))


def test_accept_track_complete_and_rewind(tmp_path):
    start = pd.Timestamp("2020-01-05")
    lead = cases.suggest(features(), start)[0]
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
