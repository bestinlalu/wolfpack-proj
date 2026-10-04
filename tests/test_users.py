"""Static demo users."""
from bodylab.users import authenticate, check_password, find_by_email, load_users, password_required


def test_users_map_to_distinct_participants():
    users = load_users()
    pids = [u.pid for u in users]
    assert len(pids) == len(set(pids))
    assert {"001", "002", "004", "005", "006"} <= set(pids)
    assert len({u.username for u in users}) == len(users)


def test_email_match_is_case_insensitive():
    users = load_users()
    tagged = [u.__class__(**{**u.__dict__, "email": "sam@example.edu"}) if u.username == "sam" else u for u in users]
    assert find_by_email(tagged, "SAM@example.edu").username == "sam"
    assert find_by_email(tagged, "nobody@example.edu") is None
    assert find_by_email(tagged, None) is None


def test_password(monkeypatch):
    monkeypatch.delenv("BODYLAB_DEMO_PASSWORD", raising=False)
    assert not password_required() and not check_password("")
    monkeypatch.setenv("BODYLAB_DEMO_PASSWORD", "lab-demo")
    assert password_required()
    assert check_password("lab-demo") and not check_password("wrong")


def test_username_password_login(monkeypatch):
    users = load_users()
    monkeypatch.setenv("BODYLAB_DEMO_PASSWORD", "lab-demo")
    assert authenticate(users, " SAM ", "lab-demo").username == "sam"
    assert authenticate(users, "sam", "wrong") is None
    assert authenticate(users, "unknown", "lab-demo") is None
    monkeypatch.delenv("BODYLAB_DEMO_PASSWORD", raising=False)
    assert authenticate(users, "sam", "") is None
