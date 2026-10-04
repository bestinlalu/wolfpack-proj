"""Static demo users, each assigned one participant's data (bodylab/users.json).

Sign-in is for the demo, not real accounts: a username and one shared password from BODYLAB_DEMO_PASSWORD (login disabled when unset).
On Databricks Apps, a user whose `email` matches the signed-in workspace account is signed in automatically.
"""
from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass
from pathlib import Path

USERS_FILE = Path(__file__).with_name("users.json")


@dataclass(frozen=True)
class User:
    username: str
    name: str
    pid: str
    email: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} · participant {self.pid}"


def load_users(path: Path = USERS_FILE) -> list[User]:
    return [User(**u) for u in json.loads(path.read_text())]


def find_by_username(users: list[User], username: str | None) -> User | None:
    if not username:
        return None
    normalized = username.strip().lower()
    return next((u for u in users if u.username.lower() == normalized), None)


def find_by_email(users: list[User], email: str | None) -> User | None:
    if not email:
        return None
    return next((u for u in users if u.email and u.email.lower() == email.strip().lower()), None)


def password_required() -> bool:
    return bool(os.getenv("BODYLAB_DEMO_PASSWORD"))


def check_password(attempt: str) -> bool:
    expected = os.getenv("BODYLAB_DEMO_PASSWORD", "")
    return bool(expected) and hmac.compare_digest(attempt.encode(), expected.encode())


def authenticate(users: list[User], username: str, password: str) -> User | None:
    user = find_by_username(users, username)
    if user is None or not check_password(password):
        return None
    return user
