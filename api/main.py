from __future__ import annotations

import json
import math
import secrets
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bodylab.agent.chat import BodyLabChat
from bodylab.config import gemini_api_key
from bodylab.labs import LABS
from bodylab.store import open_store
from bodylab.users import authenticate, load_users, password_required

app = FastAPI(title="Body Lab API", version="1.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# Mobile presentation name requested by the product UI. The underlying lab key remains
# "movement" everywhere in the existing Body Lab engine and stored data.
LAB_NAMES = {"fuel": "Fuel", "stress": "Stress", "movement": "Rhythm", "sleep": "Sleep"}
LAB_SITUATIONS = {"fuel": "meal", "stress": "2-hour window", "movement": "walk", "sleep": "night"}


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 3)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def rows(items) -> list[dict]:
    return [clean(dict(item)) for item in items]


def parse_json(value, default):
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def load_participant(pid: str):
    store = open_store()
    if pid not in store.pids():
        raise HTTPException(status_code=404, detail=f"Participant {pid!r} was not found")
    minute, meals = store.read_inputs(pid)
    features, notebook, until, agent = store.read_state(pid)
    if until is None:
        until = pd.to_datetime(minute["ts"]).min() if len(minute) and "ts" in minute.columns else pd.Timestamp.now()
    return store, minute, meals, features, notebook, pd.Timestamp(until), agent


def lab_payload(nb) -> list[dict]:
    # Same counts shown in the browser's "Your labs" cards.
    order = ["fuel", "stress", "movement", "sleep"]
    out = []
    for key in order:
        cards = sum(d.get("lab") == key and d.get("status") != "rejected" for d in nb.discoveries)
        open_h = sum(h.get("lab") == key and h.get("status") == "testing" for h in nb.hypotheses)
        out.append({"key": key, "name": LAB_NAMES[key], "situation": LAB_SITUATIONS[key], "cards": cards, "open": open_h})
    return out



# Demo authentication. Tokens live only in server memory and are invalidated when
# the API restarts. Each token is bound to exactly one participant.
SESSIONS: dict[str, dict[str, Any]] = {}
SESSION_TTL_SECONDS = 12 * 60 * 60


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


def current_session(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Sign in required")
    token = authorization[7:].strip()
    session = SESSIONS.get(token)
    if not session or session["expires_at"] <= time.time():
        SESSIONS.pop(token, None)
        raise HTTPException(status_code=401, detail="Session expired. Please sign in again.")
    return session


def authorize_pid(pid: str, session: dict[str, Any]) -> None:
    if session["pid"] != pid:
        raise HTTPException(status_code=403, detail="This account cannot access that participant")

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    participant_id: str = Field(min_length=1)
    message: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessage] = Field(default_factory=list)


@app.get("/api/health")
def health():
    return {"ok": True, "gemini_configured": bool(gemini_api_key())}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    if not password_required():
        raise HTTPException(
            status_code=503,
            detail="BODYLAB_DEMO_PASSWORD is not configured",
        )

    users = load_users()
    user = authenticate(users, req.username, req.password)

    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "username": user.username,
        "name": user.name,
        "pid": user.pid,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }

    return {
        "token": token,
        "user": {
            "username": user.username,
            "name": user.name,
            "participant_id": user.pid,
        },
    }
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {
        "username": user.username, "name": user.name, "pid": user.pid,
        "expires_at": time.time() + SESSION_TTL_SECONDS,
    }
    return {"token": token, "user": {"username": user.username, "name": user.name, "participant_id": user.pid}}


@app.get("/api/auth/me")
def me(session: dict[str, Any] = Depends(current_session)):
    return {"user": {"username": session["username"], "name": session["name"], "participant_id": session["pid"]}}


@app.get("/api/today/{pid}")
def today(pid: str, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(pid, session)
    _, _, _, _, nb, now, agent = load_participant(pid)
    rank, points, next_rank = nb.rank()
    feed = sorted(nb.messages, key=lambda m: pd.Timestamp(m["ts"]), reverse=True)[:6]
    week = [e for e in nb.events if pd.Timestamp(e["ts"]) > now - pd.Timedelta(days=7)]
    unexplained = sum(e.get("verdict") == "unexplained" for e in week)
    bad_data = sum(e.get("verdict") == "bad_data" for e in week)
    return clean({
        "participant_id": pid,
        "as_of": now,
        "agent": agent,
        "rank": {"name": rank, "points": points, "next_rank_points": next_rank},
        "messages": feed,
        "closed_quietly": {"unexplained": unexplained, "bad_data": bad_data},
        "quests": list(nb.quests[-2:]),
        "labs": lab_payload(nb),
    })


@app.get("/api/cases/{pid}")
def cases(pid: str, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(pid, session)
    _, _, _, _, nb, _, _ = load_participant(pid)
    items = list(nb.events)
    items.sort(key=lambda e: pd.Timestamp(e.get("ts")), reverse=True)
    return {"cases": rows(items)}


@app.get("/api/case/{pid}/{event_id}")
def case_detail(pid: str, event_id: str, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(pid, session)
    _, _, _, _, nb, _, _ = load_participant(pid)
    ev = next((e for e in nb.events if e.get("event_id") == event_id), None)
    if ev is None:
        raise HTTPException(status_code=404, detail="Case not found")
    item = clean(dict(ev))
    item["checks"] = clean(parse_json(ev.get("checks_json"), []))
    item["differences"] = clean(parse_json(ev.get("differences_json"), []))
    item["tools"] = clean(parse_json(ev.get("tools_json"), []))
    h = next((h for h in nb.hypotheses if h.get("hyp_id") == ev.get("hyp_id")), None)
    item["hypothesis"] = clean(dict(h)) if h else None
    item["lab_name"] = LAB_NAMES.get(ev.get("lab"), ev.get("lab"))
    return {"case": item}


@app.get("/api/discoveries/{pid}")
def discoveries(pid: str, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(pid, session)
    _, _, _, _, nb, _, _ = load_participant(pid)
    shown = [d for d in nb.discoveries if d.get("status") != "rejected"]
    rarity_order = {"legendary": 0, "rare": 1, "common": 2}
    shown.sort(key=lambda d: (rarity_order.get(d.get("rarity"), 9), str(d.get("confirmed_at"))))
    counts = {r: sum(d.get("rarity") == r for d in shown) for r in ("legendary", "rare", "common")}
    testing_close = [h for h in nb.hypotheses if h.get("status") == "testing" and (h.get("supports") or 0) >= 2]
    rejected = [d.get("title") for d in nb.discoveries if d.get("status") == "rejected"]
    payload = rows(shown)
    for d in payload:
        d["lab_name"] = LAB_NAMES.get(d.get("lab"), d.get("lab"))
    return {"discoveries": payload, "counts": counts, "close": rows(testing_close), "rejected_titles": rejected}


@app.get("/api/notebook/{pid}")
def notebook(pid: str, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(pid, session)
    _, _, _, _, nb, _, _ = load_participant(pid)
    order = {"testing": 0, "fading": 1, "confirmed": 2, "inconclusive": 3, "expired": 4, "rejected": 5}
    hyps = sorted(nb.hypotheses, key=lambda h: (order.get(h.get("status"), 9), h.get("hyp_id", "")))
    payload = []
    for h in hyps:
        item = clean(dict(h))
        item["lab_name"] = LAB_NAMES.get(h.get("lab"), h.get("lab"))
        item["evidence"] = clean([dict(e) for e in nb.evidence if e.get("hyp_id") == h.get("hyp_id") and e.get("verdict") != "neutral"][-10:])
        payload.append(item)
    testing = [h for h in nb.hypotheses if h.get("status") == "testing"]
    return clean({
        "hypotheses": payload,
        "open_slots": len(testing),
        "meal_related_open": sum(bool(h.get("meal_related")) for h in testing),
        "funnel": nb.funnel(),
        "situations_watched": len(nb.processed),
        "situations_good_data": sum(bool(p.get("good_data", True)) for p in nb.processed),
    })


# Kept for compatibility with the first mobile build. It now returns the same notebook data.
@app.get("/api/hypotheses/{pid}")
def hypotheses(pid: str, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(pid, session)
    _, _, _, _, nb, _, _ = load_participant(pid)
    order = {"testing": 0, "fading": 1, "confirmed": 2, "inconclusive": 3, "expired": 4, "rejected": 5}
    hyps = sorted(nb.hypotheses, key=lambda h: (order.get(h.get("status"), 9), h.get("hyp_id", "")))
    return {"hypotheses": rows(hyps)}


@app.post("/api/chat")
def chat(req: ChatRequest, session: dict[str, Any] = Depends(current_session)):
    authorize_pid(req.participant_id, session)
    if not gemini_api_key():
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")
    _, _, _, features, nb, until, _ = load_participant(req.participant_id)
    try:
        bot = BodyLabChat(notebook=nb, features=features, until=until)
        answer = bot.ask(req.message, [m.model_dump() for m in req.history])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Ask Body Lab failed: {exc}") from exc
    return {"answer": answer, "as_of": clean(until)}
