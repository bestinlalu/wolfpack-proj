"""Sherlock Howls web app (Streamlit). Run locally with `streamlit run app/app.py`, or as a Databricks App."""
from __future__ import annotations

import base64
import json
import os
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bodylab import celebrity_cases, meal_photo, voice  # noqa: E402
from bodylab import patterns  # noqa: E402
from bodylab import voice_sessions  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from bodylab.agent.investigator import make_investigator  # noqa: E402
from bodylab.agent.chat import BodyLabChat  # noqa: E402
from bodylab.formatting import fmt as _fmt  # noqa: E402
from bodylab.config import elevenlabs_api_key, gemini_api_key  # noqa: E402
from bodylab.engine import Engine  # noqa: E402
from bodylab.labs import LABS  # noqa: E402
from bodylab.pipeline.features import Signals  # noqa: E402
from bodylab.store import open_store  # noqa: E402
from bodylab.stress_scale import EXPLAINER, band  # noqa: E402
from bodylab.users import authenticate, find_by_email, password_required  # noqa: E402

st.set_page_config(page_title="Sherlock Howls", page_icon="🐺", layout="wide")

INK, INK3, GLU, ACC, OK = "#262321", "#756b61", "#b83a32", "#b5121b", "#2c8556"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --ink:#262321; --ink2:#574e46; --ink3:#756b61; --line:rgba(69,50,35,.16); --card:#fffdf8; --sunk:#eee7dc;
  --acc:#b5121b; --accs:#f8e3e3; --glu:#b83a32; --glus:#fbe6dd; --ok:#2c8556; --oks:#e0f1e7; --warn:#a86c12; --warns:#f8ecd6; --gold:#9a6d05; --golds:#f7edcf; }
[data-testid="stAppViewContainer"] { background:#f5f0e7; color:var(--ink); }
[data-testid="stSidebar"] { background:#eee7dc; border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stTabs"] [role="tablist"] { border-bottom:1px solid var(--line); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--acc); }
[data-testid="stExpander"] { background:transparent; }
[data-testid="stExpander"] details { background:var(--card); border-color:var(--line); border-radius:12px; overflow:hidden; }
.sh-brand { display:flex; gap:12px; align-items:center; margin-bottom:20px; }
.sh-brand svg { width:48px; height:48px; flex-shrink:0; }
.sh-brand-name { font-family:'Bricolage Grotesque',system-ui,sans-serif; font-weight:700; font-size:23px; letter-spacing:-.03em; line-height:1.1; }
.sh-tagline { font-size:12px; color:var(--ink2); margin-top:5px; }
.case-number { font:11px 'IBM Plex Mono',monospace; color:var(--ink3); letter-spacing:.06em; margin-bottom:8px; }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Bricolage Grotesque', system-ui, sans-serif !important; letter-spacing: -0.015em; }
.bl-card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin-bottom: 12px; color: var(--ink); box-shadow:0 2px 3px rgba(69,50,35,.035); }
.bl-card.case-file { border-top:3px solid var(--ink3); }
.lab-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); grid-auto-rows:1fr; gap:12px; margin-bottom:16px; }
.lab-grid .bl-card { display:flex; flex-direction:column; margin:0; min-width:0; }
.lab-details { min-height:40px; margin-bottom:10px; }
.lab-badges { display:flex; flex-wrap:wrap; gap:4px; margin-top:auto; }
.lab-badges .pill { margin-right:0; }
.bl-card.glu { border: 1.5px solid var(--glu); } .bl-card.ok { border: 1.5px solid var(--ok); } .bl-card.acc { border: 1.5px solid var(--acc); }
.bl-card.gold { border: 1.5px solid var(--gold); } .bl-card.fading { opacity: .7; } .bl-card.locked { border: 1.5px dashed var(--line); background: transparent; color: var(--ink3); }
.bl-label { font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink3); margin-bottom: 6px; }
.bl-title { font-weight: 600; font-size: 15.5px; margin: 6px 0 4px; }
.bl-sub { color: var(--ink2); font-size: 13px; }
.bl-big { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 28px; line-height: 1; }
.bl-num { font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 999px; margin-right: 4px; }
.p-acc { background: var(--accs); color: var(--acc); } .p-glu { background: var(--glus); color: var(--glu); } .p-ok { background: var(--oks); color: var(--ok); }
.p-warn { background: var(--warns); color: var(--warn); } .p-gold { background: var(--golds); color: var(--gold); } .p-plain { background: var(--sunk); color: var(--ink2); }
.bar { height: 6px; background: var(--sunk); border-radius: 3px; overflow: hidden; margin-top: 6px; } .bar i { display: block; height: 100%; border-radius: 3px; }
.dots i { display: inline-block; width: 13px; height: 13px; border-radius: 50%; border: 1.5px solid var(--line); margin-right: 4px; vertical-align: middle; }
.dots i.s { background: var(--ok); border-color: var(--ok); } .dots i.c { background: var(--glu); border-color: var(--glu); }
.pattern-evidence { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:10px; }
.pattern-evidence .dots { display:flex; flex-wrap:wrap; gap:4px; }
.pattern-evidence .dots i { display:inline-flex; align-items:center; justify-content:center; width:16px; height:16px; margin:0; color:#fff; font:700 11px sans-serif; font-style:normal; }
.pattern-evidence .dots i.c { border-radius:50%; }
.check::before { content: "✓"; color: var(--ok); font-weight: 700; margin-right: 8px; } .fail::before { content: "✕"; color: var(--glu); font-weight: 700; margin-right: 8px; }
</style>
""", unsafe_allow_html=True)

LAB_PILL = {"fuel": "p-glu", "stress": "p-warn", "sleep": "p-acc", "movement": "p-ok"}
RARITY_PILL = {"legendary": ("p-gold", "gold"), "rare": ("p-acc", "acc"), "common": ("p-plain", "")}


def card(body: str, cls: str = "") -> None:
    st.markdown(f'<div class="bl-card {cls}">{body}</div>', unsafe_allow_html=True)


def pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{escape(text)}</span>'


def brand() -> None:
    # A code-native wolf mark inside a magnifying glass; no external image dependency.
    st.markdown('''<div class="sh-brand"><svg viewBox="0 0 64 64" role="img" aria-label="Wolf detective logo">
        <circle cx="27" cy="27" r="21" fill="#fffdf8" stroke="#b5121b" stroke-width="4"/>
        <path d="M43 43 L58 58" stroke="#262321" stroke-width="7" stroke-linecap="round"/>
        <path d="M13 15 L22 20 L32 20 L41 15 L38 32 L27 40 L16 32 Z" fill="#262321"/>
        <path d="M18 26 L24 28 L20 30 M36 26 L30 28 L34 30" fill="#fffdf8"/>
        <path d="M24 33 L30 33 L27 36 Z" fill="#fffdf8"/>
        </svg><div><div class="sh-brand-name">Sherlock Howls</div>
        <div class="sh-tagline">Your body leaves clues.</div></div></div>''', unsafe_allow_html=True)


def quest_body(q: dict) -> str:
    badges = pill(LABS[q["lab"]].name, LAB_PILL[q["lab"]])
    badges += pill("Completed" if q["done"] else "Tracking automatically", "p-ok" if q["done"] else "p-acc")
    dots = "".join(f'<i class="{"s" if i < q["progress"] else ""}"></i>' for i in range(int(q["target"])))
    return (badges + f'<div class="bl-title">{escape(q["title"])}</div>'
            f'<div class="pattern-evidence"><span class="dots">{dots}</span>'
            f'<span class="bl-sub">{q["progress"]} of {q["target"]}</span></div>')


def celebrity_quest_body(q: dict) -> str:
    badges = pill("Celebrity-inspired case", "p-plain")
    badges += pill("Completed" if q["done"] else "Tracking automatically", "p-ok" if q["done"] else "p-acc")
    dots = ''.join(f'<i class="{"s" if i < q["progress"] else ""}"></i>' for i in range(q["target"]))
    return (badges + f'<div class="bl-title">{escape(q["celebrity"])} · {escape(q["theme"])}</div>'
            f'<div class="bl-sub">{escape(q["question"])}</div>'
            f'<div class="bl-sub" style="margin-top:6px">{escape(q["target_text"])}</div>'
            f'<div class="pattern-evidence"><span class="dots">{dots}</span><span class="bl-sub">{q["progress"]} of {q["target"]} observations</span></div>'
            + ('<div class="bl-sub" style="margin-top:8px">Observations collected. Check the Casebook for tested patterns; completing a quest does not confirm this link.</div>' if q["done"] else ''))


def sherlock_leads(quests: list[dict], now: pd.Timestamp) -> None:
    visible_key = f"sherlock_visible_{pid}"
    if visible_key not in st.session_state:
        st.session_state[visible_key] = True
    with st.sidebar:
        visible = st.checkbox("Show Sherlock", key=visible_key)
    related = celebrity_cases.suggest(celebrity_features, now, nb.hypotheses)
    if not visible or not related:
        return
    asset = Path(__file__).parent / "assets" / "sherlock.png"
    png = base64.b64encode(asset.read_bytes()).decode("ascii")
    st.markdown(f'''<style>
        .st-key-sherlock-mascot {{ position:fixed; right:24px; bottom:18px; width:130px; z-index:1000; }}
        .st-key-sherlock-mascot button {{ background:transparent url("data:image/png;base64,{png}") center top / 290px auto no-repeat;
            height:190px; width:130px; padding:165px 0 0; border:0; color:var(--ink); font-weight:600; box-shadow:none; }}
        .st-key-sherlock-mascot button:hover {{ color:var(--acc); }}
        .st-key-sherlock-mascot button:focus-visible {{ outline:2px solid var(--acc); outline-offset:3px; }}
        [data-testid="stPopoverBody"]:has(#a-lead-from-sherlock) {{ width:380px !important; min-width:0 !important; max-width:calc(100vw - 24px) !important; }}
        @media (max-width:640px) {{ .st-key-sherlock-mascot {{ right:10px; bottom:12px; }} }}
        </style>''', unsafe_allow_html=True)
    with st.container(key="sherlock-mascot"):
        with st.popover("New lead", help="Open Sherlock's celebrity-inspired cases"):
            st.markdown("### A lead from Sherlock")
            notice = st.session_state.pop(f"celebrity_notice_{pid}", None)
            if notice:
                st.success(notice)
            available = celebrity_cases.suggest(celebrity_features, now, nb.hypotheses, quests)
            if available:
                lead = available[0]
                st.caption(f'Celebrity-inspired case · {lead["hyp_id"]}')
                st.write(f'The {lead["celebrity"]} case: {lead["question"]} Let’s investigate.')
                st.markdown(f'**Your target:** {lead["target_text"]}')
                st.caption(lead["basis"] + " I'll watch three new situations in your replay.")
                if eng.until is not None and now >= eng.end:
                    st.caption("This replay has ended. Start a new replay after accepting to collect new observations.")
                if st.button("Investigate", key=f"accept_celebrity_{pid}", type="primary"):
                    celebrity_cases.accept(quests, lead, now)
                    persist_celebrity_quests(quests)
                    st.session_state[f"celebrity_notice_{pid}"] = f'{lead["celebrity"]} case added to your quests on Today.'
                    st.rerun()
            else:
                st.write("No further leads currently. Your relevant cases are already in quests.")
            st.caption("Inspired challenges, based on your data. Findings are tested separately in the Casebook.")
            st.button("Later · minimize Sherlock", key=f"hide_sherlock_{pid}",
                      on_click=lambda: st.session_state.update({visible_key: False}))


def persist_celebrity_quests(quests: list[dict]) -> None:
    try:
        celebrity_cases.save(pid, quests)
    except OSError:
        st.warning("Quest saved for this session. Local storage is unavailable, so it may not survive a restart.")


def pattern_body(h: dict, *, show_name: bool = False, update: dict | None = None) -> str:
    status_cls = {"testing": "p-plain", "confirmed": "p-ok", "fading": "p-warn", "rejected": "p-glu"}
    badges = pill(LABS[h["lab"]].name, LAB_PILL[h["lab"]]) + pill(patterns.STATUS[h["status"]], status_cls.get(h["status"], "p-plain"))
    if h["status"] in ("testing", "fading"):
        badges += pill("Tracking automatically", "p-acc")
    wording = patterns.question(h)
    if h["status"] == "confirmed":
        wording = patterns.statement(h)
    title = patterns.name(h) if show_name else wording
    timestamp = f'<div class="bl-label" style="float:right">{pd.Timestamp(update["ts"]):%a %H:%M}</div>' if update else ""
    body = (f'<div class="case-number">CASE FILE · {escape(str(h["hyp_id"]))}</div>'
            + timestamp + badges + f'<div class="bl-title">{escape(title)}</div>')
    if show_name:
        body += f'<div class="bl-sub">{escape(wording)}</div>'
    reason = patterns.origin(h, nb.events)
    if update and h["status"] == "confirmed":
        from bodylab.engine import effect_text
        discovery = next((d for d in nb.discoveries if d["hyp_id"] == h["hyp_id"]), None)
        if discovery:
            reason = effect_text(discovery) or reason
    if reason and not show_name:
        body += f'<div class="bl-sub">{escape(reason)}</div>'
    evidence = [e for e in nb.evidence if e["hyp_id"] == h["hyp_id"] and e["verdict"] in ("supports", "contradicts")]
    dots = ""
    for e in evidence:
        matched = e["verdict"] == "supports"
        label = "Matched" if matched else "Didn’t match"
        ts = e.get("ts")
        if ts is not None and pd.notna(ts):
            label += f" · {pd.Timestamp(ts):%a %b %d}"
        dots += f'<i class="{"s" if matched else "c"}" title="{escape(label)}" aria-label="{escape(label)}">{"✓" if matched else "×"}</i>'
    body += (f'<div class="pattern-evidence"><span class="dots">{dots}</span>'
             f'<span class="bl-sub">{h["supports"]} matched</span></div>')
    return body


def case_file(h: dict) -> None:
    card(pattern_body(h), "case-file")
    with st.expander(f'Evidence log · {h["hyp_id"]}'):
        entries = [e for e in nb.evidence if e["hyp_id"] == h["hyp_id"]]
        if not entries:
            st.caption("No comparisons recorded yet.")
        for e in sorted(entries, key=lambda e: pd.Timestamp(e["ts"])):
            label = {"supports": "Matched", "contradicts": "Didn't match", "neutral": "No clear result"}.get(e["verdict"], "No clear result")
            st.write(f'{pd.Timestamp(e["ts"]):%a %b %d, %H:%M} · {label}')


def _level(v) -> int | None:
    try:
        return None if v is None or pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return None


def stress_scale_html(level, usual) -> str:
    """Ten cells, 1 (calmest for you) to 10; typical band shaded, this window filled, your usual outlined."""
    level, usual = _level(level), _level(usual)
    head = '<div class="bl-label">Stress level · your personal 1–10</div>'
    if level is None:
        return head + '<div class="bl-sub">Learning your baseline: the scale needs about 2 days of data.</div>'
    cells = ""
    for i in range(1, 11):
        bg = "var(--glu)" if i == level else "var(--sunk)" if 4 <= i <= 7 else "#fff"
        border = "2px solid var(--ink)" if i == usual else "1px solid var(--line)"
        color = "#fff" if i == level else "var(--ink2)"
        cells += (f'<div style="flex:1;height:30px;border-radius:6px;background:{bg};border:{border};color:{color};'
                  f'display:flex;align-items:center;justify-content:center;font:500 12px \'IBM Plex Mono\',monospace">{i}</div>')
    usual_txt = f" · your usual here: <b>{usual}/10</b> (outlined)" if usual is not None else ""
    return (head + f'<div style="display:flex;gap:4px;margin:6px 0">{cells}</div>'
            f'<div class="bl-sub bl-num" style="display:flex;justify-content:space-between"><span>calmer</span><span>typical</span><span>more stressed</span></div>'
            f'<div style="margin-top:8px">This window: <b>{level}/10</b>, {escape(band(level))}{usual_txt}</div>')


# ---------------------------------------------------------------- state
CACHE_SECONDS = 60  # Databricks mode: how long results are reused before the next interaction reloads them


@st.cache_resource(show_spinner="Connecting to Databricks…")
def get_store():
    return open_store()


# Cached reads are keyed by the results signature, so new agent results are picked up as soon as it changes.
@st.cache_data(ttl=CACHE_SECONDS, show_spinner="Loading users…")
def _cached_users(_store) -> list:
    return _store.users()


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _cached_inputs(_store, pid: str, signature: str):
    return _store.read_inputs(pid)


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def _cached_state(_store, pid: str, signature: str):
    return _store.read_state(pid)


# Local files are instant and change with every replay step, so only Databricks reads are cached.
store = get_store()
REFRESH_SECONDS = float(os.getenv("BODYLAB_REFRESH_SECONDS", "2"))
results_signature = store.signature() if store.read_only else ""
read_inputs = (lambda p: _cached_inputs(store, p, results_signature)) if store.read_only else store.read_inputs
read_state = (lambda p: _cached_state(store, p, results_signature)) if store.read_only else store.read_state

if store.read_only:
    @st.fragment(run_every=REFRESH_SECONDS)
    def sync_panel(seen: str, pid: str) -> None:
        """Runs on its own every few seconds without blocking the page: checks Databricks for new agent results,
        loads them in the background, and only then swaps them in (an instant rerun from the warm cache)."""
        # The status line always holds text (the last message until a new one replaces it), so its height never
        # changes and the rest of the sidebar doesn't jump while a check runs.
        status = st.empty()
        status.caption(st.session_state.get("sync_status", "● In sync with Databricks"))
        sync_now = st.button("⟳ Sync now", key="sync_now", use_container_width=True,
                             help="Fetch the latest results from Databricks right away")
        if any(v is True and (k in ("recap_pending", "story_pending") or k.startswith("voice_pending_"))
               for k, v in st.session_state.items()):
            _show(status, "⏸ Sync paused while your audio is prepared")
            return
        latest = store.signature()
        if sync_now or latest != seen:
            _show(status, "⟳ Syncing with Databricks…")
            if sync_now:
                _cached_inputs.clear()
                _cached_state.clear()
            _cached_inputs(store, pid, latest)
            _cached_state(store, pid, latest)
            st.session_state["synced"] = (pid, latest)
            st.rerun()
        _show(status, f"● In sync with Databricks · checked {pd.Timestamp.now():%H:%M:%S}")

    def _show(status, text: str) -> None:
        st.session_state["sync_status"] = text
        status.caption(text)

# Sign-in list: in Databricks mode the `users` table (all users, whether or not the agent has results for them yet);
# locally, users whose participant is prepared on this laptop.
users = _cached_users(store) if store.read_only else store.users()
if not users:
    st.title("Sherlock Howls")
    st.info("No user in bodylab/users.json has prepared data yet. Prepare a participant listed there, "
            "for example `python scripts/prepare.py --pid 001`.")
    st.stop()

if st.session_state.get("username") not in {u.username for u in users}:
    st.session_state.pop("username", None)
    auto = find_by_email(users, st.context.headers.get("X-Forwarded-Email"))  # Databricks Apps sign-in
    if auto is not None:
        st.session_state["username"] = auto.username

if "username" not in st.session_state:
    _, middle, _ = st.columns([1, 2, 1])
    with middle:
        brand()
        st.markdown("Your personal body detective. Sign in to open your casebook.")
        if not password_required():
            st.error("Login is not configured. Set BODYLAB_DEMO_PASSWORD in .env and restart Sherlock Howls.")
            st.stop()
        with st.form("sign_in"):
            entered_username = st.text_input("Username", placeholder="Enter your username", key="signin_username")
            attempt = st.text_input("Password", type="password", placeholder="Enter your password", key="signin_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            signed_in = authenticate(users, entered_username, attempt)
            if signed_in is not None:
                st.session_state["username"] = signed_in.username
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Each account is mapped to one participant. Participant selection is not exposed after sign-in.")
    st.stop()

user = next(u for u in users if u.username == st.session_state["username"])
pid = user.pid

with st.sidebar:
    brand()
    st.caption(f"Signed in as **{escape(user.name)}** · {escape(user.label.split(' · ')[1])}")
    if st.button("Sign out", use_container_width=True):
        st.session_state.pop("username", None)
        st.rerun()
    use_llm = st.toggle("Gemini agent", value=bool(gemini_api_key()), disabled=not gemini_api_key() or store.read_only,
                        help="Without GEMINI_API_KEY the rule-based investigator runs the same tools.")
    if store.read_only:
        sync_panel(results_signature, pid)


def load_engine(pid: str) -> Engine:
    minute, meals = read_inputs(pid)
    features, nb, until, _ = read_state(pid)
    eng = Engine(pid, minute, meals, nb, make_investigator(prefer_llm=use_llm))
    eng.features, eng.until = features, until
    return eng


first_load = store.read_only and st.session_state.get("synced") != (pid, results_signature)
loading_note = st.empty()
if first_load:
    loading_note.caption("Loading your casebook from Databricks…")
eng = load_engine(pid)
loading_note.empty()
if store.read_only:
    st.session_state["synced"] = (pid, results_signature)
if eng.minute.empty or pd.isna(eng.start):
    # Happens while a live replay restarts: 02_replayer cleared this participant and the stream hasn't refilled it yet.
    st.title(f"Hi, {user.name}")
    st.info("Your data is streaming in and nothing has arrived yet. This page updates by itself"
            + (" as soon as it does." if store.read_only else "; press Refresh in a moment."))
    if not store.read_only and st.button("Refresh"):
        st.rerun()
    st.stop()
nb = eng.notebook
now = eng.until or eng.start
celebrity_key = f"celebrity_quests_{pid}"
if celebrity_key not in st.session_state:
    try:
        st.session_state[celebrity_key] = celebrity_cases.load(pid)
    except (OSError, ValueError):
        st.session_state[celebrity_key] = []
        st.warning("Saved celebrity quests could not be loaded. New quests will be tracked in this session.")
celebrity_quests = st.session_state[celebrity_key]
bad_situations = {p["sid"] for p in nb.processed if not p.get("good_data", True)}
celebrity_features = {lab: df[~df["sid"].isin(bad_situations)] if "sid" in df else df
                      for lab, df in eng.features.items()}
previous_quests = json.dumps(celebrity_quests)
celebrity_cases.update(celebrity_quests, celebrity_features, now)
if json.dumps(celebrity_quests) != previous_quests:
    persist_celebrity_quests(celebrity_quests)
sherlock_leads(celebrity_quests, now)

with st.sidebar:
    st.markdown("#### Replay")
    pct = 0.0 if eng.until is None else (eng.until - eng.start) / (eng.end - eng.start)
    st.progress(min(max(pct, 0.0), 1.0), text=f"{now:%a %b %d, %H:%M}" if eng.until is not None else "Not started")
    if store.read_only:
        st.caption("Results stream in from Databricks automatically; use ⟳ Sync now above to fetch them right away.")
    else:
        c1, c2 = st.columns(2)
        step = None
        if c1.button("+6 hours", use_container_width=True):
            step = pd.Timedelta(hours=6)
        if c2.button("+1 day", use_container_width=True):
            step = pd.Timedelta(days=1)
        if c1.button("Play to end", use_container_width=True):
            bar = st.progress(0.0, text="Replaying")
            t = (eng.until or eng.start) + pd.Timedelta(hours=6)
            while t <= eng.end + pd.Timedelta(hours=6):
                eng.step(t)
                bar.progress(min((t - eng.start) / (eng.end - eng.start), 1.0), text=f"{min(t, eng.end):%a %H:%M}")
                t += pd.Timedelta(hours=6)
            store.write_state(pid, eng.features, eng.notebook, eng.until, eng.investigator.name)
            st.rerun()
        if c2.button("Reset", use_container_width=True):
            store.reset_state(pid)
            st.rerun()
        if step is not None:
            with st.spinner("Agent investigating"):
                eng.step((eng.until or eng.start) + step)
            store.write_state(pid, eng.features, eng.notebook, eng.until, eng.investigator.name)
            st.rerun()

    with st.expander("Log a meal by photo"):
        photo = st.file_uploader("Meal photo", type=["jpg", "jpeg", "png", "webp"], key="photo")
        if photo is not None and st.button("Estimate with Gemini", disabled=not gemini_api_key()):
            try:
                st.session_state["meal_guess"] = meal_photo.analyze(photo.getvalue(), photo.type or "image/jpeg")
            except Exception as exc:
                st.error(f"Could not read the photo: {exc}")
        guess = st.session_state.get("meal_guess")
        if guess:
            st.caption(f"Confidence: {guess['confidence']}. Photo carbs are rough, so this meal gets wider comparison ranges.")
            items = ", ".join(f"{f['name']} ({f['portion']})" for f in guess["foods"])
            carbs = st.number_input("Carbs (g)", value=float(round(guess["carbs_g"])), step=1.0)
            st.write(items)
            if not store.read_only and st.button("Add to meal log"):
                minute, meals = store.read_inputs(pid)
                row = {"pid": pid, "meal_id": f"{pid}-p{len(meals):03d}", "ts": now.floor("min"), "carbs": carbs, "carbs_missing": False,
                       "sugar": guess["sugar_g"], "fiber": guess["fiber_g"], "protein": guess["protein_g"], "fat": guess["fat_g"],
                       "calories": guess["calories"], "items": items}
                store.write_inputs(pid, minute, pd.concat([meals, pd.DataFrame([row])], ignore_index=True))
                st.session_state.pop("meal_guess")
                st.success("Added at the current replay time.")
    st.caption("Sherlock Howls reports what was different, never causes. Not medical advice.")

def _speak_cached(text: str) -> bytes | None:
    """ElevenLabs audio, kept for the session so the same line is never paid for twice."""
    cache = st.session_state.setdefault("voice_cache", {})
    if text not in cache:
        cache[text] = voice.speak(text)
    return cache[text]


def voices_tab() -> None:
    st.markdown("## Voices")
    st.caption("Guided sessions and bedtime stories. Gemini writes the words; ElevenLabs reads them.")
    if not elevenlabs_api_key():
        st.info("Set `ELEVENLABS_API_KEY` (and your own `ELEVENLABS_VOICE_ID`) to hear these read aloud.")

    def _ask(key: str) -> None:  # runs before the rerun, so the background sync pauses while we generate
        st.session_state[key] = True

    # ---- guided sessions
    st.markdown("### Guided sessions")
    mode = voice_sessions.MODES[st.radio("Session", list(voice_sessions.MODES), horizontal=True,
                                         format_func=lambda k: voice_sessions.MODES[k].name, key="voice_mode")]
    c1, c2, c3 = st.columns(3)
    work = c1.number_input(f"{mode.work_label} (min)", 1, 180, mode.work_min, key=f"vw_{mode.key}")
    rest = c2.number_input(f"{mode.rest_label} (min)", 0, 60, mode.rest_min, key=f"vr_{mode.key}",
                           disabled=mode.rounds == 1)
    rounds = c3.number_input("Rounds", 1, 8, mode.rounds, key=f"vn_{mode.key}")
    session_key = f"voice_session_{pid}_{mode.key}"
    pending = f"voice_pending_{mode.key}"
    st.button(f"🎙 Prepare {mode.name.lower()} session", key=f"prep_{mode.key}", on_click=_ask, args=(pending,))
    if st.session_state.get(pending):
        try:
            with st.spinner("Writing and recording your cues…"):
                cues = voice_sessions.write_cues(mode, user.name, voice.recap_text(nb, now), work, rest)
                audio, note = {}, ""
                for c in voice_sessions.cue_names(mode):
                    try:
                        audio[c] = _speak_cached(cues[c])
                    except Exception as exc:
                        note = f"Voice unavailable: {exc}"
            st.session_state[session_key] = {"cues": cues, "audio": audio, "note": note,
                                             "plan": (int(work), int(rest), int(rounds))}
        finally:
            st.session_state[pending] = False
    prepared = st.session_state.get(session_key)
    if prepared:
        w, r, n = prepared["plan"]
        if (w, r, n) != (work, rest, rounds):
            st.caption("Settings changed: prepare the session again to use them.")
        components.html(voice_sessions.timer_html(mode, w, r if mode.rounds > 1 else 0, n, prepared["audio"],
                                                  f"{pid}-{mode.key}"), height=200)
        if prepared["note"]:
            st.caption(prepared["note"])
        elif not any(prepared["audio"].values()):
            st.caption("No voice yet: the timer runs silently until ElevenLabs is set up.")
        with st.expander("What you'll hear"):
            for c in voice_sessions.cue_names(mode):
                st.markdown(f"**{c.title()}** · {prepared['cues'][c]}")
        if mode.key == "drive":
            st.caption("Start it before you set off, then keep your eyes on the road; the cues play by themselves.")

    # ---- bedtime stories
    st.markdown("### Bedtime story")
    if not gemini_api_key():
        st.info("Set `GEMINI_API_KEY` to have Gemini write bedtime stories.")
        return
    s1, s2 = st.columns([3, 1])
    theme = s1.text_input("Theme (optional)", placeholder="a lighthouse keeper, a slow train through snow…",
                          key="story_theme")
    length = s2.selectbox("Length", list(voice_sessions.STORY_LENGTHS), key="story_length")
    story_key = f"voice_story_{pid}"
    st.button("🌙 Tell me a story", key="story_go", on_click=_ask, args=("story_pending",))
    if st.session_state.get("story_pending"):
        try:
            with st.spinner("Writing tonight's story…"):
                title, story = voice_sessions.write_story(user.name, theme, voice_sessions.STORY_LENGTHS[length])
            note, audio = "", None
            with st.spinner("Recording it…"):
                try:
                    audio = _speak_cached(story)
                except Exception as exc:
                    note = f"Voice unavailable: {exc}"
            st.session_state[story_key] = {"title": title, "text": story, "audio": audio, "note": note}
        except Exception as exc:
            st.error(f"Couldn't write a story right now: {exc}")
        finally:
            st.session_state["story_pending"] = False
    told = st.session_state.get(story_key)
    if told:
        st.markdown(f"#### {escape(told['title'])}")
        if told["audio"]:
            st.audio(told["audio"], format="audio/mpeg")
        elif told["note"]:
            st.caption(told["note"])
        with st.expander("📖 Read the story", expanded=False):
            st.write(told["text"])


tab_today, tab_case, tab_disc, tab_nb, tab_chat, tab_voice = st.tabs(
    ["Today", "Cases", "Findings", "Casebook", "💬 Ask Sherlock Howls", "🎧 Voices"])
with tab_voice:
    voices_tab()

if eng.until is None:
    with tab_today:
        st.title("Sherlock Howls")
        st.markdown(f"Hi {escape(user.name)}. Your data runs from **{eng.start:%b %d}** to **{eng.end:%b %d}**. "
                    "Use the replay controls on the left to stream it through the agent.")
        _, quest_area = st.columns([3, 2], gap="large")
        with quest_area:
            if not celebrity_quests:
                card('<div class="bl-label">Quests · optional</div><div class="bl-sub">No quests currently</div>')
            for q in celebrity_quests:
                card('<div class="bl-label">Quest · optional</div>' + celebrity_quest_body(q))
    for tab in (tab_case, tab_disc, tab_nb, tab_chat):
        with tab:
            st.caption("Start the replay to open your casebook.")
    st.stop()


@st.cache_data(show_spinner=False)
def signals_frame(pid: str, until: str) -> pd.DataFrame:
    s = Signals(eng.minute, pd.Timestamp(until))
    return s.df.assign(stress=s.stress)


sig_df = signals_frame(pid, str(eng.until))

# ---------------------------------------------------------------- Today
with tab_today:
    rank, pts, nxt = nb.rank()
    day_start = now.normalize()
    st.markdown("## Daily briefing")
    st.caption(f"{user.name} · {now:%A, %b %d, %H:%M} in the replay")
    left, right = st.columns([3, 2], gap="large")
    with left:
        feed = sorted([m for m in nb.messages if day_start <= pd.Timestamp(m["ts"]) <= now],
                      key=lambda m: pd.Timestamp(m["ts"]), reverse=True)
        if not feed:
            card('<div class="bl-label">No new clues today</div><div class="bl-sub">New possible links and pattern updates will appear here.</div>')
        seen_patterns = set()
        displayed = 0
        for m in feed:
            if displayed >= 6:
                break
            kind = m["kind"]
            ref = m.get("ref")
            event = next((e for e in nb.events if e["event_id"] == ref), None) if kind == "case" else None
            discovery = next((d for d in nb.discoveries if d["card_id"] == ref), None) if kind == "discovery" else None
            hyp_id = event.get("hyp_id") if event else discovery["hyp_id"] if discovery else ref
            h = next((h for h in nb.hypotheses if h["hyp_id"] == hyp_id), None)
            if h:
                if h["hyp_id"] in seen_patterns:
                    continue
                seen_patterns.add(h["hyp_id"])
                card(pattern_body(h, update=m))
                displayed += 1
                continue
            cls, label = {"discovery": ("ok", ("Confirmed", "p-ok")), "case": ("glu", ("Possible link", "p-plain")),
                          "rejected": ("", ("Denied", "p-glu")), "fading": ("", ("Mixed evidence", "p-warn"))}[kind]
            title = m["title"].replace("Not confirmed:", "Denied:").replace("Fading:", "Mixed evidence:").replace("Case solved", "Possible link")
            card(f'{pill(label[0], label[1])}{pill(LABS[m["lab"]].name, LAB_PILL[m["lab"]])}'
                 f'<span class="bl-sub bl-num" style="float:right">{pd.Timestamp(m["ts"]):%a %H:%M}</span>'
                 f'<div class="bl-title">{escape(title)}</div><div class="bl-sub">{escape(m["body"])}</div>', cls)
            displayed += 1
        today_events = [e for e in nb.events if day_start <= pd.Timestamp(e.get("end_ts", e["ts"])) <= now]
        unexplained = sum(e["verdict"] == "unexplained" for e in today_events)
        bad = sum(e["verdict"] == "bad_data" for e in today_events)
        card(f'<div class="bl-label">Closed quietly today</div>'
             f'<div>{unexplained} surprise{"s" if unexplained != 1 else ""} with no clear reason · {bad} dismissed as bad data</div>'
             f'<div class="bl-sub">No alerts were sent for these. Details are in the Casebook.</div>')
    with right:
        to_next = f"{nxt - pts} points to the next rank" if nxt else "Top rank reached"
        width = 100 if not nxt else int(pts / nxt * 100)
        card(f'<div class="bl-label">Detective rank</div><div class="bl-big">{escape(rank)}</div>'
             f'<div class="bl-sub bl-num">{pts} points · {to_next}</div><div class="bar"><i style="width:{width}%;background:var(--acc)"></i></div>')
        if not nb.quests and not celebrity_quests:
            card('<div class="bl-label">Quests · optional</div><div class="bl-sub">No quests currently</div>')
        for q in nb.quests[-2:]:
            card('<div class="bl-label">Quest · optional</div>' + quest_body(q))
        for q in celebrity_quests:
            card('<div class="bl-label">Quest · optional</div>' + celebrity_quest_body(q))
        st.markdown('<div class="bl-label">Your labs</div>', unsafe_allow_html=True)
        lab_cards = []
        for lab in LABS.values():
            cards = sum(d["lab"] == lab.key and d["status"] != "rejected" for d in nb.discoveries)
            open_h = sum(h["lab"] == lab.key and h["status"] == "testing" for h in nb.hypotheses)
            extra = (f'<div class="bl-sub" title="{escape(EXPLAINER)}" style="cursor:help">stress 1–10, personal ⓘ</div>'
                     if lab.key == "stress" else "")
            lab_cards.append(f'<div class="bl-card"><div class="bl-title">{lab.name}</div>'
                             f'<div class="lab-details"><div class="bl-sub">{lab.situation}s</div>{extra}</div>'
                             f'<div class="lab-badges">{pill(f"{cards} cards", "p-acc")}{pill(f"{open_h} open", "p-plain")}</div></div>')
        st.markdown('<div class="lab-grid">' + ''.join(lab_cards) + '</div>', unsafe_allow_html=True)
        # The recap is kept in session state: the page reloads by itself when new results arrive (Databricks mode),
        # which would otherwise wipe a recap that only existed on the click's run.
        def _request_recap() -> None:
            st.session_state["recap_pending"] = True
            st.session_state.pop("recap", None)

        st.button("▶ Weekly recap", use_container_width=True, on_click=_request_recap)
        if st.session_state.get("recap_pending"):
            with st.spinner("Writing your weekly recap…"):
                text = voice.polish(voice.recap_text(nb, now))
                audio, note = None, ""
                try:
                    audio = voice.speak(text)
                    if not audio:
                        note = "Set ELEVENLABS_API_KEY to hear this read aloud."
                except Exception as exc:
                    note = f"Voice unavailable: {exc}"
            st.session_state["recap"] = {"pid": pid, "text": text, "audio": audio, "note": note}
            st.session_state["recap_pending"] = False
        recap = st.session_state.get("recap")
        if recap and recap["pid"] == pid:
            st.write(recap["text"])
            if recap["audio"]:
                st.audio(recap["audio"], format="audio/mpeg")
            if recap["note"]:
                st.caption(recap["note"])
            if st.button("✕ Close recap", key="close_recap"):
                st.session_state.pop("recap", None)
                st.rerun()

# ---------------------------------------------------------------- Case
def situation_row(sid: str | None) -> tuple[str, pd.Series] | None:
    if not sid:
        return None
    lab = sid.split("-")[1]
    df = eng.features.get(lab, pd.DataFrame())
    hit = df[df["sid"] == sid] if not df.empty else df
    return (lab, hit.iloc[0]) if len(hit) else None


def window(row: pd.Series, lab: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if lab == "fuel":
        return row["ts"] - pd.Timedelta(minutes=15), row["ts"] + pd.Timedelta(minutes=135), "glucose"
    if lab == "movement":
        return row["ts"] - pd.Timedelta(minutes=10), row["end_ts"] + pd.Timedelta(minutes=40), "hr"
    if lab == "sleep":
        night = row["night_of"] + pd.Timedelta(hours=20)
        return night, night + pd.Timedelta(hours=14), "hr"
    return row["ts"], row["end_ts"], "stress"


def trace(row: pd.Series, lab: str) -> tuple[np.ndarray, np.ndarray]:
    a, b, col = window(row, lab)
    seg = sig_df[(sig_df["ts"] >= a) & (sig_df["ts"] <= b)]
    y = seg[col]
    if col == "glucose":
        y = y.interpolate(limit=6, limit_area="inside")
    else:
        y = y.rolling(10 if lab != "sleep" else 30, min_periods=1, center=True).mean()
    x = (seg["ts"] - (row["ts"] if lab != "sleep" else a)).dt.total_seconds() / (60 if lab != "sleep" else 3600)
    return x.to_numpy(), y.to_numpy()


with tab_case:
    cases = sorted(nb.events, key=lambda e: pd.Timestamp(e["ts"]), reverse=True)
    if not cases:
        st.info("No cases investigated yet.")
    else:
        verdict_label = {"lead": "Possible link", "unexplained": "No clear link", "bad_data": "Bad data"}
        def case_hypothesis(event):
            return next((h for h in nb.hypotheses if h["hyp_id"] == event.get("hyp_id")), None)

        def case_label(event):
            h = case_hypothesis(event)
            return patterns.STATUS[h["status"]] if h else verdict_label.get(event["verdict"], event["verdict"])

        def case_title(event):
            h = case_hypothesis(event)
            return (patterns.statement(h) if h["status"] == "confirmed" else patterns.question(h)) if h else event["title"]

        options = {e["event_id"]: f'{case_label(e)} · {LABS[e["lab"]].name} · {pd.Timestamp(e["ts"]):%a %b %d %H:%M} · {case_title(e)[:90]}' for e in cases}
        keys = list(options)
        leads = [e["event_id"] for e in cases if e["verdict"] == "lead"]
        fuel_leads = [e["event_id"] for e in cases if e["verdict"] == "lead" and e["lab"] == "fuel"]
        default = (fuel_leads or leads or keys)[0]
        pick = st.selectbox("Case", keys, index=keys.index(default), format_func=options.get)
        ev = nb.event(pick)
        lab = ev["lab"]
        L = LABS[lab]
        h = case_hypothesis(ev)
        label_cls = {"Confirmed": "p-ok", "Mixed evidence": "p-warn", "Denied": "p-glu", "Bad data": "p-warn"}.get(case_label(ev), "p-plain")
        st.markdown(f'{pill(L.name + " Lab", LAB_PILL[lab])}{pill(case_label(ev), label_cls)}'
                    f'<span class="bl-sub bl-num">{escape(ev["event_id"])} · agent: {escape(str(ev.get("agent", "")))}</span>', unsafe_allow_html=True)
        st.markdown(f"### {escape(case_title(ev) or 'Case')}")
        if h:
            st.caption(f"Observation · {pd.Timestamp(ev['ts']):%a %b %d, %H:%M}")
            with st.expander("Explanation at the time"):
                st.write(ev["message"])
        else:
            st.markdown(f'<div class="bl-sub" style="font-size:15px">{escape(ev["message"])}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            this = situation_row(ev["sid"])
            comp = situation_row(ev.get("comparison_sid"))
            fig = go.Figure()
            if comp:
                x, y = trace(comp[1], lab)
                fig.add_trace(go.Scatter(x=x, y=y, name=f"Similar: {comp[1]['ts']:%a %b %d}", line=dict(color=INK3, width=2, dash="dash")))
            if this:
                x, y = trace(this[1], lab)
                fig.add_trace(go.Scatter(x=x, y=y, name=f"This: {this[1]['ts']:%a %b %d}", line=dict(color=GLU, width=3)))
            xlab = {"fuel": "minutes after eating", "movement": "minutes from walk start", "sleep": "hours after 20:00", "stress": "minutes into the window"}[lab]
            ylab = {"fuel": "glucose (mg/dL)", "movement": "heart rate (bpm)", "sleep": "heart rate (bpm)", "stress": "stress signal (0 = your usual)"}[lab]
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), xaxis_title=xlab, yaxis_title=ylab,
                              legend=dict(orientation="h", y=1.12), plot_bgcolor="white", font=dict(family="IBM Plex Sans"))
            fig.update_xaxes(gridcolor="#eef1f0")
            fig.update_yaxes(gridcolor="#eef1f0")
            st.plotly_chart(fig, use_container_width=True)
            tools = json.loads(ev.get("tools_json") or "[]")
            if tools:
                st.caption("Agent steps: " + " → ".join(tools))
        with c2:
            if lab == "stress":
                card(stress_scale_html(ev.get("level"), ev.get("usual_level")))
            checks = json.loads(ev.get("checks_json") or "[]")
            if checks:
                rows = "".join(f'<div class="{"check" if c["passed"] else "fail"}">{escape(c["check"])}: <span class="bl-sub">{escape(c["detail"])}</span></div>' for c in checks)
                card(f'<div class="bl-label">Step 1 · Data check</div>{rows}')
            diffs = json.loads(ev.get("differences_json") or "[]")
            if diffs:
                html = '<div class="bl-label">Step 2 · What was different</div>'
                for d in diffs[:5]:
                    z = abs(d.get("difference_z") or 0)
                    lvl = d["level"]
                    color = {"very unusual": "var(--glu)", "somewhat": "var(--warn)", "normal": "var(--ink3)"}[lvl]
                    html += (f'<div style="margin:8px 0"><b>{escape(d["label"])}</b> {pill(lvl, {"very unusual": "p-glu", "somewhat": "p-warn", "normal": "p-plain"}[lvl])}'
                             f'<div class="bar"><i style="width:{min(z / 4, 1) * 100:.0f}%;background:{color}"></i></div>'
                             f'<div class="bl-sub bl-num">this time {escape(d.get("this_time_text") or _fmt(d["this_time"], d["unit"]))} · '
                             f'typical {escape(d.get("typical_text") or _fmt(d["similar_median"], d["unit"]))}</div></div>')
                card(html)
            if lab == "stress" or any(d["unit"] == "z" for d in diffs[:5]):
                st.caption(EXPLAINER)
            h = next((x for x in nb.hypotheses if x["hyp_id"] == ev.get("hyp_id")), None)
            if h:
                card('<div class="bl-label">Step 3 · Link status</div>' + pattern_body(h), "acc")

# ---------------------------------------------------------------- Discoveries
with tab_disc:
    shown = [d for d in nb.discoveries if d["status"] != "rejected"]
    counts = {r: sum(d["rarity"] == r for d in shown) for r in ("legendary", "rare", "common")}
    st.markdown(f"## Findings · {len(shown)} collected")
    st.markdown("".join(pill(f"{r.title()} {n}", RARITY_PILL[r][0]) for r, n in counts.items()), unsafe_allow_html=True)
    cols = st.columns(3)
    from bodylab.engine import effect_text  # noqa: E402
    items = sorted(shown, key=lambda d: ({"legendary": 0, "rare": 1, "common": 2}[d["rarity"]], str(d["confirmed_at"])))
    for i, d in enumerate(items):
        pcls, ccls = RARITY_PILL[d["rarity"]]
        status = pill(patterns.STATUS[d["status"]], "p-warn" if d["status"] == "fading" else "p-ok")
        with cols[i % 3]:
            card(f'{pill(d["rarity"].title() + " · " + LABS[d["lab"]].name, pcls)}{status}'
                 f'<h3 style="margin:8px 0 4px;font-size:18px">{escape(d["title"])}</h3><div class="bl-sub">{escape(d["claim"])}</div>'
                 f'<div class="bl-big" style="margin-top:10px">{escape(effect_text(d) or "")}</div>'
                 f'<div class="bl-sub bl-num" style="margin-top:6px">{escape(d["evidence"])}</div>', ccls + (" fading" if d["status"] == "fading" else ""))
    close = [h for h in nb.hypotheses if h["status"] == "testing" and h["supports"] >= 2]
    for j, h in enumerate(close):
        with cols[(len(items) + j) % 3]:
            card(pattern_body(h, show_name=True), "locked")
    rejected = [d for d in nb.discoveries if d["status"] == "rejected"]
    if rejected:
        st.caption("Denied after newer data disagreed: " + ", ".join(d["title"] for d in rejected))

# ---------------------------------------------------------------- Notebook
with tab_nb:
    st.markdown("## Casebook")
    c1, c2 = st.columns(2, gap="large")
    order = {"testing": 0, "fading": 1, "confirmed": 2, "inconclusive": 3, "expired": 4, "rejected": 5}
    unconfirmed = sorted([h for h in nb.hypotheses if h["status"] != "confirmed"], key=lambda h: (order[h["status"]], h["hyp_id"]))
    confirmed = sorted([h for h in nb.hypotheses if h["status"] == "confirmed"], key=lambda h: h["hyp_id"])
    with c1:
        st.markdown(f"### Unconfirmed · {len(unconfirmed)}")
        active = [h for h in unconfirmed if h["status"] in ("testing", "fading")]
        closed = [h for h in unconfirmed if h["status"] not in ("testing", "fading")]
        for h in active:
            case_file(h)
        if not active:
            st.caption("No open investigations.")
        if closed:
            with st.expander(f"Closed patterns · {len(closed)}"):
                for h in closed:
                    case_file(h)
    with c2:
        st.markdown(f"### Confirmed · {len(confirmed)}")
        for h in confirmed:
            case_file(h)
        if not confirmed:
            st.caption("No confirmed patterns yet.")

    summary, watched = st.columns(2, gap="large")
    with summary:
        f = nb.funnel()
        top = max(f["surprises"], 1)
        rows = [("Surprising events", f["surprises"], "var(--ink3)"), ("Bad data, dismissed", f["bad_data"], "var(--warn)"),
                ("No clear reason", f["unexplained"], "var(--ink3)"), ("Became leads", f["leads"], "var(--acc)"),
                ("Confirmed discoveries", f["discoveries"], "var(--ok)")]
        html = '<div class="bl-label">Every surprise so far</div>'
        for name, n, color in rows:
            html += f'<div style="display:flex;justify-content:space-between;margin-top:8px"><span>{name}</span><span class="bl-num">{n}</span></div><div class="bar" style="height:8px"><i style="width:{n / top * 100:.0f}%;background:{color}"></i></div>'
        card(html + '<div class="bl-sub" style="margin-top:10px">Most surprises are noise or bad data. Only repeated patterns become discoveries.</div>')
    with watched:
        processed = len(nb.processed)
        good = sum(bool(p.get("good_data", True)) for p in nb.processed)
        card(f'<div class="bl-label">Situations watched</div><div class="bl-big bl-num">{processed}</div>'
             f'<div class="bl-sub">{good} passed the data check and counted as natural experiments.</div>')


# ---------------------------------------------------------------- Ask Sherlock Howls
with tab_chat:
    st.markdown("## Ask Sherlock Howls")
    st.caption("Ask about patterns, discoveries, hypotheses, and cases Sherlock Howls has actually observed in your data.")

    if not gemini_api_key():
        st.info("Set `GEMINI_API_KEY` to enable Ask Sherlock Howls.")
    else:
        chat_key = f"bodylab_chat_{pid}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        starter_questions = [
            "What have you learned about me so far?",
            "Which hypothesis has the strongest evidence?",
            "Have any of your ideas been proven wrong?",
            "What should Sherlock Howls investigate next?",
        ]
        if not st.session_state[chat_key]:
            st.markdown('<div class="bl-label">Try asking</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, starter in enumerate(starter_questions):
                if cols[i % 2].button(starter, key=f"starter_{pid}_{i}", use_container_width=True):
                    st.session_state[f"bodylab_pending_{pid}"] = starter
                    st.rerun()

        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        typed = st.chat_input("Ask about your Sherlock Howls data…", key=f"chat_input_{pid}")
        pending_key = f"bodylab_pending_{pid}"
        prompt = st.session_state.pop(pending_key, None) or typed

        if prompt:
            history = list(st.session_state[chat_key])
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Reading your lab notebook…"):
                        chatbot = BodyLabChat(nb, eng.features, pd.Timestamp(now))
                        answer = chatbot.ask(prompt, history=history)
                    st.markdown(answer)
                except Exception as exc:
                    answer = "I couldn't query the Sherlock Howls notebook right now. Please try again in a moment."
                    st.error(answer)
                    st.caption(str(exc))
            st.session_state[chat_key].append({"role": "assistant", "content": answer})

        if st.session_state[chat_key]:
            if st.button("Clear chat", key=f"clear_chat_{pid}"):
                st.session_state[chat_key] = []
                st.rerun()
