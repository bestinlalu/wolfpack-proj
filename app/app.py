"""Body Lab web app (Streamlit). Run locally with `streamlit run app/app.py`, or as a Databricks App."""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bodylab import meal_photo, voice  # noqa: E402
from bodylab.agent.investigator import make_investigator  # noqa: E402
from bodylab.agent.chat import BodyLabChat  # noqa: E402
from bodylab.formatting import fmt as _fmt  # noqa: E402
from bodylab.config import elevenlabs_api_key, gemini_api_key  # noqa: E402
from bodylab.engine import Engine  # noqa: E402
from bodylab.labs import LABS  # noqa: E402
from bodylab.pipeline.features import Signals  # noqa: E402
from bodylab.store import open_store  # noqa: E402
from bodylab.stress_scale import EXPLAINER, band  # noqa: E402

st.set_page_config(page_title="Body Lab", page_icon="🧪", layout="wide")

INK, INK3, GLU, ACC, OK = "#13262b", "#82918e", "#d4532a", "#2b55c9", "#2c8556"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --ink:#13262b; --ink2:#4b5d61; --ink3:#82918e; --line:rgba(19,38,43,.12); --card:#fff; --sunk:#f4f7f6;
  --acc:#2b55c9; --accs:#e5ebfb; --glu:#d4532a; --glus:#fbe6dd; --ok:#2c8556; --oks:#e0f1e7; --warn:#a86c12; --warns:#f8ecd6; --gold:#9a6d05; --golds:#f7edcf; }
html, body, [class*="css"] { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
h1, h2, h3 { font-family: 'Bricolage Grotesque', system-ui, sans-serif !important; letter-spacing: -0.015em; }
.bl-card { background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; margin-bottom: 12px; color: var(--ink); }
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
.check::before { content: "✓"; color: var(--ok); font-weight: 700; margin-right: 8px; } .fail::before { content: "✕"; color: var(--glu); font-weight: 700; margin-right: 8px; }
</style>
""", unsafe_allow_html=True)

LAB_PILL = {"fuel": "p-glu", "stress": "p-warn", "sleep": "p-acc", "movement": "p-ok"}
RARITY_PILL = {"legendary": ("p-gold", "gold"), "rare": ("p-acc", "acc"), "common": ("p-plain", "")}


def card(body: str, cls: str = "") -> None:
    st.markdown(f'<div class="bl-card {cls}">{body}</div>', unsafe_allow_html=True)


def pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{escape(text)}</span>'


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
store = open_store()
pids = store.pids()
if not pids:
    st.title("Body Lab")
    st.info("No participants yet. Prepare data first:\n\n`python scripts/prepare.py --synthetic` (demo data) or "
            "`python scripts/prepare.py --pid 001` after downloading the BIG IDEAs files.")
    st.stop()

with st.sidebar:
    st.markdown("### 🧪 Body Lab")
    pid = st.selectbox("Participant", pids, index=0)
    use_llm = st.toggle("Gemini agent", value=bool(gemini_api_key()), disabled=not gemini_api_key() or store.read_only,
                        help="Without GEMINI_API_KEY the rule-based investigator runs the same tools.")


def load_engine(pid: str) -> Engine:
    minute, meals = store.read_inputs(pid)
    features, nb, until, _ = store.read_state(pid)
    eng = Engine(pid, minute, meals, nb, make_investigator(prefer_llm=use_llm))
    eng.features, eng.until = features, until
    return eng


eng = load_engine(pid)
nb = eng.notebook
now = eng.until or eng.start

with st.sidebar:
    st.markdown("#### Replay")
    pct = 0.0 if eng.until is None else (eng.until - eng.start) / (eng.end - eng.start)
    st.progress(min(max(pct, 0.0), 1.0), text=f"{now:%a %b %d, %H:%M}" if eng.until is not None else "Not started")
    if store.read_only:
        st.caption("Streaming on Databricks; this view refreshes from the Delta tables.")
        if st.button("Refresh", use_container_width=True):
            st.rerun()
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
    st.caption("Body Lab reports what was different, never causes. Not medical advice.")

if eng.until is None:
    st.title("Body Lab")
    st.markdown(f"Participant **{escape(pid)}** has data from **{eng.start:%b %d}** to **{eng.end:%b %d}**. "
                "Use the replay controls on the left to stream it through the agent.")
    st.stop()


@st.cache_data(show_spinner=False)
def signals_frame(pid: str, until: str) -> pd.DataFrame:
    s = Signals(eng.minute, pd.Timestamp(until))
    return s.df.assign(stress=s.stress)


sig_df = signals_frame(pid, str(eng.until))

tab_today, tab_case, tab_disc, tab_nb, tab_chat = st.tabs(["Today", "Case", "Discoveries", "Notebook", "💬 Ask Body Lab"])

# ---------------------------------------------------------------- Today
with tab_today:
    rank, pts, nxt = nb.rank()
    st.markdown(f"## {now:%A, %b %d}")
    left, right = st.columns([3, 2], gap="large")
    with left:
        feed = sorted(nb.messages, key=lambda m: pd.Timestamp(m["ts"]), reverse=True)
        if not feed:
            card('<div class="bl-label">Nothing to report yet</div><div class="bl-sub">The agent only writes when it has an answer: a solved case or a confirmed discovery.</div>')
        for m in feed[:6]:
            kind = m["kind"]
            cls, label = {"discovery": ("ok", ("Discovery confirmed", "p-ok")), "case": ("glu", ("Case solved", "p-glu")),
                          "rejected": ("", ("Not confirmed", "p-plain")), "fading": ("", ("Fading", "p-warn"))}[kind]
            card(f'{pill(label[0], label[1])}{pill(LABS[m["lab"]].name, LAB_PILL[m["lab"]])}'
                 f'<span class="bl-sub bl-num" style="float:right">{pd.Timestamp(m["ts"]):%a %H:%M}</span>'
                 f'<div class="bl-title">{escape(m["title"])}</div><div class="bl-sub">{escape(m["body"])}</div>', cls)
        week = [e for e in nb.events if pd.Timestamp(e["ts"]) > now - pd.Timedelta(days=7)]
        unexplained = sum(e["verdict"] == "unexplained" for e in week)
        bad = sum(e["verdict"] == "bad_data" for e in week)
        card(f'<div class="bl-label">Closed quietly this week</div>'
             f'<div>{unexplained} surprise{"s" if unexplained != 1 else ""} with no clear reason · {bad} dismissed as bad data</div>'
             f'<div class="bl-sub">No alerts were sent for these. Details are in the Notebook.</div>')
    with right:
        to_next = f"{nxt - pts} points to the next rank" if nxt else "Top rank reached"
        width = 100 if not nxt else int(pts / nxt * 100)
        card(f'<div class="bl-label">Scientist rank</div><div class="bl-big">{escape(rank)}</div>'
             f'<div class="bl-sub bl-num">{pts} points · {to_next}</div><div class="bar"><i style="width:{width}%;background:var(--acc)"></i></div>')
        for q in nb.quests[-2:]:
            dots = "".join(f'<i class="{"s" if i < q["progress"] else ""}"></i>' for i in range(int(q["target"])))
            progress_text = "Done" if q["done"] else f'{q["progress"]} of {q["target"]}'
            card(f'<div class="bl-label">Quest · optional, detected automatically</div><div class="bl-title">{escape(q["title"])}</div>'
                 f'<div class="dots">{dots}</div><div class="bl-sub">{progress_text}</div>')
        st.markdown('<div class="bl-label">Your labs</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for i, lab in enumerate(LABS.values()):
            cards = sum(d["lab"] == lab.key and d["status"] != "rejected" for d in nb.discoveries)
            open_h = sum(h["lab"] == lab.key and h["status"] == "testing" for h in nb.hypotheses)
            with cols[i % 2]:
                extra = (f'<div class="bl-sub" title="{escape(EXPLAINER)}" style="cursor:help">stress 1–10, personal ⓘ</div>'
                         if lab.key == "stress" else "")
                card(f'<div class="bl-title">{lab.name}</div><div class="bl-sub">{lab.situation}s</div>{extra}{pill(f"{cards} cards", "p-acc")}{pill(f"{open_h} open", "p-plain")}')
        if st.button("▶ Weekly recap", use_container_width=True):
            text = voice.recap_text(nb, now)
            text = voice.polish(text)
            st.write(text)
            try:
                audio = voice.speak(text)
                if audio:
                    st.audio(audio, format="audio/mpeg")
                else:
                    st.caption("Set ELEVENLABS_API_KEY to hear this read aloud.")
            except Exception as exc:
                st.caption(f"Voice unavailable: {exc}")

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
        st.info("No surprises investigated yet.")
    else:
        verdict_label = {"lead": "Solved", "unexplained": "Unexplained", "bad_data": "Bad data"}
        options = {e["event_id"]: f'{verdict_label.get(e["verdict"], e["verdict"])} · {LABS[e["lab"]].name} · {pd.Timestamp(e["ts"]):%a %b %d %H:%M} · {e["title"][:70]}' for e in cases}
        keys = list(options)
        leads = [e["event_id"] for e in cases if e["verdict"] == "lead"]
        fuel_leads = [e["event_id"] for e in cases if e["verdict"] == "lead" and e["lab"] == "fuel"]
        default = (fuel_leads or leads or keys)[0]
        pick = st.selectbox("Case", keys, index=keys.index(default), format_func=options.get)
        ev = nb.event(pick)
        lab = ev["lab"]
        L = LABS[lab]
        st.markdown(f'{pill(L.name + " Lab", LAB_PILL[lab])}{pill(verdict_label.get(ev["verdict"], ev["verdict"]), {"lead": "p-glu", "unexplained": "p-plain", "bad_data": "p-warn"}[ev["verdict"]])}'
                    f'<span class="bl-sub bl-num">{escape(ev["event_id"])} · agent: {escape(str(ev.get("agent", "")))}</span>', unsafe_allow_html=True)
        st.markdown(f"### {escape(ev['title'] or 'Case')}")
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
                card(f'<div class="bl-label">Step 3 · Verdict</div><div class="bl-title">{escape(h["hyp_id"])}: {escape(h["claim"])}</div>'
                     f'<div class="bl-sub">Status: {escape(h["status"])} · {h["supports"]} supporting, {h["contradicts"]} against</div>', "acc")

# ---------------------------------------------------------------- Discoveries
with tab_disc:
    shown = [d for d in nb.discoveries if d["status"] != "rejected"]
    counts = {r: sum(d["rarity"] == r for d in shown) for r in ("legendary", "rare", "common")}
    st.markdown(f"## Discoveries · {len(shown)} collected")
    st.markdown("".join(pill(f"{r.title()} {n}", RARITY_PILL[r][0]) for r, n in counts.items()), unsafe_allow_html=True)
    cols = st.columns(3)
    from bodylab.engine import effect_text  # noqa: E402
    items = sorted(shown, key=lambda d: ({"legendary": 0, "rare": 1, "common": 2}[d["rarity"]], str(d["confirmed_at"])))
    for i, d in enumerate(items):
        pcls, ccls = RARITY_PILL[d["rarity"]]
        status = pill("Fading", "p-warn") if d["status"] == "fading" else ""
        with cols[i % 3]:
            card(f'{pill(d["rarity"].title() + " · " + LABS[d["lab"]].name, pcls)}{status}'
                 f'<h3 style="margin:8px 0 4px;font-size:18px">{escape(d["title"])}</h3><div class="bl-sub">{escape(d["claim"])}</div>'
                 f'<div class="bl-big" style="margin-top:10px">{escape(effect_text(d) or "")}</div>'
                 f'<div class="bl-sub bl-num" style="margin-top:6px">{escape(d["evidence"])}</div>', ccls + (" fading" if d["status"] == "fading" else ""))
    close = [h for h in nb.hypotheses if h["status"] == "testing" and h["supports"] >= 2]
    for j, h in enumerate(close):
        with cols[(len(items) + j) % 3]:
            card(f'<div class="bl-title" style="font-size:22px">?</div><div>{LABS[h["lab"]].name} Lab</div>'
                 f'<div class="bl-sub">{escape(h["hyp_id"])} is {max(3 - h["supports"], 1)} test away</div>', "locked")
    rejected = [d for d in nb.discoveries if d["status"] == "rejected"]
    if rejected:
        st.caption("Withdrawn after newer data disagreed: " + ", ".join(d["title"] for d in rejected))

# ---------------------------------------------------------------- Notebook
with tab_nb:
    st.markdown("## The agent's lab notebook")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        order = {"testing": 0, "fading": 1, "confirmed": 2, "inconclusive": 3, "expired": 4, "rejected": 5}
        status_pill = {"testing": "p-plain", "confirmed": "p-ok", "fading": "p-warn", "rejected": "p-glu", "inconclusive": "p-plain", "expired": "p-plain"}
        testing = [h for h in nb.hypotheses if h["status"] == "testing"]
        meal_open = sum(h["meal_related"] for h in testing)
        st.markdown(f'<div class="bl-label">Hypotheses · {len(testing)} of 10 open slots · meal-related {meal_open} of 2 max</div>', unsafe_allow_html=True)
        for h in sorted(nb.hypotheses, key=lambda h: (order[h["status"]], h["hyp_id"])):
            ev_rows = [e for e in nb.evidence if e["hyp_id"] == h["hyp_id"] and e["verdict"] != "neutral"]
            dots = "".join(f'<i class="{"s" if e["verdict"] == "supports" else "c"}"></i>' for e in ev_rows[-10:])
            dots += "".join("<i></i>" for _ in range(max(0, 3 - h["supports"]))) if h["status"] == "testing" else ""
            card(f'{pill(h["hyp_id"], "p-acc")}{pill(LABS[h["lab"]].name, LAB_PILL[h["lab"]])}{pill(h["status"], status_pill[h["status"]])}'
                 f'<div class="bl-title">{escape(h["claim"])}</div><div class="dots">{dots}</div>'
                 f'<div class="bl-sub">{h["supports"]} supporting · {h["contradicts"]} against · {h["chances"]} chances</div>')
    with c2:
        f = nb.funnel()
        top = max(f["surprises"], 1)
        rows = [("Surprising events", f["surprises"], "var(--ink3)"), ("Bad data, dismissed", f["bad_data"], "var(--warn)"),
                ("No clear reason", f["unexplained"], "var(--ink3)"), ("Became leads", f["leads"], "var(--acc)"),
                ("Confirmed discoveries", f["discoveries"], "var(--ok)")]
        html = '<div class="bl-label">Every surprise so far</div>'
        for name, n, color in rows:
            html += f'<div style="display:flex;justify-content:space-between;margin-top:8px"><span>{name}</span><span class="bl-num">{n}</span></div><div class="bar" style="height:8px"><i style="width:{n / top * 100:.0f}%;background:{color}"></i></div>'
        card(html + '<div class="bl-sub" style="margin-top:10px">Most surprises are noise or bad data. Only repeated patterns become discoveries.</div>')
        processed = len(nb.processed)
        good = sum(bool(p.get("good_data", True)) for p in nb.processed)
        card(f'<div class="bl-label">Situations watched</div><div class="bl-big bl-num">{processed}</div>'
             f'<div class="bl-sub">{good} passed the data check and counted as natural experiments.</div>')


# ---------------------------------------------------------------- Ask Body Lab
with tab_chat:
    st.markdown("## Ask Body Lab")
    st.caption("Ask about patterns, discoveries, hypotheses, and cases Body Lab has actually observed in your data.")

    if not gemini_api_key():
        st.info("Set `GEMINI_API_KEY` to enable Ask Body Lab.")
    else:
        chat_key = f"bodylab_chat_{pid}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        starter_questions = [
            "What have you learned about me so far?",
            "Which hypothesis has the strongest evidence?",
            "Have any of your ideas been proven wrong?",
            "What should Body Lab investigate next?",
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

        typed = st.chat_input("Ask about your Body Lab data…", key=f"chat_input_{pid}")
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
                    answer = "I couldn't query the Body Lab notebook right now. Please try again in a moment."
                    st.error(answer)
                    st.caption(str(exc))
            st.session_state[chat_key].append({"role": "assistant", "content": answer})

        if st.session_state[chat_key]:
            if st.button("Clear chat", key=f"clear_chat_{pid}"):
                st.session_state[chat_key] = []
                st.rerun()
