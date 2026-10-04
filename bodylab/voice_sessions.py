"""Voices tab: guided Pomodoro-style sessions (focus, sleep, drive) and bedtime stories.

Gemini writes the words (cues personalised from the casebook, stories from a theme); ElevenLabs speaks them
through voice.speak. Without a Gemini key the sessions fall back to built-in cues; stories need Gemini.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass

from bodylab.config import SETTINGS, gemini_api_key

log = logging.getLogger(__name__)

CUES = ("start", "break", "resume", "finish")


@dataclass(frozen=True)
class Mode:
    key: str
    name: str
    work_label: str  # what the long phase is called on the timer
    rest_label: str
    work_min: int
    rest_min: int
    rounds: int
    brief: str  # what the cues are for, for Gemini
    fallback: dict


MODES = {
    "focus": Mode("focus", "Focus", "Focus", "Break", 25, 5, 4,
                  "a Pomodoro focus session: focused work blocks with short breaks",
                  {"start": "Let's focus. Phone face down, one task only. I'll tell you when it's time for a break.",
                   "break": "Nice work. Take five: stand up, stretch, look at something far away.",
                   "resume": "Break's over. Back to that one task.",
                   "finish": "Session done. Good focus today. Take a proper rest."}),
    "sleep": Mode("sleep", "Sleep", "Wind down", "Rest", 20, 0, 1,
                  "a wind-down before sleep: slow breathing, relaxing the body, letting the day go",
                  {"start": "Time to wind down. Dim the lights, breathe in slowly for four, and out for six.",
                   "break": "", "resume": "",
                   "finish": "That's your wind-down. Let your breathing stay slow. Goodnight."}),
    "drive": Mode("drive", "Drive", "Drive", "Rest stop", 50, 10, 3,
                  "a drive with regular rest stops; the listener is driving, so every cue must be short, "
                  "calm and never ask them to look at or touch the phone",
                  {"start": "Drive safe. Eyes on the road. I'll remind you when it's time for a rest stop.",
                   "break": "You've been driving a while. Pull over somewhere safe for a ten-minute break when you can.",
                   "resume": "Feeling fresh? Back on the road when you're ready. Drive safe.",
                   "finish": "That's the drive. Well done, and take a real rest now."}),
}


def cue_names(mode: Mode) -> list[str]:
    return [c for c in CUES if mode.fallback[c]]


def _gemini(prompt: str, timeout_s: int, json_out: bool = False) -> str:
    """One Gemini call, trying the fallback models in turn; raises if none answers."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key(), http_options=types.HttpOptions(timeout=timeout_s * 1000))
    cfg = types.GenerateContentConfig(automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                                      response_mime_type="application/json" if json_out else None)
    last = None
    for model in dict.fromkeys((SETTINGS.gemini_model, *SETTINGS.gemini_fallback_models)):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            if resp.text:
                return resp.text.strip()
        except Exception as exc:
            last = exc
            log.warning("Gemini %s failed: %s", model, exc)
    raise RuntimeError(f"Gemini didn't answer: {last}")


def parse_cues(raw: str, mode: Mode) -> dict:
    """Cue texts from Gemini's JSON; anything missing or empty falls back to the built-in line."""
    try:
        got = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip())
    except (ValueError, TypeError):
        got = {}
    got = got if isinstance(got, dict) else {}
    return {c: (str(got.get(c) or "").strip() or mode.fallback[c]) if mode.fallback[c] else "" for c in CUES}


def write_cues(mode: Mode, name: str, context: str, work_min: int, rest_min: int) -> dict:
    """Short spoken cues for one session, personalised by Gemini when a key is set."""
    if not gemini_api_key():
        return dict(mode.fallback)
    wanted = cue_names(mode)
    prompt = (f"You are Sherlock Howls, a warm personal body-science companion. Write spoken cues for {mode.brief}. "
              f"Blocks are {work_min} minutes" + (f" with {rest_min}-minute breaks" if rest_min else "") + ". "
              f"The listener is {name}. What their wearable lab notebook found recently (use at most one light, "
              f"accurate reference; never invent facts; no medical advice):\n{context}\n\n"
              f"Return JSON with exactly these keys: {', '.join(wanted)}. Each value is 1-2 short spoken sentences "
              "(under 30 words), plain text, no emojis.")
    try:
        return parse_cues(_gemini(prompt, timeout_s=25, json_out=True), mode)
    except Exception as exc:
        log.warning("Cue writing failed, using built-in cues: %s", exc)
        return dict(mode.fallback)


STORY_LENGTHS = {"Short (~2 min)": 250, "Longer (~4 min)": 450}


def write_story(name: str, theme: str, words: int) -> tuple[str, str]:
    """(title, story) for a calm bedtime story. Needs GEMINI_API_KEY."""
    theme = theme.strip() or "a gentle surprise, chosen by you"
    prompt = (f"Write a calm, cosy bedtime story for an adult named {name}, about {words} words, to be read aloud. "
              f"Theme: {theme}. Slow pace, soft sensory detail, no cliffhangers, nothing scary, and let it drift "
              "towards rest at the end. Plain prose for speech: no headings, lists, emojis or stage directions. "
              "First line: the title only. Then a blank line, then the story.")
    text = _gemini(prompt, timeout_s=60)
    title, _, body = text.partition("\n")
    title = title.strip().strip("#*\"' ")
    return (title or "Tonight's story"), (body.strip() or text)


def timer_html(mode: Mode, work_min: int, rest_min: int, rounds: int, audio: dict, timer_id: str) -> str:
    """A self-contained timer that plays each cue as its phase starts. It runs in the browser, so the page's
    background syncs don't interrupt it, and it resumes from its start time if the page reloads."""
    phases = []
    for r in range(rounds):
        phases.append({"label": f"{mode.work_label} {r + 1}/{rounds}" if rounds > 1 else mode.work_label,
                       "sec": work_min * 60, "cue": "start" if r == 0 else "resume", "work": True})
        if rest_min and r < rounds - 1:
            phases.append({"label": mode.rest_label, "sec": rest_min * 60, "cue": "break", "work": False})
    sounds = {c: "data:audio/mpeg;base64," + base64.b64encode(b).decode() for c, b in audio.items() if b}
    return _TIMER.replace("__PHASES__", json.dumps(phases)).replace("__SOUNDS__", json.dumps(sounds)) \
                 .replace("__ID__", json.dumps(timer_id))


_TIMER = """
<div id="t" style="font-family:system-ui,sans-serif;text-align:center;padding:12px;border:1px solid #8884;border-radius:14px">
  <div id="label" style="font-size:15px;opacity:.75">Ready</div>
  <div id="clock" style="font-size:52px;font-weight:700;font-variant-numeric:tabular-nums;margin:2px 0 8px">--:--</div>
  <div style="height:6px;background:#8883;border-radius:3px;overflow:hidden;margin:0 12px 12px">
    <div id="bar" style="height:100%;width:0;background:#6c63ff;transition:width .5s"></div></div>
  <button id="go" style="padding:8px 20px;border-radius:10px;border:0;background:#6c63ff;color:#fff;font-size:15px;cursor:pointer">▶ Start</button>
  <button id="stop" style="padding:8px 16px;border-radius:10px;border:1px solid #8886;background:none;color:inherit;font-size:15px;cursor:pointer;display:none">■ Stop</button>
</div>
<script>
const P = __PHASES__, S = __SOUNDS__, KEY = "bodylab-timer-" + __ID__;
const total = P.reduce((a, p) => a + p.sec, 0);
const $ = id => document.getElementById(id);
let start = null, played = -1, tick = null;
try { start = Number(localStorage.getItem(KEY)) || null; } catch (e) {}
function save(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (e) {} }
function play(cue) { if (S[cue]) { new Audio(S[cue]).play().catch(() => {}); } }
function fmt(s) { s = Math.max(0, Math.ceil(s)); return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0"); }
function render() {
  if (!start) { $("label").textContent = "Ready"; $("clock").textContent = fmt(P[0].sec); $("bar").style.width = "0";
                $("go").style.display = ""; $("stop").style.display = "none"; return; }
  let t = (Date.now() - start) / 1000;
  $("go").style.display = "none"; $("stop").style.display = "";
  $("bar").style.width = Math.min(100, 100 * t / total) + "%";
  if (t >= total) { if (played < P.length) { play("finish"); played = P.length; }
                    $("label").textContent = "Done"; $("clock").textContent = "00:00"; stop(false); return; }
  let i = 0; while (t >= P[i].sec) { t -= P[i].sec; i++; }
  if (i > played) { if (played >= 0 || t < 3) play(P[i].cue); played = i; }  // after a reload, don't replay an old cue
  $("label").textContent = P[i].label; $("clock").textContent = fmt(P[i].sec - t);
}
function stop(reset) { clearInterval(tick); tick = null; save(null); if (reset) { start = null; played = -1; render(); }
                       else { start = null; $("go").style.display = ""; $("stop").style.display = "none"; } }
$("go").onclick = () => { start = Date.now(); played = -1; save(start); render(); tick = setInterval(render, 500); };
$("stop").onclick = () => stop(true);
render(); if (start) tick = setInterval(render, 500);
</script>
"""
