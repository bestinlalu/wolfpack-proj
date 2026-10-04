"""Mindfulness tab: timed guided meditations and guided visualizations.

Gemini writes the words (meditation cues personalised from the casebook, visualizations from a theme); ElevenLabs
speaks them through voice.speak. Without a Gemini key the timed meditations fall back to built-in cues;
visualizations need Gemini.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass

from bodylab.config import SETTINGS, gemini_api_key

log = logging.getLogger(__name__)

CUES = ("start", "checkin", "finish")  # opening guidance, a gentle prompt at each new round, closing


@dataclass(frozen=True)
class Practice:
    key: str
    name: str
    round_label: str  # what each round is called on the timer
    round_min: int
    rounds: int
    brief: str  # what the cues are for, for Gemini
    fallback: dict


PRACTICES = {
    "breathing": Practice("breathing", "Breathing", "Breathe", 3, 3,
                          "a breathing meditation: slow, even breaths, attention resting on the breath",
                          {"start": "Sit comfortably and close your eyes. Breathe in for four, and slowly out for six.",
                           "checkin": "If your mind has wandered, that's fine. Gently come back to the breath.",
                           "finish": "Let your breathing return to normal. When you're ready, open your eyes."}),
    "body_scan": Practice("body_scan", "Body scan", "Scan", 3, 4,
                          "a body scan meditation: attention moving slowly through the body, noticing and softening",
                          {"start": "Settle in and bring your attention to your feet. Notice any warmth, weight or tension.",
                           "checkin": "Let your attention move a little higher through the body. Notice, and soften.",
                           "finish": "Feel your whole body at once, resting here. Slowly come back to the room."}),
    "pause": Practice("pause", "Mindful pause", "Pause", 5, 1,
                      "a short mindful pause in the middle of the day: arriving in the present moment",
                      {"start": "Pause what you're doing. Feel your feet on the floor and take three slow breaths.",
                       "checkin": "",
                       "finish": "Carry this calm with you as you go back to your day."}),
    "sleep": Practice("sleep", "Sleep wind-down", "Wind down", 15, 1,
                      "a meditation before sleep: slow breathing, releasing the body, letting the day go",
                      {"start": "Lie back and let the bed hold you. Breathe slowly, and let each breath out be longer.",
                       "checkin": "",
                       "finish": "Let go of the day. There's nothing left to do. Goodnight."}),
}


def cue_names(practice: Practice, rounds: int | None = None) -> list[str]:
    multi = (rounds or practice.rounds) > 1
    return [c for c in CUES if c != "checkin" or (multi and practice.fallback["checkin"])]


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


def parse_cues(raw: str, practice: Practice) -> dict:
    """Cue texts from Gemini's JSON; anything missing or empty falls back to the built-in line."""
    try:
        got = json.loads(re.sub(r"^```(?:json)?|```$", "", raw.strip()).strip())
    except (ValueError, TypeError):
        got = {}
    got = got if isinstance(got, dict) else {}
    return {c: (str(got.get(c) or "").strip() or practice.fallback[c]) if practice.fallback[c] else "" for c in CUES}


def write_cues(practice: Practice, name: str, context: str, round_min: int, rounds: int) -> dict:
    """Short spoken cues for one meditation, personalised by Gemini when a key is set."""
    if not gemini_api_key():
        return dict(practice.fallback)
    wanted = cue_names(practice, rounds)
    prompt = (f"You are Sherlock Howls, a calm meditation guide. Write spoken cues for {practice.brief}. "
              f"It runs {rounds} round{'s' if rounds > 1 else ''} of {round_min} minutes. The listener is {name}. "
              f"What their wearable lab notebook found recently (you may make at most one gentle, accurate reference; "
              f"never invent facts; no medical advice):\n{context}\n\n"
              f"Return JSON with exactly these keys: {', '.join(wanted)}. 'start' opens the meditation, "
              + ("'checkin' is said quietly at the start of each later round, " if "checkin" in wanted else "")
              + "'finish' closes it. Each value is 1-2 slow, short spoken sentences (under 30 words), plain text, no emojis.")
    try:
        return parse_cues(_gemini(prompt, timeout_s=25, json_out=True), practice)
    except Exception as exc:
        log.warning("Cue writing failed, using built-in cues: %s", exc)
        return dict(practice.fallback)


VISUALIZATION_LENGTHS = {"Short (~2 min)": 250, "Longer (~4 min)": 450}


def write_visualization(name: str, theme: str, words: int) -> tuple[str, str]:
    """(title, script) for a guided visualization meditation. Needs GEMINI_API_KEY."""
    theme = theme.strip() or "a calm place in nature, chosen by you"
    prompt = (f"Write a guided visualization meditation for an adult named {name}, about {words} words, to be read "
              f"aloud slowly. Setting: {theme}. Speak to the listener in the second person, guide their breathing a "
              "few times, use soft sensory detail, nothing scary or exciting, and end by gently bringing them back "
              "to the present. Plain prose for speech: no headings, lists, emojis or stage directions. "
              "First line: the title only. Then a blank line, then the meditation.")
    text = _gemini(prompt, timeout_s=60)
    title, _, body = text.partition("\n")
    title = title.strip().strip("#*\"' ")
    return (title or "Guided visualization"), (body.strip() or text)


def timer_html(practice: Practice, round_min: int, rounds: int, audio: dict, timer_id: str) -> str:
    """A self-contained timer that plays each cue as its round starts. It runs in the browser, so the page's
    background syncs don't interrupt it, and it resumes from its start time if the page reloads."""
    phases = [{"label": f"{practice.round_label} {r + 1}/{rounds}" if rounds > 1 else practice.round_label,
               "sec": round_min * 60, "cue": "start" if r == 0 else "checkin"} for r in range(rounds)]
    sounds = {c: "data:audio/mpeg;base64," + base64.b64encode(b).decode() for c, b in audio.items() if b}
    return _TIMER.replace("__PHASES__", json.dumps(phases)).replace("__SOUNDS__", json.dumps(sounds)) \
                 .replace("__ID__", json.dumps(timer_id))


_TIMER = """
<div id="t" style="font-family:system-ui,sans-serif;text-align:center;padding:12px;border:1px solid #8884;border-radius:14px">
  <div id="label" style="font-size:15px;opacity:.75">Ready</div>
  <div id="clock" style="font-size:52px;font-weight:700;font-variant-numeric:tabular-nums;margin:2px 0 8px">--:--</div>
  <div style="height:6px;background:#8883;border-radius:3px;overflow:hidden;margin:0 12px 12px">
    <div id="bar" style="height:100%;width:0;background:#6c63ff;transition:width .5s"></div></div>
  <button id="go" style="padding:8px 20px;border-radius:10px;border:0;background:#6c63ff;color:#fff;font-size:15px;cursor:pointer">Begin</button>
  <button id="stop" style="padding:8px 16px;border-radius:10px;border:1px solid #8886;background:none;color:inherit;font-size:15px;cursor:pointer;display:none">End</button>
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
                    $("label").textContent = "Complete"; $("clock").textContent = "00:00"; stop(false); return; }
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
