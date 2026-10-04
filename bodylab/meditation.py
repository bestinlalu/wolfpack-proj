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

import requests

from bodylab.config import SETTINGS, elevenlabs_api_key, gemini_api_key

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
    music: str  # prompt for the ElevenLabs background track
    fallback: dict


PRACTICES = {
    "breathing": Practice("breathing", "Breathing", "Breathe", 3, 3,
                          "a breathing meditation: slow, even breaths, attention resting on the breath",
                      "Slow ambient meditation music, warm soft pads, gentle and spacious, about 60 bpm, no percussion, no vocals",
                          {"start": "Sit comfortably and close your eyes. Breathe in for four, and slowly out for six.",
                           "checkin": "If your mind has wandered, that's fine. Gently come back to the breath.",
                           "finish": "Let your breathing return to normal. When you're ready, open your eyes."}),
    "body_scan": Practice("body_scan", "Body scan", "Scan", 3, 4,
                          "a body scan meditation: attention moving slowly through the body, noticing and softening",
                      "Calm ambient drone with soft piano notes and distant singing bowls, very slow, no percussion, no vocals",
                          {"start": "Settle in and bring your attention to your feet. Notice any warmth, weight or tension.",
                           "checkin": "Let your attention move a little higher through the body. Notice, and soften.",
                           "finish": "Feel your whole body at once, resting here. Slowly come back to the room."}),
    "pause": Practice("pause", "Mindful pause", "Pause", 5, 1,
                      "a short mindful pause in the middle of the day: arriving in the present moment",
                      "Light peaceful ambient music with soft flute and gentle nature texture, airy and calm, no vocals",
                      {"start": "Pause what you're doing. Feel your feet on the floor and take three slow breaths.",
                       "checkin": "",
                       "finish": "Carry this calm with you as you go back to your day."}),
    "sleep": Practice("sleep", "Sleep wind-down", "Wind down", 15, 1,
                      "a meditation before sleep: slow breathing, releasing the body, letting the day go",
                      "Very slow, dreamy sleep music, deep soft pads and faint low piano, quiet and dark, no percussion, no vocals",
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


MUSIC_SECONDS = 60  # one track per practice, looped under the cues (kept short to save ElevenLabs credits)


def compose_music(prompt: str, seconds: int = MUSIC_SECONDS) -> bytes | None:
    """An instrumental MP3 from ElevenLabs Music, or None without a key. Raises on API errors (e.g. plan limits)."""
    key = elevenlabs_api_key()
    if not key:
        return None
    resp = requests.post("https://api.elevenlabs.io/v1/music", params={"output_format": "mp3_44100_64"},
                         headers={"xi-api-key": key, "Accept": "audio/mpeg"},
                         json={"prompt": prompt, "music_length_ms": seconds * 1000, "force_instrumental": True},
                         timeout=240)
    if not resp.ok:
        detail = resp.text[:200]
        raise RuntimeError(f"ElevenLabs music {resp.status_code}: {detail}")
    return resp.content


def _sessions_path(pid: str):
    from pathlib import Path

    if not pid.isalnum():
        raise ValueError("Invalid participant ID")
    return Path(SETTINGS.lakehouse_dir) / "mindfulness" / f"{pid}.json"


def load_sessions(pid: str) -> list[dict]:
    """Meditations this person marked as finished: [{"practice": ..., "at": iso time}]."""
    path = _sessions_path(pid)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def record_session(pid: str, practice: str) -> list[dict]:
    import pandas as pd

    sessions = load_sessions(pid) + [{"practice": practice, "at": pd.Timestamp.now().isoformat()}]
    path = _sessions_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    return sessions


MINDFUL_QUEST_TARGET = 2


def mindful_quest(sessions: list[dict]) -> dict:
    """The Stress lab's mindfulness quest, in the same shape as the agent's quests."""
    done = min(len(sessions), MINDFUL_QUEST_TARGET)
    return {"quest_id": "QM", "lab": "stress", "title": "Calm a stressful day: finish two guided meditations in Mindfulness",
            "progress": done, "target": MINDFUL_QUEST_TARGET, "done": done >= MINDFUL_QUEST_TARGET}


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


def timer_html(practice: Practice, round_min: int, rounds: int, audio: dict, timer_id: str,
               music: bytes | None = None) -> str:
    """A self-contained timer that plays each cue as its round starts. It runs in the browser, so the page's
    background syncs don't interrupt it, and it resumes from its start time if the page reloads."""
    phases = [{"label": f"{practice.round_label} {r + 1}/{rounds}" if rounds > 1 else practice.round_label,
               "sec": round_min * 60, "cue": "start" if r == 0 else "checkin"} for r in range(rounds)]
    sounds = {c: "data:audio/mpeg;base64," + base64.b64encode(b).decode() for c, b in audio.items() if b}
    track = "data:audio/mpeg;base64," + base64.b64encode(music).decode() if music else ""
    return _TIMER.replace("__PHASES__", json.dumps(phases)).replace("__SOUNDS__", json.dumps(sounds)) \
                 .replace("__ID__", json.dumps(timer_id)).replace("__MUSIC__", json.dumps(track))


_TIMER = """
<div id="t" style="font-family:system-ui,sans-serif;text-align:center;padding:12px;border:1px solid #8884;border-radius:14px">
  <div id="label" style="font-size:15px;opacity:.75">Ready</div>
  <div id="clock" style="font-size:52px;font-weight:700;font-variant-numeric:tabular-nums;margin:2px 0 8px">--:--</div>
  <div style="height:6px;background:#8883;border-radius:3px;overflow:hidden;margin:0 12px 12px">
    <div id="bar" style="height:100%;width:0;background:#6c63ff;transition:width .5s"></div></div>
  <button id="go" style="padding:8px 20px;border-radius:10px;border:0;background:#6c63ff;color:#fff;font-size:15px;cursor:pointer">Begin</button>
  <button id="stop" style="padding:8px 16px;border-radius:10px;border:1px solid #8886;background:none;color:inherit;font-size:15px;cursor:pointer;display:none">End</button>
  <div id="mus" style="display:none;margin-top:10px;font-size:13px;opacity:.8">
    <button id="mtoggle" style="padding:3px 10px;border-radius:8px;border:1px solid #8886;background:none;color:inherit;font-size:13px;cursor:pointer">Music on</button>
    <label style="margin-left:8px">Volume <input id="mvol" type="range" min="0" max="100" value="35" style="vertical-align:middle;width:110px"></label>
  </div>
</div>
<script>
const P = __PHASES__, S = __SOUNDS__, M = __MUSIC__, KEY = "bodylab-timer-" + __ID__;
const music = M ? new Audio(M) : null;
let musicOn = true;
if (music) { music.loop = true; $mus(); }
function $mus() { document.getElementById("mus").style.display = ""; }
function vol() { return document.getElementById("mvol").value / 100; }
function musicPlay() { if (music && musicOn) { music.volume = vol(); music.play().catch(() => {}); } }
function musicStop() { if (music) { music.pause(); music.currentTime = 0; } }
const total = P.reduce((a, p) => a + p.sec, 0);
const $ = id => document.getElementById(id);
let start = null, played = -1, tick = null;
try { start = Number(localStorage.getItem(KEY)) || null; } catch (e) {}
function save(v) { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch (e) {} }
function play(cue) {  // the music dips while the voice speaks
  if (!S[cue]) return;
  const a = new Audio(S[cue]);
  if (music) { music.volume = vol() * 0.35; a.onended = () => { music.volume = vol(); }; }
  a.play().catch(() => {});
}
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
function stop(reset) { clearInterval(tick); tick = null; save(null); if (reset) musicStop(); else setTimeout(musicStop, 8000); if (reset) { start = null; played = -1; render(); }
                       else { start = null; $("go").style.display = ""; $("stop").style.display = "none"; } }
$("go").onclick = () => { start = Date.now(); played = -1; save(start); musicPlay(); render(); tick = setInterval(render, 500); };
if (music) {
  $("mtoggle").onclick = () => { musicOn = !musicOn; $("mtoggle").textContent = musicOn ? "Music on" : "Music off";
                                 musicOn && start ? musicPlay() : music.pause(); };
  $("mvol").oninput = () => { music.volume = vol(); };
  // after a reload mid-meditation the browser needs a click before sound can play again
  document.getElementById("t").addEventListener("click", () => { if (start && musicOn && music.paused) musicPlay(); });
}
$("stop").onclick = () => stop(true);
render(); if (start) tick = setInterval(render, 500);
</script>
"""
