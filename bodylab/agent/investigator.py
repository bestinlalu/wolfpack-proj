"""Investigators: a Gemini tool-calling agent, and a rule-based one used as fallback and in tests."""
from __future__ import annotations

import json
import logging
import re
import time

import numpy as np
import pandas as pd

from bodylab.agent import tools as T
from bodylab.config import SETTINGS, gemini_api_key
from bodylab.labs import LABS, cause

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are Body Lab's investigator, a careful personal scientist for one person's body.
A situation (a meal, a walk, a night, or an hour of the day) produced a response far from what is normal for this person.
Investigate it with the tools, in this order of thinking:
1. check_data_quality first. If any check fails, close_case with verdict bad_data.
2. find_similar_situations, then compare_situations with their ids.
3. If the top difference has |difference_z| >= 1.5, call list_hypotheses, then open_hypothesis for that factor
   (direction +1 if a higher factor value goes with a higher response, -1 otherwise), then close_case with verdict lead.
4. If nothing differs clearly, or there are no similar situations, close_case with verdict unexplained.
Rules: one event is a lead, never a finding. Say "was different", never "caused". No medical advice.
The close_case title is a headline under 14 words that leads with the why, like
"Thursday's lunch spiked higher with fewer steps before eating".
The close_case message is 1-2 short, plain sentences: quote values only from the *_text fields (response_text, usual_text,
this_time_text, typical_text), never z-scores, difference_z or raw flags, and end with the open_hypothesis result's
more_tests_needed (for example "2 more tests needed to confirm"; if 0, say it fits a confirmed discovery).
Stress values in the *_text fields are on the person's own 1-10 scale (for example "4/10"); keep that form."""

THING = {"fuel": None, "movement": "walk", "sleep": "night", "stress": None}


def _plural(word: str) -> str:
    return word + ("es" if word.endswith(("ch", "sh")) else "s")


def _thing(lab: str, s: dict) -> str:
    if lab == "fuel":
        return s.get("slot", "meal")
    if lab == "stress":
        hh = int(s["start"][11:13])
        return f"{hh:02d}:00–{hh + 2:02d}:00 window"
    return THING[lab]


def _verb(lab: str, higher: bool) -> str:
    return {
        "fuel": "spiked higher" if higher else "spiked less",
        "movement": "ran a higher heart rate" if higher else "ran a lower heart rate",
        "sleep": "started later" if higher else "started earlier",
        "stress": "ran more stressed" if higher else "ran calmer",
    }[lab]


def lead_message(lab: str, comp: dict, top: dict, hyp: dict) -> tuple[str, str]:
    s = comp["situation"]
    higher = comp["response_direction"] == "higher"
    ca = cause(lab, top["factor"])
    phrase = ca.phrase_high if (top["this_time"] or 0) > (top["similar_median"] or 0) else ca.phrase_low
    unit = LABS[lab].response_unit
    title = f"{s['weekday']}'s {_thing(lab, s)} {_verb(lab, higher)} with {phrase}"
    supports = hyp.get("supports", 1)
    if hyp.get("status") in ("confirmed", "fading"):
        progress = f"This fits a confirmed discovery ({hyp.get('hypothesis')})."
    elif supports <= 1:
        progress = f"First time seen: {SETTINGS.hypothesis.confirm_supports - 1} more tests needed to confirm."
    else:
        need = max(SETTINGS.hypothesis.confirm_supports - supports, 0)
        progress = f"Seen {supports} times: {need} more test{'s' if need != 1 else ''} needed to confirm." if need else f"Seen {supports} times."
    label = "Stress level" if lab == "stress" else LABS[lab].response_label.capitalize()
    if s["response_text"] == s["usual_text"]:  # same step at the ends of the 1-10 scale
        response = f"{s['response_text']}, {'above' if higher else 'below'} your usual even for this time of day"
    else:
        response = f"{s['response_text']} vs your usual {s['usual_text']}"
    if top["unit"] == "z" and top["this_time_text"] == top["typical_text"]:
        up = (top["this_time"] or 0) > (top["similar_median"] or 0)
        diff = f"{top['this_time_text']}, {'higher' if up else 'lower'} than usual even within that step"
    else:
        diff = f"{top['this_time_text']} vs a typical {top['typical_text']}"
    body = (f"{label}: {response}. Biggest difference: {top['label'].lower()}, {diff} "
            f"in similar {_plural(_thing(lab, s))}. {progress}")
    return title, body


class RuleInvestigator:
    name = "rules"

    def investigate(self, ctx: T.ToolContext, event: dict) -> dict:
        trace = []

        def call(name, **kw):
            out = T.TOOL_FUNCS[name](ctx, **kw)
            trace.append({"tool": name, "args": kw, "result": out})
            return out

        sid = event["sid"]
        dq = call("check_data_quality", situation_id=sid)
        if not dq["passed"]:
            failed = [c["check"] for c in dq["checks"] if not c["passed"]]
            return {"verdict": "bad_data", "title": "Dismissed: bad data", "message": "Closed quietly: " + ", ".join(failed) + ".", "trace": trace}
        sim = call("find_similar_situations", situation_id=sid)
        if not sim["similar"]:
            return {"verdict": "unexplained", "title": "Unexplained", "message": "No similar situations yet to compare with.", "trace": trace}
        comp = call("compare_situations", situation_id=sid, similar_ids=[x["sid"] for x in sim["similar"]])
        clear = [d for d in comp["differences"] if abs(d["difference_z"] or 0) >= SETTINGS.detection.clear_difference_z]
        if not clear:
            return {"verdict": "unexplained", "title": "Unexplained", "message": "Nothing stood out against similar situations; logged as normal variation.", "trace": trace, "comparison": comp}
        top = clear[0]
        direction = int(np.sign((top["this_time"] - top["similar_median"]) * (event["z"] or 0))) or 1
        hyp = call("open_hypothesis", situation_id=sid, factor=top["factor"], direction=direction)
        if "conflict" in hyp:
            return {"verdict": "unexplained", "title": "Unexplained", "message": f"Pointed against {hyp['conflict']}; counted as evidence against it.", "trace": trace, "comparison": comp}
        title, body = lead_message(event["lab"], comp, top, hyp)
        return {"verdict": "lead", "title": title, "message": body, "trace": trace, "comparison": comp, "factor": top["factor"]}


class GeminiInvestigator:
    name = "gemini"

    def __init__(self, model: str | None = None, max_turns: int = 10, wait_on_rate_limit: bool = False):
        from google import genai
        from google.genai import types

        self.types = types
        retry = types.HttpRetryOptions(attempts=3, initial_delay=1.0, max_delay=8.0, http_status_codes=[429, 500, 503, 504])
        self.client = genai.Client(api_key=gemini_api_key(), http_options=types.HttpOptions(retry_options=retry, timeout=60_000))
        self.model = model or SETTINGS.gemini_model
        self.models = [self.model] + [m for m in SETTINGS.gemini_fallback_models if m != self.model]
        self.wait_on_rate_limit = wait_on_rate_limit
        self.cases = 0
        self.usage = {"requests": 0, "input_tokens": 0, "output_tokens": 0}
        self.max_turns = max_turns
        decls = [types.FunctionDeclaration(name=s["name"], description=s["description"], parameters_json_schema=s["parameters"]) for s in T.TOOL_SPECS]
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[types.Tool(function_declarations=decls)],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.2,
        )
        self.fallback = RuleInvestigator()

    def investigate(self, ctx: T.ToolContext, event: dict) -> dict:
        before = dict(self.usage)
        out = self._investigate(ctx, event)
        out["usage"] = {k: self.usage[k] - before[k] for k in self.usage}
        return out

    def _investigate(self, ctx: T.ToolContext, event: dict) -> dict:
        budget = SETTINGS.gemini_max_cases
        if budget and self.cases >= budget:
            out = self.fallback.investigate(ctx, event)
            out["agent"] = "rules (Gemini case limit reached)"
            return out
        self.cases += 1
        try:
            return self._run(ctx, event)
        except Exception as exc:  # network, quota, malformed calls: the rules still finish the case
            log.warning("Gemini investigation failed, using rules: %s", exc)
            out = self.fallback.investigate(ctx, event)
            out["fallback"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            return out

    def _generate(self, contents):
        """Call the first model that answers; move to the next on overload, rate limit or retirement."""
        from google.genai import errors

        for attempt in range(4 if self.wait_on_rate_limit else 1):
            try:
                return self._generate_once(contents)
            except errors.APIError as exc:
                if not self.wait_on_rate_limit or getattr(exc, "code", None) != 429 or attempt == 3:
                    raise
                delay = _retry_delay(exc)
                if delay is None:  # a daily quota, not worth waiting for
                    raise
                log.warning("Gemini rate limited; waiting %ss", delay)
                time.sleep(delay)

    def _generate_once(self, contents):
        from google.genai import errors

        last = None
        for i, model in enumerate(self.models):
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=self.config)
                meta = getattr(resp, "usage_metadata", None)
                for key, attr in (("input_tokens", "prompt_token_count"), ("output_tokens", "candidates_token_count")):
                    self.usage[key] += int(getattr(meta, attr, 0) or 0)
                self.usage["requests"] += 1
                if i:  # stick with the model that worked
                    self.models = self.models[i:] + self.models[:i]
                self.used_model = model
                return resp
            except errors.APIError as exc:
                if getattr(exc, "code", None) not in (404, 429, 500, 503, 504):
                    raise
                last = exc
        raise last

    def _run(self, ctx: T.ToolContext, event: dict) -> dict:
        types = self.types
        brief = {"situation_id": event["sid"], "lab": event["lab"], "situation": T.describe(event["lab"], ctx.situation(event["sid"])[1], ctx)}
        contents = [types.Content(role="user", parts=[types.Part.from_text(text="Investigate this surprise:\n" + json.dumps(brief, default=str))])]
        trace, comparison, factor = [], None, None
        for _ in range(self.max_turns):
            resp = self._generate(contents)
            calls = resp.function_calls or []
            if not calls:
                contents.append(resp.candidates[0].content)
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text="Finish by calling close_case.")]))
                continue
            contents.append(resp.candidates[0].content)
            parts = []
            for fc in calls:
                args = dict(fc.args or {})
                if fc.name == "close_case":
                    verdict = args.get("verdict", "unexplained")
                    if verdict == "lead" and not any(t["tool"] == "open_hypothesis" and "error" not in t["result"] for t in trace):
                        verdict = "unexplained"
                    message = args.get("message", "")
                    title = (args.get("title") or message.split(". ")[0])[:140] if verdict == "lead" else ("Dismissed: bad data" if verdict == "bad_data" else "Unexplained")
                    return {"verdict": verdict, "title": title, "message": message, "trace": trace, "comparison": comparison, "factor": factor,
                            "agent": f"{self.name} ({getattr(self, 'used_model', self.model)})"}
                fn = T.TOOL_FUNCS.get(fc.name)
                result = fn(ctx, **args) if fn else {"error": f"unknown tool {fc.name}"}
                if fc.name == "compare_situations":
                    comparison = result
                if fc.name == "open_hypothesis" and "error" not in result:
                    factor = args.get("factor")
                trace.append({"tool": fc.name, "args": args, "result": result})
                parts.append(types.Part.from_function_response(name=fc.name, response={"result": json.loads(json.dumps(result, default=str))}))
            contents.append(types.Content(role="user", parts=parts))
        raise RuntimeError("agent did not close the case")


def _retry_delay(exc) -> float:
    m = re.search(r"retry in ([0-9.]+)s", str(exc)) or re.search(r"'retryDelay': '([0-9.]+)s'", str(exc))
    if not m:
        return 30.0
    seconds = float(m.group(1))
    return seconds + 1 if seconds <= 90 else None


def make_investigator(prefer_llm: bool = True, wait_on_rate_limit: bool = False):
    if prefer_llm and gemini_api_key():
        try:
            return GeminiInvestigator(wait_on_rate_limit=wait_on_rate_limit)
        except Exception as exc:
            log.warning("Gemini unavailable, using rules: %s", exc)
    return RuleInvestigator()
