"""Runs the investigation loop over data revealed up to a point in time.

Each call to `step(until)` processes situations whose windows finished since the last call:
1. test open and confirmed hypotheses of that lab against the situation (natural experiments),
2. apply lifecycle rules (confirm, reject, fade, expire) and write messages,
3. update quests,
4. if the situation is a surprise, hand it to the investigator.
The same code runs locally and inside a Databricks notebook.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from bodylab.agent import checks
from bodylab.agent.investigator import RuleInvestigator
from bodylab.agent.notebook import Notebook
from bodylab.agent.tools import FLOORS, ToolContext, personal_normal
from bodylab.config import SETTINGS
from bodylab.labs import LABS, cause
from bodylab.pipeline.features import Signals, build_all


def effect_text(d: dict) -> str:
    lab, a, p = d["lab"], d["effect_abs"], d["effect_pct"]
    if not np.isfinite(a):
        return ""
    if lab == "fuel":
        return f"{abs(p):.0f}% {'bigger' if a > 0 else 'smaller'} glucose rise" if np.isfinite(p) else f"{abs(a):.0f} mg/dL {'bigger' if a > 0 else 'smaller'} rise"
    if lab == "movement":
        return f"{abs(a):.0f} bpm {'higher' if a > 0 else 'lower'} walk heart rate"
    if lab == "sleep":
        return f"sleep starts {abs(a):.0f} min {'later' if a > 0 else 'earlier'}"
    return f"{'higher' if a > 0 else 'lower'} stress signal ({a:+.1f})"


class Engine:
    def __init__(self, pid: str, minute: pd.DataFrame, meals: pd.DataFrame, notebook: Notebook | None = None, investigator=None):
        self.pid = pid
        self.minute = minute.sort_values("ts").reset_index(drop=True)
        self.meals = meals.sort_values("ts").reset_index(drop=True)
        self.notebook = notebook or Notebook(pid)
        self.investigator = investigator or RuleInvestigator()
        self.features: dict[str, pd.DataFrame] = {}
        self.until: pd.Timestamp | None = None

    @property
    def start(self) -> pd.Timestamp:
        return self.minute["ts"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.minute["ts"].max()

    def step(self, until: pd.Timestamp) -> dict:
        until = min(pd.Timestamp(until), self.end)
        self.until = until
        self.features = build_all(self.minute, self.meals, until)
        sig = Signals(self.minute, until)
        meals = self.meals[self.meals["ts"] <= until]
        ctx = ToolContext(self.pid, self.features, sig, meals, self.notebook, until)
        nb = self.notebook
        done = nb.processed_ids()
        pending = []
        for lab, df in self.features.items():
            if df.empty:
                continue
            for _, row in df[~df["sid"].isin(done)].iterrows():
                pending.append((row["end_ts"], lab, row))
        pending.sort(key=lambda r: r[0])

        new_events, new_messages = 0, len(nb.messages)
        for end_ts, lab, row in pending:
            ctx.now = end_ts
            ok = checks.passed(checks.check_situation(sig, lab, row, meals))
            if ok:
                self._test_hypotheses(ctx, lab, row, end_ts)
                nb.update_quests(lab, row)
            if self._is_surprise(row):
                new_events += 1
                self._investigate(ctx, lab, row)
            nb.processed.append({"pid": self.pid, "sid": row["sid"], "lab": lab, "ts": row["ts"], "end_ts": end_ts, "good_data": ok})

        for h in nb.hypotheses:
            if h["status"] == "testing":
                self._apply_status(ctx, h, until)
        nb.sync_discovery_status()
        return {"until": until, "processed": len(pending), "events": new_events, "messages": len(nb.messages) - new_messages}

    def run(self, step_hours: float = 6.0) -> None:
        t = self.start + pd.Timedelta(hours=step_hours)
        while t < self.end + pd.Timedelta(hours=step_hours):
            self.step(t)
            t += pd.Timedelta(hours=step_hours)

    # ------------------------------------------------------------------
    def _is_surprise(self, row: pd.Series) -> bool:
        cfg = SETTINGS.detection
        return row.get("history", 0) >= cfg.history_needed(row["lab"]) and np.isfinite(row.get("z", np.nan)) and abs(row["z"]) >= cfg.surprise_z

    def _test_hypotheses(self, ctx: ToolContext, lab: str, row: pd.Series, ts: pd.Timestamp) -> None:
        for h in self.notebook.active(lab):
            if row["ts"] <= pd.Timestamp(h["opened_at"]) or not np.isfinite(row.get(h["factor"], np.nan)):
                continue
            med, scale = personal_normal(ctx, lab, h["factor"], row["ts"], row)
            if not np.isfinite(med):
                continue
            self.notebook.record_evidence(h, row, (row[h["factor"]] - med) / scale, ts)
            self._apply_status(ctx, h, ts)

    def _apply_status(self, ctx: ToolContext, h: dict, ts: pd.Timestamp) -> None:
        nb = self.notebook
        new = nb.update_status(h, ts)
        if new is None:
            return
        lab = h["lab"]
        if new == "confirmed" and not any(d["hyp_id"] == h["hyp_id"] for d in nb.discoveries):
            floor = FLOORS.get(cause(lab, h["factor"]).unit, 1.0)
            d = nb.make_discovery(h, self.features[lab], ts, floor)
            eff = effect_text(d)
            c = cause(lab, h["factor"])
            phrase = c.phrase_high if d["side"] > 0 else c.phrase_low
            body = f"After {phrase}: {eff}. Held in {d['evidence']}." if eff else f"{h['claim']}. Held in {d['evidence']}."
            nb.message(ts, "discovery", f"Confirmed: {d['title']}", body, lab, d["card_id"])
        elif new == "confirmed":
            nb.message(ts, "discovery", f"Holding again: {h['claim']}", "Newer data agrees again.", lab, h["hyp_id"])
        elif new == "rejected":
            nb.message(ts, "rejected", f"Not confirmed: {h['claim']}", f"Held in {h['supports']} of {h['supports'] + h['contradicts']} tests. Logged as noise.", lab, h["hyp_id"])
        elif new == "fading":
            nb.message(ts, "fading", f"Fading: {h['claim']}", "Recent data disagrees, so this card is being rechecked.", lab, h["hyp_id"])

    def _investigate(self, ctx: ToolContext, lab: str, row: pd.Series) -> None:
        nb = self.notebook
        event = {
            "pid": self.pid, "event_id": f"E{len(nb.events) + 1:03d}", "sid": row["sid"], "lab": lab, "ts": row["ts"],
            "end_ts": row["end_ts"], "response": float(row["response"]), "expected": float(row["expected"]), "z": float(row["z"]),
            "verdict": "investigating", "hyp_id": None, "title": "", "message": "", "agent": "", "factor": None, "comparison_sid": None,
        }
        nb.events.append(event)
        result = self.investigator.investigate(ctx, event)
        comp = result.get("comparison") or {}
        dq = next((t["result"] for t in result.get("trace", []) if t["tool"] == "check_data_quality"), {})
        event["checks_json"] = json.dumps(dq.get("checks", []), default=str)
        event["differences_json"] = json.dumps(comp.get("differences", []), default=str)
        event["tools_json"] = json.dumps([t["tool"] for t in result.get("trace", [])])
        event.update(
            verdict=result["verdict"], title=result.get("title", ""), message=result.get("message", ""),
            agent=result.get("agent", getattr(self.investigator, "name", "rules")) + (" (fallback)" if result.get("fallback") else ""),
            factor=result.get("factor"), comparison_sid=(comp.get("comparison") or {}).get("sid"),
            n_tools=len(result.get("trace", [])), fallback_reason=result.get("fallback", ""),
            llm_requests=result.get("usage", {}).get("requests", 0),
            input_tokens=result.get("usage", {}).get("input_tokens", 0),
            output_tokens=result.get("usage", {}).get("output_tokens", 0),
        )
        if result["verdict"] == "lead":
            nb.message(row["end_ts"], "case", result.get("title", "Case solved"), result.get("message", ""), lab, event["event_id"])
            h = next((x for x in nb.hypotheses if x["hyp_id"] == event.get("hyp_id")), None)
            if h is not None:
                self._apply_status(ctx, h, row["end_ts"])
