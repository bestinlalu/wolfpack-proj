"""The agent's lab notebook: events, hypotheses, evidence, discoveries, quests and messages.

State lives in plain tables (lists of dicts here, Delta or parquet tables when stored), never in
the language model. Hypothesis lifecycle rules live here too.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bodylab.config import SETTINGS
from bodylab.labs import COMMON_PATTERNS, CROSS_LAB_CAUSES, LABS, cause

TABLES = ("events", "hypotheses", "evidence", "discoveries", "quests", "messages", "processed")

TITLES = {
    ("fuel", "steps_before", -1): "The Pre-Meal Walk",
    ("fuel", "steps_after", -1): "The After-Meal Walk",
    ("fuel", "stress_before", 1): "Stress Multiplier",
    ("fuel", "prev_sleep_h", -1): "Short-Night Spikes",
    ("fuel", "prev_sleep_h", 1): "Short-Night Meals",
    ("fuel", "hour", 1): "Late Meal Penalty",
    ("fuel", "start_glucose", 1): "Head Start",
    ("movement", "prev_sleep_h", -1): "Short-Night Walks",
    ("movement", "stress_before", 1): "Stressed Strides",
    ("movement", "hour", 1): "Evening Engine",
    ("sleep", "late_steps", 1): "Night Owl Lag",
    ("sleep", "dinner_gap_h", -1): "Late Dinner, Late Sleep",
    ("sleep", "day_stress", 1): "Wired at Night",
    ("stress", "prev_sleep_h", -1): "Running on Empty",
    ("stress", "weekend", -1): "Weekday Wave",
}

QUESTS = {
    ("fuel", "steps_before", -1): ("Walk 10 minutes before any meal", "steps_before", ">=", 800.0, 3),
    ("fuel", "steps_after", -1): ("Walk 10 minutes after any meal", "steps_after", ">=", 800.0, 3),
    ("fuel", "stress_before", 1): ("Eat one meal while calm", "stress_before", "<=", 0.5, 2),
    ("movement", "prev_sleep_h", -1): ("Sleep 7+ hours before a walk day", "prev_sleep_h", ">=", 7.0, 2),
    ("sleep", "late_steps", 1): ("Keep it calm after 21:00 twice", "late_steps", "<=", 500.0, 2),
    ("sleep", "dinner_gap_h", -1): ("Finish dinner 3+ hours before bed", "dinner_gap_h", ">=", 3.0, 2),
}

RANKS = [(0, "Rookie"), (3, "Sleuth"), (8, "Detective"), (20, "Commissioner")]
RARITY_POINTS = {"common": 1, "rare": 2, "legendary": 3}


def claim_text(lab: str, factor: str, direction: int, side: int = 1) -> str:
    """Phrased from the side of the factor that was seen, e.g. 'A shorter night's sleep → higher walk heart rate'."""
    c = cause(lab, factor)
    resp = LABS[lab].response_label
    phrase = c.phrase_high if side >= 0 else c.phrase_low
    up = direction * (1 if side >= 0 else -1) > 0
    if lab == "sleep":
        effect = "later sleep onset" if up else "earlier sleep onset"
    else:
        effect = f"higher {resp}" if up else f"lower {resp}"
    return f"{phrase[0].upper() + phrase[1:]} → {effect}"


@dataclass
class Notebook:
    pid: str
    events: list[dict] = field(default_factory=list)
    hypotheses: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    discoveries: list[dict] = field(default_factory=list)
    quests: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    processed: list[dict] = field(default_factory=list)

    # ---------- persistence ----------
    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {t: pd.DataFrame(getattr(self, t)) for t in TABLES}

    @classmethod
    def from_frames(cls, pid: str, frames: dict[str, pd.DataFrame]) -> "Notebook":
        nb = cls(pid)
        for t in TABLES:
            df = frames.get(t)
            if df is not None and not df.empty:
                rows = df[df["pid"] == pid] if "pid" in df.columns else df
                setattr(nb, t, rows.to_dict("records"))
        return nb

    # ---------- lookups ----------
    def processed_ids(self) -> set[str]:
        return {p["sid"] for p in self.processed}

    def hyp(self, hyp_id: str) -> dict:
        return next(h for h in self.hypotheses if h["hyp_id"] == hyp_id)

    def active(self, lab: str | None = None) -> list[dict]:
        return [h for h in self.hypotheses if h["status"] in ("testing", "confirmed", "fading") and (lab is None or h["lab"] == lab)]

    def open_testing(self) -> list[dict]:
        return [h for h in self.hypotheses if h["status"] == "testing"]

    def event(self, event_id: str) -> dict:
        return next(e for e in self.events if e["event_id"] == event_id)

    def message(self, ts: pd.Timestamp, kind: str, title: str, body: str, lab: str, ref: str) -> None:
        self.messages.append({"pid": self.pid, "msg_id": f"msg{len(self.messages) + 1:04d}", "ts": ts, "kind": kind, "title": title, "body": body, "lab": lab, "ref": ref})

    # ---------- hypotheses ----------
    def open_hypothesis(self, event: dict, lab: str, factor: str, direction: int, ts: pd.Timestamp, side: int = 1) -> tuple[dict, str]:
        """Open a hypothesis or add support to an existing one. Returns (hypothesis, note)."""
        cfg = SETTINGS.hypothesis
        existing = [h for h in self.active(lab) if h["factor"] == factor and h["direction"] == direction]
        if existing:
            h = existing[0]
            self._add_evidence(h, event["sid"], "supports", ts, event.get("z", np.nan), np.nan)
            return h, "existing hypothesis gained support"
        c = cause(lab, factor)
        testing = self.open_testing()
        dropped = None
        if c.meal_related and sum(1 for h in testing if h["meal_related"]) >= cfg.max_open_meal:
            dropped = self._weakest([h for h in testing if h["meal_related"]])
        elif len(testing) >= cfg.max_open:
            dropped = self._weakest(testing)
        if dropped is not None:
            dropped["status"] = "inconclusive"
            dropped["closed_at"] = ts
        h = {
            "pid": self.pid, "hyp_id": f"H{len(self.hypotheses) + 1}", "lab": lab, "factor": factor, "direction": int(direction),
            "claim": claim_text(lab, factor, direction, side), "meal_related": c.meal_related, "side": int(side),
            "cross_lab": factor in CROSS_LAB_CAUSES.get(lab, set()),
            "status": "testing", "supports": 0, "contradicts": 0, "chances": 0,
            "opened_at": ts, "last_tested_at": ts, "closed_at": pd.NaT, "trigger_event": event["event_id"],
        }
        self.hypotheses.append(h)
        self._add_evidence(h, event["sid"], "supports", ts, event.get("z", np.nan), np.nan)
        note = "opened" + (f"; {dropped['hyp_id']} dropped to stay under the slot limit" if dropped is not None else "")
        return h, note

    def _weakest(self, hyps: list[dict]) -> dict:
        return min(hyps, key=lambda h: (h["supports"] - h["contradicts"], -pd.Timestamp(h["opened_at"]).value))

    def _add_evidence(self, h: dict, sid: str, verdict: str, ts: pd.Timestamp, resp_z: float, factor_z: float) -> None:
        if any(e["hyp_id"] == h["hyp_id"] and e["sid"] == sid for e in self.evidence):
            return
        self.evidence.append({"pid": self.pid, "hyp_id": h["hyp_id"], "sid": sid, "verdict": verdict, "ts": ts, "resp_z": resp_z, "factor_z": factor_z})
        h["chances"] += 1
        h["last_tested_at"] = ts
        if verdict == "supports":
            h["supports"] += 1
        elif verdict == "contradicts":
            h["contradicts"] += 1

    def record_evidence(self, h: dict, situation: pd.Series, factor_z: float, ts: pd.Timestamp) -> str | None:
        """Test one hypothesis against one situation. Returns the verdict or None if not a test."""
        cfg = SETTINGS.hypothesis
        if not np.isfinite(factor_z) or abs(factor_z) < cfg.factor_present_z or not np.isfinite(situation["z"]):
            return None
        expected_sign = h["direction"] * np.sign(factor_z)
        signed = situation["z"] * expected_sign
        verdict = "supports" if signed >= cfg.response_z else "contradicts" if signed <= -cfg.response_z else "neutral"
        self._add_evidence(h, situation["sid"], verdict, ts, float(situation["z"]), float(factor_z))
        return verdict

    def update_status(self, h: dict, ts: pd.Timestamp) -> str | None:
        """Apply lifecycle rules. Returns the new status if it changed."""
        cfg = SETTINGS.hypothesis
        s, c = h["supports"], h["contradicts"]
        old = h["status"]
        if old == "testing":
            if s >= cfg.confirm_supports and s / max(s + c, 1) >= cfg.confirm_ratio:
                h["status"] = "confirmed"
            elif c > s and s + c >= 3:
                h["status"] = "rejected"
            elif h["chances"] >= cfg.max_chances + 1:
                h["status"] = "inconclusive"
            elif (ts - pd.Timestamp(h["last_tested_at"])) > pd.Timedelta(days=cfg.expire_days):
                h["status"] = "expired"
        elif old in ("confirmed", "fading"):
            recent = [e for e in self.evidence if e["hyp_id"] == h["hyp_id"] and e["verdict"] != "neutral"][-cfg.fading_window:]
            if len(recent) >= cfg.fading_window:
                ratio = sum(e["verdict"] == "supports" for e in recent) / len(recent)
                if ratio < cfg.fading_ratio:
                    h["status"] = "rejected" if old == "fading" and ratio <= 0.25 else "fading"
                elif old == "fading" and ratio >= cfg.confirm_ratio:
                    h["status"] = "confirmed"
        if h["status"] != old:
            if h["status"] in ("rejected", "inconclusive", "expired"):
                h["closed_at"] = ts
            return h["status"]
        return None

    # ---------- discoveries ----------
    def make_discovery(self, h: dict, situations: pd.DataFrame, ts: pd.Timestamp, floor: float = 1e-9) -> dict:
        """Effect = mean residual on the side of the factor that triggered the hypothesis, minus the rest,
        within the trigger's context (same meal slot, same stress window)."""
        tested = [e for e in self.evidence if e["hyp_id"] == h["hyp_id"] and e["verdict"] != "neutral"]
        rows = situations[situations["ts"] <= ts].dropna(subset=[h["factor"], "resid"])
        trigger = situations[situations["sid"] == next((e["sid"] for e in self.events if e["event_id"] == h["trigger_event"]), None)]
        if len(trigger):
            for col in ("slot", "hour") if h["lab"] in ("fuel", "stress") else ():
                if col in rows and (h["lab"] == "fuel" or col == "hour"):
                    same = rows[rows[col] == trigger.iloc[0][col]]
                    if len(same) >= 3:
                        rows = same
                    break
        effect_abs, effect_pct, base = np.nan, np.nan, np.nan
        side = h.get("side", 1) or 1
        if len(rows) >= 3:
            f = rows[h["factor"]]
            spread = max(1.4826 * float((f - f.median()).abs().median()), floor)
            dev = (f - f.median()) * side
            on_side = dev > SETTINGS.hypothesis.factor_present_z * spread
            other = dev < -SETTINGS.hypothesis.factor_present_z * spread
            if other.sum() < 2:
                other = ~on_side
            calm = other & (rows["z"].abs() < SETTINGS.detection.surprise_z)  # surprises have their own explanations
            if calm.sum() >= 2:
                other = calm
            if on_side.sum() and other.sum():
                col = "resid" if h["lab"] in ("fuel", "movement") else "response"  # carbs and pace need adjusting for
                effect_abs = float(rows.loc[on_side, col].median() - rows.loc[other, col].median())
                base = float(rows.loc[other, "response"].median())
                if h["lab"] == "fuel" and base and np.isfinite(base):
                    effect_pct = effect_abs / base * 100
        lab = h["lab"]
        key = (lab, h["factor"], h["direction"])
        if (lab, h["factor"]) in COMMON_PATTERNS:
            rarity = "common"
        elif h["cross_lab"] and h["supports"] >= 3:
            rarity = "legendary" if lab == "fuel" else "rare"
        else:
            rarity = "rare"
        d = {
            "pid": self.pid, "card_id": f"D{len(self.discoveries) + 1}", "hyp_id": h["hyp_id"], "lab": lab,
            "title": TITLES.get(key, f"The {cause(lab, h['factor']).label} Effect"), "claim": h["claim"],
            "effect_abs": effect_abs, "effect_pct": effect_pct, "base_response": base, "unit": LABS[lab].response_unit,
            "evidence": f"{h['supports']} of {h['supports'] + h['contradicts']} tests", "rarity": rarity,
            "confirmed_at": ts, "status": "confirmed", "n_tested": len(tested), "side": side, "factor": h["factor"],
        }
        self.discoveries.append(d)
        q = QUESTS.get(key)
        if q:
            self.quests.append({"pid": self.pid, "quest_id": f"Q{len(self.quests) + 1}", "card_id": d["card_id"], "lab": lab, "title": q[0], "feature": q[1], "op": q[2], "threshold": q[3], "target": q[4], "progress": 0, "started_at": ts, "done": False})
        return d

    def sync_discovery_status(self) -> None:
        by_h = {h["hyp_id"]: h for h in self.hypotheses}
        for d in self.discoveries:
            st = by_h[d["hyp_id"]]["status"]
            d["status"] = st if st in ("confirmed", "fading", "rejected") else d["status"]

    def update_quests(self, lab: str, situation: pd.Series) -> None:
        for q in self.quests:
            if q["done"] or q["lab"] != lab or situation["ts"] < pd.Timestamp(q["started_at"]):
                continue
            v = situation.get(q["feature"], np.nan)
            if np.isfinite(v) and (v >= q["threshold"] if q["op"] == ">=" else v <= q["threshold"]):
                q["progress"] += 1
                q["done"] = q["progress"] >= q["target"]

    # ---------- summary ----------
    def points(self) -> int:
        return sum(RARITY_POINTS.get(d["rarity"], 1) for d in self.discoveries if d["status"] != "rejected")

    def rank(self) -> tuple[str, int, int | None]:
        pts = self.points()
        name = RANKS[0][1]
        nxt = None
        for i, (threshold, rank_name) in enumerate(RANKS):
            if pts >= threshold:
                name = rank_name
                nxt = RANKS[i + 1][0] if i + 1 < len(RANKS) else None
        return name, pts, nxt

    def funnel(self) -> dict[str, int]:
        ev = self.events
        return {
            "surprises": len(ev),
            "bad_data": sum(e["verdict"] == "bad_data" for e in ev),
            "unexplained": sum(e["verdict"] == "unexplained" for e in ev),
            "leads": sum(e["verdict"] == "lead" for e in ev),
            "hypotheses": len(self.hypotheses),
            "discoveries": sum(d["status"] != "rejected" for d in self.discoveries),
        }
