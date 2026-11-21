"""The three agents that run one sepsis assessment inside a hospital unit.

Agent 1  Data Gathering and Preprocessing   gathers the latest 16-hour window, checks it,
                                            cleans it, imputes and scales it, measures trends
Agent 2  Sepsis Prediction                  scores the window with every model, calibrates,
                                            combines with the hospital's weights, decides the alert
                                            and reports the hospital's precision at that threshold
Agent 3  Explainability and Clinical        explains which signals drove the risk, flags values
         Guidance                           outside reference ranges, writes safe guidance and
                                            the clinician report

The orchestrator is not an agent: it runs the three in order, times them, stops when data is
unusable, and writes the audit record (hospital, model version, SHA-256 of the input).
"""
from __future__ import annotations

import datetime as dt
import time
import uuid

import numpy as np

from .config import FEATURES, FIO2_MAX, N_FEATURES, WINDOW
from .inference import HospitalPredictor, window_sha256

VITALS = FEATURES[:8]
LABS = FEATURES[8:34]

# Physiologically possible values; anything outside is a recording error and is set to missing.
PLAUSIBLE = {"HR": (0, 300), "O2Sat": (0, 100), "Temp": (20, 45), "SBP": (0, 300), "MAP": (0, 300),
             "DBP": (0, 300), "Resp": (0, 100), "FiO2": (0, FIO2_MAX), "pH": (6.0, 8.0),
             "Age": (0, 120)}
# Adult reference ranges used for observational flags only.
REFERENCE = {"HR": (60, 100, "bpm", "Heart rate"), "O2Sat": (94, 100, "%", "SpO2"),
             "Temp": (36.1, 37.8, "°C", "Temperature"), "SBP": (90, 140, "mmHg", "Systolic BP"),
             "DBP": (60, 90, "mmHg", "Diastolic BP"), "MAP": (65, 110, "mmHg", "Mean arterial pressure"),
             "Resp": (12, 20, "/min", "Respiratory rate"), "Lactate": (0.5, 2.0, "mmol/L", "Lactate"),
             "WBC": (4, 11, "×10⁹/L", "White cell count"), "Creatinine": (0.6, 1.2, "mg/dL", "Creatinine"),
             "BUN": (7, 20, "mg/dL", "Urea nitrogen"), "Platelets": (150, 450, "×10⁹/L", "Platelets")}
TREND_SIGNALS = ["HR", "Temp", "SBP", "MAP", "Resp", "O2Sat"]
NAMES = {k: v[3] for k, v in REFERENCE.items()}
BLOCKED = ("diagnosis", "diagnose", "treatment", "treat ", "medication", "prescribe", "dose")


class DataError(Exception):
    pass


# --------------------------------------------------------------------------- agent 1
class DataAgent:
    name = "Data Gathering and Preprocessing Agent"

    def __init__(self, predictor: HospitalPredictor):
        self.prep = predictor.prep

    def run(self, record: np.ndarray, end_hour: int | None = None) -> dict:
        """record: the patient's hourly (T, 39) values; the window ends at end_hour (default: last)."""
        record = np.asarray(record, np.float32)
        if record.ndim != 2 or record.shape[1] != N_FEATURES:
            raise DataError(f"record must have {N_FEATURES} columns, got shape {record.shape}")
        end = len(record) - 1 if end_hour is None else int(end_hour)
        if end < WINDOW - 1 or end >= len(record):
            raise DataError(f"need {WINDOW} hours ending at hour {end + 1}; record has {len(record)} hours")
        window = record[end - WINDOW + 1:end + 1].copy()
        received_sha = window_sha256(window)          # fingerprint of the input as received

        cleaned = []
        for f, (lo, hi) in PLAUSIBLE.items():
            j = FEATURES.index(f)
            bad = ~np.isnan(window[:, j]) & ((window[:, j] < lo) | (window[:, j] > hi))
            if bad.any():
                cleaned.append({"signal": f, "hours": int(bad.sum()), "action": "set to missing"})
                window[bad, j] = np.nan
        vit = 1 - np.isnan(window[:, :8]).mean()
        lab = 1 - np.isnan(window[:, 8:34]).mean()
        if np.isnan(window[:, :8]).all():
            raise DataError("no vital signs recorded in the 16-hour window")

        trends = {}
        hours = np.arange(WINDOW)
        for f in TREND_SIGNALS:
            v = window[:, FEATURES.index(f)]
            ok = ~np.isnan(v)
            if ok.sum() >= 2:
                slope = float(np.polyfit(hours[ok], v[ok], 1)[0])
                recent = v[-5:][~np.isnan(v[-5:])]
                change = float(recent[-1] - recent[0]) if len(recent) >= 2 else None
                trends[f] = {"slope_per_hour": round(slope, 3),
                             "change_4h": None if change is None else round(change, 2)}
        return {"window": window, "tensor": self.prep.transform(window[None])[0],
                "sha256": received_sha, "start_hour": end - WINDOW + 2, "end_hour": end + 1,
                "quality": {"vitals_observed": round(float(vit), 3), "labs_observed": round(float(lab), 3),
                            "cleaned": cleaned},
                "trends": trends}


# --------------------------------------------------------------------------- agent 2
class PredictionAgent:
    name = "Sepsis Prediction Agent"

    def __init__(self, predictor: HospitalPredictor):
        self.p = predictor

    def run(self, data: dict) -> dict:
        s = self.p.score_batch(data["window"][None])
        prob = float(s["probability"][0])
        models = []
        for m in s["raw"]:
            w = float(self.p.weights.get(m, 0.0))
            c = float(s["calibrated"][m][0])
            models.append({"model": m, "raw": round(float(s["raw"][m][0]), 4), "calibrated": round(c, 4),
                           "weight": round(w, 4), "contribution_points": round(100 * w * c, 2)})
        models.sort(key=lambda r: -r["weight"])
        op = self.p.card.get("operating_point_on_validation", {})
        return {"probability": prob, "risk_percent": round(100 * prob, 2), "threshold": self.p.threshold,
                "alert": prob >= self.p.threshold, "models": models,
                "operating_point": op, "model_version": self.p.card["model_version"],
                "trained_on": self.p.card["trained_on"]}


# --------------------------------------------------------------------------- agent 3
class ExplainabilityAgent:
    name = "Explainability and Clinical Guidance Agent"

    def __init__(self, predictor: HospitalPredictor):
        self.p = predictor

    def explain(self, window: np.ndarray, prob: float, top: int = 8) -> list:
        """Occlusion: hide one signal across all 16 hours (it is then imputed), score again;
        contribution = risk with the signal - risk without it, in percentage points."""
        observed = [j for j in range(N_FEATURES) if not np.isnan(window[:, j]).all()]
        if not observed:
            return []
        batch = np.repeat(window[None], len(observed), 0)
        for k, j in enumerate(observed):
            batch[k, :, j] = np.nan
        hidden = self.p.score_batch(batch)["probability"]
        rows = [{"signal": FEATURES[j], "name": NAMES.get(FEATURES[j], FEATURES[j]),
                 "contribution_points": round(100 * (prob - float(h)), 2)} for j, h in zip(observed, hidden)]
        rows = [r for r in rows if r["contribution_points"] != 0]
        rows.sort(key=lambda r: -abs(r["contribution_points"]))
        for r in rows:
            r["effect"] = "raises risk" if r["contribution_points"] > 0 else "lowers risk"
        return rows[:top]

    @staticmethod
    def flags(window: np.ndarray) -> list:
        out = []
        for f, (lo, hi, unit, label) in REFERENCE.items():
            v = window[:, FEATURES.index(f)]
            ok = np.where(~np.isnan(v))[0]
            if not len(ok):
                continue
            last = int(ok[-1])
            val = float(v[last])
            if val < lo or val > hi:
                when = "latest hour" if last == WINDOW - 1 else f"{WINDOW - 1 - last} h before the latest hour"
                out.append({"signal": f, "observation": f"{label} {val:g} {unit}",
                            "reference": f"{lo:g}-{hi:g} {unit}", "direction": "low" if val < lo else "high",
                            "when": when})
        return out

    @staticmethod
    def guidance(alert: bool, flags: list) -> list:
        if alert:
            g = ["Elevated sepsis risk detected.", "Clinical review recommended."]
        else:
            g = ["Sepsis risk is below this hospital's alert threshold.", "Continue routine monitoring."]
        if flags:
            g.append(f"{len(flags)} recent value(s) outside the reference range; see observations.")
        g.append("This is decision support; the decision rests with the treating clinician.")
        return g

    @staticmethod
    def check_wording(lines: list) -> list:
        bad = [w.strip() for w in BLOCKED for line in lines if w in line.lower()]
        if bad:
            raise ValueError(f"guidance contains blocked wording: {sorted(set(bad))}")
        return lines

    def run(self, data: dict, pred: dict) -> dict:
        factors = self.explain(data["window"], pred["probability"])
        flags = self.flags(data["window"])
        guidance = self.check_wording(self.guidance(pred["alert"], flags))
        report = {
            "risk": f"{pred['risk_percent']:.2f}% risk of being within 6 hours of sepsis onset",
            "alert": "Alert raised" if pred["alert"] else "No alert",
            "threshold": pred["threshold"],
            "main_factors": [f"{r['name']} {r['contribution_points']:+.2f} points" for r in factors[:3]],
            "observations": [f"{x['observation']} ({x['when']}, reference {x['reference']})" for x in flags],
            "trends": data["trends"], "data_quality": data["quality"], "guidance": guidance}
        return {"factors": factors, "flags": flags, "guidance": guidance, "report": report}


# --------------------------------------------------------------------------- orchestrator
class Orchestrator:
    """Runs the three agents for one patient inside one hospital unit and keeps the audit trail."""

    def __init__(self, artifact_folder: str):
        self.predictor = HospitalPredictor(artifact_folder)
        self.agents = [DataAgent(self.predictor), PredictionAgent(self.predictor),
                       ExplainabilityAgent(self.predictor)]

    def assess(self, record: np.ndarray, patient_id: str = "", end_hour: int | None = None) -> dict:
        trace = {"trace_id": "SC-" + uuid.uuid4().hex[:6].upper(),
                 "hospital_id": self.predictor.card["hospital_id"], "patient_id": patient_id, "stages": []}
        t_all = time.perf_counter()
        data = pred = expl = None
        for agent in self.agents:
            t0 = time.perf_counter()
            stage = {"agent": agent.name, "status": "completed", "errors": []}
            try:
                if isinstance(agent, DataAgent):
                    data = agent.run(record, end_hour)
                    stage["output"] = {k: data[k] for k in ("start_hour", "end_hour", "quality", "trends")}
                elif isinstance(agent, PredictionAgent):
                    pred = agent.run(data)
                    stage["output"] = pred
                else:
                    expl = agent.run(data, pred)
                    stage["output"] = expl
            except (DataError, ValueError) as e:
                stage["status"] = "failed"
                stage["errors"].append(str(e))
            stage["ms"] = round(1000 * (time.perf_counter() - t0), 2)
            trace["stages"].append(stage)
            if stage["status"] == "failed":
                break
        trace["total_ms"] = round(1000 * (time.perf_counter() - t_all), 2)
        trace["completed"] = all(s["status"] == "completed" for s in trace["stages"]) and len(trace["stages"]) == 3
        trace["audit"] = {"trace_id": trace["trace_id"], "hospital_id": trace["hospital_id"],
                          "patient_id": patient_id, "input_sha256": data["sha256"] if data else None,
                          "model_version": self.predictor.card["model_version"],
                          "trained_on": self.predictor.card["trained_on"],
                          "ensemble_weights": self.predictor.weights, "threshold": self.predictor.threshold,
                          "probability": pred["probability"] if pred else None,
                          "alert": pred["alert"] if pred else None,
                          "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        return trace
