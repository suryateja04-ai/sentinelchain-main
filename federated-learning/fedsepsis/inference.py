"""Loading a hospital unit's deployed models and scoring windows with them."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import joblib
import numpy as np
import torch

from . import models as M
from .config import N_FEATURES, WINDOW
from .fedstats import Preprocessor


def window_sha256(window: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(window, np.float32).tobytes()).hexdigest()


class HospitalPredictor:
    def __init__(self, folder):
        with open(os.path.join(folder, "model_card.json")) as f:
            self.card = json.load(f)
        with open(os.path.join(folder, "preprocessor.json")) as f:
            self.prep = Preprocessor.from_dict(json.load(f))
        self.neural = {}
        for name in self.card["neural_models"]:
            m = M.build(name)
            m.load_state_dict(torch.load(os.path.join(folder, f"{name}.pt"), map_location="cpu"))
            self.neural[name] = m.eval()
        trees = os.path.join(folder, "trees.joblib")
        self.tree_sets = joblib.load(trees) if os.path.exists(trees) else []
        self.cal = joblib.load(os.path.join(folder, "calibrators.joblib"))
        self.weights = self.card["ensemble_weights"]
        self.threshold = self.card["threshold"]

    @torch.no_grad()
    def score_batch(self, windows: np.ndarray) -> dict:
        """windows: raw (n, 16, 39) with NaN for missing. Returns the ensemble probability
        and every model's raw and calibrated score, each of length n."""
        windows = np.asarray(windows, np.float32)
        if windows.ndim != 3 or windows.shape[1:] != (WINDOW, N_FEATURES):
            raise ValueError(f"expected (n, {WINDOW}, {N_FEATURES}), got {windows.shape}")
        x = self.prep.transform(windows)
        raw = {n: torch.sigmoid(m(torch.from_numpy(x))).numpy() for n, m in self.neural.items()}
        if self.tree_sets:
            flat = x.reshape(len(x), -1)
            total = sum(n for _, n in self.tree_sets)
            for name in self.tree_sets[0][0]:
                raw[name] = sum(t[name].predict_proba(flat)[:, 1] * n for t, n in self.tree_sets) / total
        calibrated = {m: self.cal[m].predict(r) for m, r in raw.items()}
        p = sum(calibrated[m] * w for m, w in self.weights.items())
        return {"probability": np.asarray(p, float), "raw": raw, "calibrated": calibrated}

    def predict(self, window: np.ndarray) -> dict:
        """One raw (16, 39) window, with the audit fields."""
        s = self.score_batch(np.asarray(window, np.float32)[None])
        p = float(s["probability"][0])
        return {"probability": p, "alert": p >= self.threshold,
                "raw": {m: float(v[0]) for m, v in s["raw"].items()},
                "calibrated": {m: float(v[0]) for m, v in s["calibrated"].items()},
                "audit": {"hospital_id": self.card["hospital_id"],
                          "model_version": self.card["model_version"],
                          "trained_on": self.card["trained_on"],
                          "input_sha256": window_sha256(window),
                          "ensemble_weights": self.weights, "threshold": self.threshold,
                          "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}}
