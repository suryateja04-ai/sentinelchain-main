"""A hospital unit: owns its data, trains locally and personalises the shared models.

Nothing in this class sends raw rows anywhere. What it gives to the server is limited to
model weights, trained tree models, sample counts and the aggregate statistics in fedstats.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from . import models as M
from .data import WindowSet, load_hospital, missingness, split_patients
from .metrics import choose_threshold, log_loss

BATCH = 256
LR = 1e-3
TREES = ("lightgbm", "xgboost")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HospitalUnit:
    def __init__(self, hid: str, name: str, patients):
        self.id, self.name = hid, name
        self.train_p, self.val_p, self.test_p = split_patients(patients)
        self.train, self.val, self.test = (WindowSet(self.train_p), WindowSet(self.val_p),
                                           WindowSet(self.test_p))
        self.missing = missingness(self.train_p)

    @classmethod
    def from_folder(cls, hid, name, folder, limit=None):
        return cls(hid, name, load_hospital(folder, limit))

    @classmethod
    def pooled(cls, units):
        """The centralised reference: one site that holds every hospital's data."""
        u = cls.__new__(cls)
        u.id, u.name = "ALL", "All hospitals pooled"
        u.train_p = [p for x in units for p in x.train_p]
        u.val_p = [p for x in units for p in x.val_p]
        u.test_p = [p for x in units for p in x.test_p]
        u.train, u.val, u.test = WindowSet(u.train_p), WindowSet(u.val_p), WindowSet(u.test_p)
        u.missing = missingness(u.train_p)
        return u

    def summary(self):
        return {"id": self.id, "name": self.name, "train": self.train.summary(),
                "validation": self.val.summary(), "test": self.test.summary(),
                "missing_in_training": self.missing}

    def counts(self):
        pos = float(self.train.labels.sum())
        return len(self.train) - pos, pos

    def prepare(self, prep):
        self.prep = prep
        for w in (self.train, self.val, self.test):
            w.prepare(prep)

    # ---------------------------------------------------------------- neural models
    def fit_local(self, name, state, epochs, pos_weight, mu=0.0, seed=0):
        """Local training from the global weights; mu > 0 adds the FedProx proximal term."""
        dev = device()
        model = M.build(name).to(dev)
        model.load_state_dict(state)
        anchor = [p.detach().clone() for p in model.parameters()] if mu > 0 else None
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev))
        rng = np.random.default_rng(seed)
        y_all = self.train.labels
        model.train()
        for _ in range(epochs):
            order = rng.permutation(len(self.train))
            for s in range(0, len(order), BATCH):
                ids = order[s:s + BATCH]
                x = torch.from_numpy(self.train.batch(ids)).to(dev)
                y = torch.from_numpy(y_all[ids]).to(dev)
                loss = loss_fn(model(x), y)
                if anchor is not None:
                    loss = loss + mu / 2 * sum(((p - a) ** 2).sum()
                                               for p, a in zip(model.parameters(), anchor))
                opt.zero_grad()
                loss.backward()
                opt.step()
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}, len(self.train)

    @torch.no_grad()
    def predict_neural(self, name, state, windows):
        dev = device()
        model = M.build(name).to(dev)
        model.load_state_dict(state)
        model.eval()
        out = []
        for s in range(0, len(windows), 4096):
            ids = np.arange(s, min(s + 4096, len(windows)))
            out.append(torch.sigmoid(model(torch.from_numpy(windows.batch(ids)).to(dev))).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    # ---------------------------------------------------------------- tree models
    def fit_trees(self, which=TREES, sample=None, seed=0):
        """Gradient-boosted trees cannot be averaged weight by weight, so every hospital
        trains its own and shares the trained model (never the data)."""
        ids = np.arange(len(self.train))
        if sample and sample < len(ids):
            ids = np.sort(np.random.default_rng(seed).choice(ids, sample, replace=False))
        x, y = self.train.flat(ids), self.train.labels[ids]
        neg, pos = float((y == 0).sum()), float(y.sum())
        spw = neg / max(pos, 1)
        trees = {}
        if "lightgbm" in which:
            import lightgbm as lgb
            trees["lightgbm"] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                                   scale_pos_weight=spw, random_state=seed, verbose=-1).fit(x, y)
        if "xgboost" in which:
            import xgboost as xgb
            trees["xgboost"] = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                 tree_method="hist", scale_pos_weight=spw,
                                                 random_state=seed, n_jobs=-1).fit(x, y)
        return trees, len(ids)

    @staticmethod
    def predict_trees(tree_sets, windows):
        """tree_sets: list of (trees dict, n_train). Each tree type is averaged over the
        hospitals that trained it, weighted by their training windows."""
        x = windows.flat()
        out = {}
        for name in tree_sets[0][0]:
            total = sum(n for _, n in tree_sets)
            out[name] = sum(t[name].predict_proba(x)[:, 1] * n for t, n in tree_sets) / total
        return out

    # ---------------------------------------------------------------- personalisation
    def personalise(self, val_raw: dict):
        """Fit this hospital's own calibrators, ensemble weights and threshold on its own
        validation patients."""
        y = self.val.labels
        cal = {m: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)
               for m, p in val_raw.items()}
        names = list(val_raw)
        P = np.stack([cal[m].predict(val_raw[m]) for m in names], 1)
        res = minimize(lambda w: log_loss(y, P @ w), np.full(len(names), 1 / len(names)),
                       method="SLSQP", bounds=[(0, 1)] * len(names),
                       constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
        w = np.clip(res.x, 0, None)
        w = w / w.sum()
        thr = choose_threshold(y, P @ w)
        return cal, dict(zip(names, w.round(4).tolist())), thr

    @staticmethod
    def ensemble(raw: dict, cal, weights):
        return sum(cal[m].predict(raw[m]) * w for m, w in weights.items())
