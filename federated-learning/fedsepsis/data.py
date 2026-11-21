"""Loading one hospital's PhysioNet files and turning them into 16-hour windows.

Everything in this file runs inside a hospital unit: raw rows never leave it.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import FEATURES, FIO2_MAX, LABEL, SEED, SPLIT, WINDOW


@dataclass
class Patient:
    pid: str
    x: np.ndarray            # (T, 39) float32, NaN where not measured
    y: np.ndarray            # (T,) int8 SepsisLabel per hour


def read_patient(path: str) -> Patient:
    df = pd.read_csv(path, sep="|")
    missing = [c for c in FEATURES + [LABEL] if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)}: missing columns {missing}")
    x = df[FEATURES].to_numpy(dtype=np.float32)
    fio2 = FEATURES.index("FiO2")
    x[x[:, fio2] > FIO2_MAX, fio2] = np.nan                    # cleaning step
    y = df[LABEL].fillna(0).to_numpy(dtype=np.int8)
    return Patient(os.path.splitext(os.path.basename(path))[0], x, y)


def load_hospital(folder: str, limit: int | None = None) -> list[Patient]:
    files = sorted(glob.glob(os.path.join(folder, "*.psv")))
    if not files:
        raise FileNotFoundError(f"no .psv files in {folder}")
    if limit:
        files = files[:limit]
    return [read_patient(f) for f in files]


def split_patients(patients: list[Patient], seed: int = SEED):
    """Patient-level split: 80/20 into train+val / test, then 80/20 into train / val."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(patients))
    n_rest = int(round(len(idx) * SPLIT[0]))
    rest, test = idx[:n_rest], idx[n_rest:]
    n_train = int(round(len(rest) * SPLIT[1]))
    train, val = rest[:n_train], rest[n_train:]
    pick = lambda ids: [patients[i] for i in sorted(ids)]
    return pick(train), pick(val), pick(test)


@dataclass
class WindowSet:
    """All 16-hour windows of a group of patients, stored without copying.

    A window ends at every hour t >= 16 (index t-1 >= 15) and takes the label of that hour.
    """
    patients: list[Patient]
    index: np.ndarray = field(init=False)     # (n, 2): patient number, end row
    labels: np.ndarray = field(init=False)

    def __post_init__(self):
        rows, labels = [], []
        for p, pat in enumerate(self.patients):
            ends = np.arange(WINDOW - 1, len(pat.y))
            rows.append(np.stack([np.full(len(ends), p), ends], 1))
            labels.append(pat.y[ends])
        self.index = np.concatenate(rows) if rows else np.zeros((0, 2), int)
        self.labels = np.concatenate(labels).astype(np.float32) if labels else np.zeros(0, np.float32)

    def __len__(self):
        return len(self.index)

    @property
    def n_patients_with_window(self):
        return len(np.unique(self.index[:, 0])) if len(self.index) else 0

    def raw(self, i: int) -> np.ndarray:
        p, end = self.index[i]
        return self.patients[p].x[end - WINDOW + 1:end + 1]

    def prepare(self, prep) -> None:
        """Impute and scale every hour once, so batches are simple slices."""
        self._rows = np.concatenate([prep.transform(p.x) for p in self.patients]) if self.patients \
            else np.zeros((0, len(FEATURES)), np.float32)
        offsets = np.cumsum([0] + [len(p.y) for p in self.patients])[:-1]
        self._start = offsets[self.index[:, 0]] + self.index[:, 1] - WINDOW + 1

    def batch(self, ids: np.ndarray) -> np.ndarray:
        """Preprocessed windows, shape (len(ids), 16, 39); call prepare() first."""
        return self._rows[self._start[ids][:, None] + np.arange(WINDOW)]

    def flat(self, ids: np.ndarray | None = None) -> np.ndarray:
        ids = np.arange(len(self)) if ids is None else ids
        return self.batch(ids).reshape(len(ids), -1)

    def row_weights(self):
        """How many windows each hourly row belongs to, per patient.

        Medians and means are taken over the values inside the training windows,
        so an hour that appears in many windows counts that many times.
        """
        for pat in self.patients:
            t = np.arange(len(pat.y))
            last = len(pat.y) - 1
            w = np.minimum(t + WINDOW - 1, last) - np.maximum(WINDOW - 1, t) + 1
            yield pat.x, np.clip(w, 0, None).astype(np.float64)

    def summary(self) -> dict:
        pos = int(self.labels.sum())
        return {"patients": len(self.patients), "patients_with_window": self.n_patients_with_window,
                "windows": len(self), "positive_windows": pos,
                "positive_rate": round(pos / max(len(self), 1), 5)}


def icu_type(p: Patient) -> str:
    """MICU / SICU from the Unit1 / Unit2 flags (first hour where either is recorded)."""
    u1, u2 = FEATURES.index("Unit1"), FEATURES.index("Unit2")
    for row in p.x:
        if row[u1] == 1:
            return "MICU"
        if row[u2] == 1:
            return "SICU"
    return "Unknown"


def split_by_icu(patients: list[Patient]) -> dict:
    groups = {}
    for p in patients:
        groups.setdefault(icu_type(p), []).append(p)
    return groups


def missingness(patients: list[Patient]) -> dict:
    x = np.concatenate([p.x for p in patients])
    return {f: round(float(np.isnan(x[:, j]).mean()), 4) for j, f in enumerate(FEATURES)}
