"""Federated preprocessing: a global median imputer and scaler without sharing rows.

Round 1: every hospital sends the per-variable minimum and maximum of its training values.
Round 2: every hospital sends a weighted histogram (fixed bins between the global min and
         max), the observed weight, the weighted sum and sum of squares of its values.
The server merges these aggregates into a global median, mean and standard deviation.
Only counts and sums leave a hospital, never a patient row.
"""
from __future__ import annotations

import json
import warnings

import numpy as np

from .config import HIST_BINS, N_FEATURES


def local_range(windows) -> dict:
    lo = np.full(N_FEATURES, np.inf)
    hi = np.full(N_FEATURES, -np.inf)
    for x, w in windows.row_weights():
        keep = w > 0
        if keep.any():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            lo = np.fmin(lo, np.nanmin(np.where(keep[:, None], x, np.nan), 0))
            hi = np.fmax(hi, np.nanmax(np.where(keep[:, None], x, np.nan), 0))
    return {"min": lo.tolist(), "max": hi.tolist()}


def merge_range(parts: list[dict]):
    lo = np.nanmin([p["min"] for p in parts], 0)
    hi = np.nanmax([p["max"] for p in parts], 0)
    lo = np.where(np.isfinite(lo), lo, 0.0)
    hi = np.where(np.isfinite(hi) & (hi > lo), hi, lo + 1.0)
    return lo, hi


def local_moments(windows, lo, hi) -> dict:
    hist = np.zeros((N_FEATURES, HIST_BINS))
    n_obs = np.zeros(N_FEATURES)
    n_all = 0.0
    s1 = np.zeros(N_FEATURES)
    s2 = np.zeros(N_FEATURES)
    for x, w in windows.row_weights():
        n_all += w.sum()
        for j in range(N_FEATURES):
            v = x[:, j]
            ok = ~np.isnan(v) & (w > 0)
            if not ok.any():
                continue
            b = np.clip(((v[ok] - lo[j]) / (hi[j] - lo[j]) * HIST_BINS).astype(int), 0, HIST_BINS - 1)
            hist[j] += np.bincount(b, weights=w[ok], minlength=HIST_BINS)
            n_obs[j] += w[ok].sum()
            s1[j] += (w[ok] * v[ok]).sum()
            s2[j] += (w[ok] * v[ok].astype(np.float64) ** 2).sum()
    return {"hist": hist, "n_obs": n_obs, "n_all": n_all, "s1": s1, "s2": s2}


class Preprocessor:
    """Median imputation followed by standard scaling, one value per variable."""

    def __init__(self, median, mean, std):
        self.median = np.asarray(median, np.float32)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.where(np.isnan(x), self.median, x)
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self):
        return {"median": self.median.tolist(), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["median"], d["mean"], d["std"])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


def merge_moments(parts: list[dict], lo, hi) -> Preprocessor:
    hist = sum(p["hist"] for p in parts)
    n_obs = sum(p["n_obs"] for p in parts)
    n_all = sum(p["n_all"] for p in parts)
    s1 = sum(p["s1"] for p in parts)
    s2 = sum(p["s2"] for p in parts)
    width = (hi - lo) / HIST_BINS
    median = np.zeros(N_FEATURES)
    for j in range(N_FEATURES):
        if n_obs[j] == 0:
            continue
        c = np.cumsum(hist[j])
        b = int(np.searchsorted(c, c[-1] / 2))
        before = c[b - 1] if b else 0.0
        frac = (c[-1] / 2 - before) / max(hist[j, b], 1e-12)
        median[j] = lo[j] + (b + frac) * width[j]
    # After imputation the missing entries equal the median, so the moments stay exact.
    n_mis = n_all - n_obs
    mean = (s1 + n_mis * median) / max(n_all, 1)
    sq = (s2 + n_mis * median ** 2) / max(n_all, 1)
    std = np.sqrt(np.maximum(sq - mean ** 2, 0))
    std[std < 1e-6] = 1.0
    return Preprocessor(median, mean, std)


def federated_preprocessor(unit_windows: list) -> tuple[Preprocessor, dict]:
    """Run both rounds over the units' training windows; returns the preprocessor and a log
    of how many numbers each hospital shared."""
    ranges = [local_range(w) for w in unit_windows]
    lo, hi = merge_range(ranges)
    moments = [local_moments(w, lo, hi) for w in unit_windows]
    shared = N_FEATURES * 2 + N_FEATURES * (HIST_BINS + 3) + 1
    return merge_moments(moments, lo, hi), {"numbers_shared_per_hospital": int(shared),
                                            "patient_rows_shared": 0}
