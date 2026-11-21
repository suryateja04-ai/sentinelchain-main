"""Evaluation, calibration helpers and the threshold rule."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import LABEL_LEAD_HOURS, MIN_RECALL, WINDOW

EPS = 1e-7


def log_loss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins: int = 10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def ranking(y, p):
    if len(np.unique(y)) < 2:
        return {"auroc": None, "auprc": None}
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p))}


def choose_threshold(y, p, min_recall: float = MIN_RECALL):
    """Best F1 among thresholds whose recall is at least min_recall (validation data only)."""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    k = np.arange(1, len(ys) + 1)
    recall = tp / max(ys.sum(), 1)
    precision = tp / k
    f1 = 2 * precision * recall / np.maximum(precision + recall, EPS)
    last = np.r_[ps[1:] != ps[:-1], True]            # only cut between distinct scores
    ok = last & (recall >= min_recall)
    if not ok.any():
        return float(ps[-1])
    i = int(np.argmax(np.where(ok, f1, -1)))
    return float(ps[i])


def at_threshold(y, p, thr):
    a = p >= thr
    tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum())
    fn = int((~a & (y == 1)).sum()); tn = int((~a & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {"threshold": float(thr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sensitivity": sens, "specificity": spec, "precision": prec,
            "npv": tn / max(tn + fn, 1), "f1": 2 * prec * sens / max(prec + sens, EPS),
            "alerts_per_100": 100 * (tp + fp) / max(len(y), 1)}


def patient_level(windows, p, thr):
    """Early warning per patient.

    Onset hour = first hour with SepsisLabel 1 plus 6 (the label starts 6 h before onset).
    A sepsis patient is flagged early if an alert is raised at a window ending before onset;
    lead time = onset hour minus the hour of the first such alert.
    """
    alert = p >= thr
    pid = windows.index[:, 0]
    end = windows.index[:, 1]
    leads, flagged, n_sep, fa, n_non = [], 0, 0, 0, 0
    for k in np.unique(pid):
        m = pid == k
        y = windows.patients[k].y
        if y.max() > 0:
            n_sep += 1
            onset = int(np.argmax(y)) + LABEL_LEAD_HOURS
            early = m & alert & (end < onset)
            if early.any():
                flagged += 1
                leads.append(onset - int(end[early].min()))
        else:
            n_non += 1
            fa += int((m & alert).any())
    leads = np.array(leads, float)
    out = {"sepsis_patients": n_sep, "flagged_before_onset": flagged,
           "flagged_rate": flagged / max(n_sep, 1),
           "non_sepsis_patients": n_non, "non_sepsis_with_alert": fa,
           "false_alert_rate": fa / max(n_non, 1)}
    if len(leads):
        out.update({"median_lead_h": float(np.median(leads)), "mean_lead_h": float(leads.mean()),
                    "q1_lead_h": float(np.percentile(leads, 25)), "q3_lead_h": float(np.percentile(leads, 75)),
                    **{f"early_{h}h": float((leads >= h).sum() / max(n_sep, 1)) for h in (1, 2, 4, 6)}})
    return out


def evaluate(windows, p, thr) -> dict:
    y = windows.labels
    return {"n_windows": int(len(y)), **ranking(y, p), "brier": float(np.mean((p - y) ** 2)),
            "log_loss": log_loss(y, p), "ece": ece(y, p), **at_threshold(y, p, thr),
            "patient_level": patient_level(windows, p, thr)}
