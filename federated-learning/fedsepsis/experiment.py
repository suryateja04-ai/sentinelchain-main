"""Runs the three training schemes and writes the results the Federation page reads.

local        every hospital trains only on its own data (no collaboration)
fedavg       hospitals train together through the central server (security layer,
fedprox      orchestrator agent, aggregation); only weights, tree models and aggregate
             statistics are exchanged; each hospital then calibrates, weights and
             thresholds on its own data. FedProx adds a proximal term for non-IID data.
centralized  all data pooled in one place: the reference that federated learning tries to
             reach without moving the data
"""
from __future__ import annotations

import datetime as dt
import json
import os

import joblib
import numpy as np
import torch

from .fedstats import federated_preprocessor
from .metrics import at_threshold, evaluate, ranking
from .server import federated_training
from .unit import HospitalUnit


def train_bundle(units, neural, trees, rounds, local_epochs, mu, tree_sample, log, attack_plan=None):
    prep, prep_log = federated_preprocessor([u.train for u in units])
    for u in units:
        u.prepare(prep)
    neg = sum(u.counts()[0] for u in units)
    pos = sum(u.counts()[1] for u in units)
    pos_weight = neg / max(pos, 1)
    states, history = {}, {}
    for name in neural:
        states[name], history[name] = federated_training(units, name, rounds, local_epochs, pos_weight,
                                                         mu, attack_plan, log)
    tree_sets = []
    if trees:
        for u in units:
            log(f"  trees at hospital {u.id}")
            tree_sets.append(u.fit_trees(trees, tree_sample))
    return {"prep": prep, "prep_log": prep_log, "pos_weight": pos_weight, "states": states,
            "history": history, "tree_sets": tree_sets, "trained_on": [u.id for u in units]}


def raw_scores(bundle, unit, windows):
    raw = {n: unit.predict_neural(n, s, windows) for n, s in bundle["states"].items()}
    if bundle["tree_sets"]:
        raw.update(HospitalUnit.predict_trees(bundle["tree_sets"], windows))
    return raw


def deploy(bundle, site, eval_units):
    """Personalise the bundle at `site` and evaluate it on each eval unit's test patients."""
    site.prepare(bundle["prep"])
    val_raw = raw_scores(bundle, site, site.val)
    cal, weights, thr = site.personalise(val_raw)
    op = at_threshold(site.val.labels, HospitalUnit.ensemble(val_raw, cal, weights), thr)
    out = {"trained_on": bundle["trained_on"], "personalised_at": site.id,
           "ensemble_weights": weights, "threshold": thr,
           "operating_point": {k: op[k] for k in ("sensitivity", "specificity", "precision", "npv", "f1",
                                                 "alerts_per_100")},
           "test": {}, "models_test": {}}
    for e in eval_units:
        e.prepare(bundle["prep"])
        raw = raw_scores(bundle, e, e.test)
        p = HospitalUnit.ensemble(raw, cal, weights)
        out["test"][e.id] = evaluate(e.test, p, thr)
        out["models_test"][e.id] = {m: ranking(e.test.labels, cal[m].predict(r)) for m, r in raw.items()}
    return out, cal, weights, thr


def save_unit_artifacts(folder, unit, bundle, cal, weights, thr, version, operating_point=None):
    os.makedirs(folder, exist_ok=True)
    bundle["prep"].save(os.path.join(folder, "preprocessor.json"))
    for name, state in bundle["states"].items():
        torch.save(state, os.path.join(folder, f"{name}.pt"))
    if bundle["tree_sets"]:
        joblib.dump(bundle["tree_sets"], os.path.join(folder, "trees.joblib"))
    joblib.dump(cal, os.path.join(folder, "calibrators.joblib"))
    with open(os.path.join(folder, "model_card.json"), "w") as f:
        json.dump({"hospital_id": unit.id, "hospital_name": unit.name, "model_version": version,
                   "trained_on": bundle["trained_on"], "neural_models": list(bundle["states"]),
                   "tree_models": list(bundle["tree_sets"][0][0]) if bundle["tree_sets"] else [],
                   "ensemble_weights": weights, "threshold": thr,
                   "personalised_on": f"validation patients of hospital {unit.id}",
                   "operating_point_on_validation": operating_point or {}}, f, indent=1)


def run(units, schemes, neural, trees, rounds, local_epochs=1, mu=0.01, tree_sample=None,
        artifacts=None, synthetic=False, attack_plan=None, log=print):
    version = "fed-" + dt.date.today().strftime("%Y%m%d") + f"-r{rounds}"
    res = {"generated_at": dt.datetime.now().isoformat(timespec="seconds"), "synthetic": synthetic,
           "model_version": version,
           "config": {"rounds": rounds, "local_epochs": local_epochs, "fedprox_mu": mu,
                      "neural_models": list(neural), "tree_models": list(trees),
                      "tree_sample": tree_sample, "attack_plan": attack_plan or {},
                      "device": "cuda" if torch.cuda.is_available() else "cpu"},
           "hospitals": {u.id: u.summary() for u in units}, "schemes": {}}

    if "local" in schemes:
        res["schemes"]["local"] = {}
        for u in units:
            log(f"[local] hospital {u.id} trains alone")
            b = train_bundle([u], neural, trees, rounds, local_epochs, 0.0, tree_sample, log)
            out, *_ = deploy(b, u, units)
            out["history"] = b["history"]
            res["schemes"]["local"][u.id] = out

    res["federation"] = {}
    for scheme, scheme_mu in (("fedavg", 0.0), ("fedprox", mu)):
        if scheme not in schemes:
            continue
        log(f"[{scheme}] {len(units)} hospital clients, {rounds} rounds")
        b = train_bundle(units, neural, trees, rounds, local_epochs, scheme_mu, tree_sample, log, attack_plan)
        res["federation"][scheme] = {
            "aggregation": "FedProx" if scheme_mu > 0 else "FedAvg", "mu": scheme_mu,
            "preprocessing": b["prep_log"], "history": b["history"], "pos_weight": b["pos_weight"],
            "shared": ["model weights after each round (signed with HMAC-SHA256)", "trained tree models",
                       "window counts", "aggregate histograms and sums for preprocessing"],
            "never_shared": ["patient rows", "patient identifiers", "labels of single patients"]}
        res["schemes"][scheme] = {}
        for u in units:
            out, cal, w, thr = deploy(b, u, units)
            res["schemes"][scheme][u.id] = out
            if artifacts:
                save_unit_artifacts(os.path.join(artifacts, scheme, f"hospital_{u.id}"), u, b, cal, w, thr,
                                    f"{version}-{scheme}", out["operating_point"])

    if "centralized" in schemes:
        log("[centralized] all data pooled (reference only)")
        pooled = HospitalUnit.pooled(units)
        b = train_bundle([pooled], neural, trees, rounds, local_epochs, 0.0, tree_sample, log)
        out, *_ = deploy(b, pooled, units)
        out["history"] = b["history"]
        res["schemes"]["centralized"] = {"ALL": out}
    return res


def to_json(obj, path):
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.integer):
            return int(o)
        return o
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=1)
