"""Central server: Security Layer -> Orchestrator Agent -> Federated Aggregation -> Global Model.

Each round:
  1. the global model is sent to every hospital client (universal update),
  2. every client trains locally on its own data and signs its update,
  3. the security layer verifies each signature,
  4. the orchestrator agent validates and filters the updates,
  5. the accepted updates are averaged, weighted by training windows (FedAvg); with
     FedProx the clients add a proximal term during local training (mu > 0),
  6. the new global model is evaluated on every client's validation patients.
"""
from __future__ import annotations

import numpy as np
import torch

from . import models as M
from .metrics import at_threshold, choose_threshold, ranking
from .orchestrator import OrchestratorAgent
from .security import SecurityLayer, digest


def average(states, weights):
    total = float(sum(weights))
    return {k: sum(s[k].float() * (w / total) for s, w in zip(states, weights)).to(states[0][k].dtype)
            for k in states[0]}


def n_bytes(state):
    return int(sum(v.numel() * v.element_size() for v in state.values()))


def attack(state, base, kind, rng):
    """Simulated faulty or malicious client, used only to demonstrate the orchestrator."""
    if kind == "scale":
        return {k: base[k] + (state[k] - base[k]) * 20 for k in state}
    if kind == "signflip":
        return {k: base[k] - (state[k] - base[k]) * 3 for k in state}
    if kind == "noise":
        return {k: state[k] + torch.from_numpy(rng.normal(0, 1, tuple(state[k].shape))).to(state[k].dtype)
                for k in state}
    if kind == "nan":
        out = dict(state)
        k = sorted(out)[0]
        out[k] = out[k].clone().float()
        out[k].view(-1)[0] = float("nan")
        return out
    return state


def val_metrics(unit, name, state):
    y = unit.val.labels
    p = unit.predict_neural(name, state, unit.val)
    out = ranking(y, p)
    if out["auroc"] is not None:
        m = at_threshold(y, p, choose_threshold(y, p))
        out.update({"f1": m["f1"], "accuracy": (m["tp"] + m["tn"]) / max(len(y), 1)})
    return out


def federated_training(units, name, rounds, local_epochs=1, pos_weight=1.0, mu=0.0,
                       attack_plan=None, log=print):
    """Train one neural model across the hospital clients. With one client this is plain
    local training. Returns the best global weights (by window-weighted validation AUROC)
    and a full round log."""
    security = SecurityLayer()
    keys = {u.id: security.register(u.id) for u in units}
    orchestrator = OrchestratorAgent({u.id: len(u.train) for u in units})
    attack_plan = attack_plan or {}
    rng = np.random.default_rng(0)

    state = {k: v.detach().cpu() for k, v in M.build(name).state_dict().items()}
    size = n_bytes(state)
    history, best, best_score = [], state, -np.inf
    for r in range(1, rounds + 1):
        updates = []
        for u in units:
            local, n = u.fit_local(name, state, local_epochs, pos_weight, mu, seed=r)
            kind = attack_plan.get(u.id)
            if kind in ("scale", "signflip", "noise", "nan"):
                local = attack(local, state, kind, rng)
            sig = SecurityLayer.sign(keys[u.id], local, n)
            if kind == "tamper":                          # changed after signing, on the way
                local = attack(local, state, "noise", rng)
            updates.append({"client": u.id, "state": local, "n": n, "signature": sig})

        verified, sec_report = [], {}
        for up in updates:
            ok, why = security.verify(up["client"], up["state"], up["n"], up["signature"])
            sec_report[up["client"]] = why
            if ok:
                verified.append(up)
        accepted, orch_report = orchestrator.validate(verified, state)
        report = []
        for up in updates:
            c = up["client"]
            if sec_report[c]:
                report.append({"client": c, "n": int(up["n"]), "status": "rejected",
                               "stage": "security layer", "reasons": [sec_report[c]]})
            else:
                rr = next(x for x in orch_report if x["client"] == c)
                rr["stage"] = "orchestrator agent"
                report.append(rr)

        if accepted:
            state = average([a["state"] for a in accepted], [a["n"] for a in accepted])
        val = {u.id: val_metrics(u, name, state) for u in units}
        ok = [(v["auroc"], len(u.val)) for u, v in zip(units, val.values()) if v["auroc"] is not None]
        score = float(np.average([a for a, _ in ok], weights=[w for _, w in ok])) if ok else float("nan")
        history.append({"round": r, "clients": report, "accepted": len(accepted), "total": len(units),
                        "global_digest": digest(state)[:16], "validation": val,
                        "weighted_val_auroc": score, "bytes_up": size * len(units),
                        "bytes_down": size * len(units)})
        rejected = [x["client"] for x in report if x["status"] == "rejected"]
        log(f"  {name:20s} round {r}/{rounds}  accepted {len(accepted)}/{len(units)}"
            + (f" (rejected {', '.join(rejected)})" if rejected else "")
            + f"  weighted val AUROC {score:.3f}")
        if score > best_score:
            best, best_score = state, score
    scores = [h["weighted_val_auroc"] for h in history]
    best_round = int(np.nanargmax(scores)) + 1 if np.isfinite(scores).any() else rounds
    return best, {"rounds": history, "best_round": best_round, "parameters_bytes": size,
                  "aggregation": "FedProx" if mu > 0 else "FedAvg", "mu": mu}
