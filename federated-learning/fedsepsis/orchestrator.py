"""Orchestrator Agent: validates and filters hospital updates before aggregation.

Checks, in order, for each update that passed the security layer:
  1. every value is finite (no NaN or infinity),
  2. the sample count is positive and not larger than the hospital registered,
  3. the size of the change from the global model is not far larger than the other
     hospitals' changes (norm > NORM_FACTOR x median of the others),
  4. the change does not point against the other hospitals' changes
     (cosine similarity with their mean change < MIN_COSINE).
Rejected updates are left out of this round and logged with the reason.
"""
from __future__ import annotations

import numpy as np
import torch

NORM_FACTOR = 3.0
MIN_COSINE = -0.1


def flat_delta(state, base):
    return torch.cat([(state[k].float() - base[k].float()).flatten() for k in sorted(base)])


class OrchestratorAgent:
    def __init__(self, registered_counts: dict):
        self.registered = registered_counts

    def validate(self, updates: list[dict], global_state) -> tuple[list[dict], list[dict]]:
        """updates: dicts with client, state, n. Returns (accepted, report)."""
        report, candidates = [], []
        for u in updates:
            r = {"client": u["client"], "n": int(u["n"]), "status": "accepted", "reasons": []}
            if not all(torch.isfinite(v).all() for v in u["state"].values()):
                r["reasons"].append("non-finite values")
            if not 0 < u["n"] <= self.registered.get(u["client"], 0):
                r["reasons"].append("sample count not plausible")
            report.append(r)
            if not r["reasons"]:
                candidates.append((u, r))

        deltas = {u["client"]: flat_delta(u["state"], global_state) for u, _ in candidates}
        norms = {c: float(d.norm()) for c, d in deltas.items()}
        for u, r in candidates:
            c = u["client"]
            r["update_norm"] = round(norms[c], 6)
            others = [k for k in deltas if k != c]
            if not others:
                continue
            ref = float(np.median([norms[k] for k in others]))
            if ref > 0 and norms[c] > NORM_FACTOR * ref:
                r["reasons"].append(f"update {norms[c] / ref:.1f}x larger than other hospitals")
            mean_other = torch.stack([deltas[k] for k in others]).mean(0)
            denom = float(deltas[c].norm() * mean_other.norm())
            cos = float(deltas[c] @ mean_other) / denom if denom > 0 else 1.0
            r["cosine_to_others"] = round(cos, 4)
            if cos < MIN_COSINE:
                r["reasons"].append("update points against the other hospitals")

        accepted = []
        for u, r in candidates:
            if r["reasons"]:
                continue
            accepted.append(u)
        for r in report:
            if r["reasons"]:
                r["status"] = "rejected"
        return accepted, report
