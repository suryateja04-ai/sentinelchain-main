"""Writes small PhysioNet-format .psv files for testing the code only.

The numbers are random with a built-in sepsis signal; results on them mean nothing clinically.
Two hospitals get different baselines so the data is non-IID, as between real hospitals.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedsepsis.config import FEATURES, LABEL  # noqa: E402

BASE = {"HR": 85, "O2Sat": 97, "Temp": 37, "SBP": 120, "MAP": 82, "DBP": 63, "Resp": 18,
        "WBC": 9, "Lactate": 1.4, "Creatinine": 1.1, "BUN": 18, "FiO2": 0.4, "Platelets": 220}
LABS = set(FEATURES[8:34])


def write_hospital(folder, n, shift, seed, prefix):
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        T = int(rng.integers(12, 60))
        sepsis = rng.random() < 0.25
        onset = int(rng.integers(10, T + 6)) if sepsis else None
        rows = []
        for t in range(T):
            drift = max(0, t - (onset - 12)) / 12 if sepsis else 0.0
            r = {}
            for f in FEATURES:
                b = BASE.get(f, 1.0) * (1 + shift)
                v = b + rng.normal(0, 0.05 * abs(b) + 0.01)
                if f in ("HR", "Resp", "Temp", "WBC", "Lactate"):
                    v += drift * 0.15 * abs(b)
                if f in ("SBP", "MAP"):
                    v -= drift * 0.15 * abs(b)
                r[f] = v
            r["Age"] = 60 + 10 * shift; r["Gender"] = i % 2; r["Unit1"] = 1; r["Unit2"] = 0
            r["HospAdmTime"] = -5.0
            for f in FEATURES:
                if f in LABS and rng.random() < 0.9:
                    r[f] = np.nan
            if rng.random() < 0.01:
                r["FiO2"] = 4000.0                      # impossible value, must be cleaned
            r["ICULOS"] = t + 1
            r[LABEL] = int(sepsis and t >= onset - 6)
            rows.append(r)
        cols = FEATURES + ["ICULOS", LABEL]
        with open(os.path.join(folder, f"{prefix}{i:06d}.psv"), "w") as fh:
            fh.write("|".join(cols) + "\n")
            for r in rows:
                fh.write("|".join("NaN" if (isinstance(r[c], float) and np.isnan(r[c])) else
                                  f"{r[c]:.4g}" for c in cols) + "\n")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    write_hospital(os.path.join(out, "hospital_A"), n, 0.0, 1, "p0")
    write_hospital(os.path.join(out, "hospital_B"), n, 0.08, 2, "p1")
    print("wrote synthetic test hospitals to", out)
