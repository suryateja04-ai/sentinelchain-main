"""Run the three agents for one patient inside one hospital unit.

    python assess.py --unit artifacts/fedprox/hospital_A --patient data/training_setA/p000009.psv
"""
import argparse
import json

from fedsepsis.agents import Orchestrator
from fedsepsis.data import read_patient


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", required=True, help="the hospital unit's artifact folder")
    ap.add_argument("--patient", required=True, help="the patient's .psv file")
    ap.add_argument("--end-hour", type=int, default=None, help="last ICU hour of the window (1-based)")
    a = ap.parse_args()
    pat = read_patient(a.patient)
    end = None if a.end_hour is None else a.end_hour - 1
    trace = Orchestrator(a.unit).assess(pat.x, pat.pid, end)
    print(json.dumps(trace, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
