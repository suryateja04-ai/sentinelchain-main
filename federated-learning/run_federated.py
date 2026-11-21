"""Federated training across hospital units.

Example (PhysioNet 2019, set A = Hospital A, set B = Hospital B):

    python run_federated.py \
        --hospital A "Hospital A (PhysioNet set A)" data/training_setA \
        --hospital B "Hospital B (PhysioNet set B)" data/training_setB \
        --rounds 5 --artifacts artifacts --out ../federation/results.json
"""
import argparse
import time

from fedsepsis.data import load_hospital, split_by_icu
from fedsepsis.experiment import run, to_json
from fedsepsis.models import NEURAL
from fedsepsis.unit import TREES, HospitalUnit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hospital", nargs=3, action="append", metavar=("ID", "NAME", "FOLDER"), required=True,
                    help="one hospital unit: short id, display name, folder of .psv files")
    ap.add_argument("--schemes", nargs="+", default=["local", "fedavg", "fedprox", "centralized"],
                    choices=["local", "fedavg", "fedprox", "centralized"])
    ap.add_argument("--neural", nargs="*", default=list(NEURAL), choices=list(NEURAL))
    ap.add_argument("--trees", nargs="*", default=list(TREES), choices=list(TREES))
    ap.add_argument("--rounds", type=int, default=5, help="federated rounds (local: epochs)")
    ap.add_argument("--local-epochs", type=int, default=1, help="epochs per hospital per round")
    ap.add_argument("--fedprox-mu", type=float, default=0.01, help="proximal term for the fedprox scheme")
    ap.add_argument("--split-by-icu", action="store_true",
                    help="turn each hospital into one client per ICU type (MICU, SICU, Unknown)")
    ap.add_argument("--attack", nargs=2, action="append", metavar=("CLIENT", "KIND"), default=[],
                    help="simulate a faulty client: KIND is scale, signflip, noise, nan or tamper")
    ap.add_argument("--tree-sample", type=int, default=None, help="max training windows for tree models")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N patients per hospital")
    ap.add_argument("--artifacts", default=None, help="folder for each hospital's deployed models")
    ap.add_argument("--out", default="results/federated_results.json")
    ap.add_argument("--synthetic", action="store_true", help="mark results as synthetic test data")
    a = ap.parse_args()

    t0 = time.time()
    units = []
    for hid, name, folder in a.hospital:
        print(f"loading hospital {hid} from {folder}")
        patients = load_hospital(folder, a.limit)
        groups = split_by_icu(patients) if a.split_by_icu else {None: patients}
        for icu, group in sorted(groups.items(), key=lambda g: str(g[0])):
            cid = f"{hid}-{icu}" if icu else hid
            units.append(HospitalUnit(cid, f"{name} {icu}" if icu else name, group))
            s = units[-1].summary()
            print(f"  client {cid}: train {s['train']['windows']:,} windows, "
                  f"val {s['validation']['windows']:,}, test {s['test']['windows']:,}")
    res = run(units, a.schemes, a.neural, a.trees, a.rounds, a.local_epochs, a.fedprox_mu,
              a.tree_sample, a.artifacts, a.synthetic, dict(a.attack))
    res["runtime_seconds"] = round(time.time() - t0, 1)
    to_json(res, a.out)
    print(f"done in {res['runtime_seconds']} s -> {a.out}")
    for scheme, sites in res["schemes"].items():
        for site, r in sites.items():
            for hid, m in r["test"].items():
                print(f"{scheme:12s} model of {site:3s} on test of {hid}: AUROC {m['auroc']:.3f}  "
                      f"AUPRC {m['auprc']:.3f}  sens {m['sensitivity']:.1%}  spec {m['specificity']:.1%}")


if __name__ == "__main__":
    main()
