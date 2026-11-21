import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedsepsis.config import FEATURES, WINDOW  # noqa: E402
from fedsepsis.data import WindowSet, load_hospital, split_patients  # noqa: E402
from fedsepsis.experiment import run  # noqa: E402
from fedsepsis.fedstats import federated_preprocessor  # noqa: E402
from fedsepsis.inference import HospitalPredictor  # noqa: E402
from fedsepsis.metrics import at_threshold, choose_threshold  # noqa: E402
from fedsepsis.orchestrator import OrchestratorAgent  # noqa: E402
from fedsepsis.security import SecurityLayer  # noqa: E402
from fedsepsis.server import average  # noqa: E402
from fedsepsis.unit import HospitalUnit  # noqa: E402
from make_synthetic import write_hospital  # noqa: E402


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    write_hospital(str(root / "A"), 150, 0.0, 1, "p0")
    write_hospital(str(root / "B"), 150, 0.08, 2, "p1")
    return root


def test_cleaning_and_columns(data):
    pats = load_hospital(str(data / "A"))
    fio2 = FEATURES.index("FiO2")
    assert all(np.nanmax(np.r_[p.x[:, fio2], 0]) <= 1000 for p in pats)
    assert pats[0].x.shape[1] == 39


def test_patient_split_is_disjoint_and_repeatable(data):
    pats = load_hospital(str(data / "A"))
    a = split_patients(pats)
    b = split_patients(pats)
    ids = [set(p.pid for p in s) for s in a]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(map(len, ids)) == len(pats)
    assert [p.pid for p in a[2]] == [p.pid for p in b[2]]


def test_windows_and_labels(data):
    pats = load_hospital(str(data / "A"))
    w = WindowSet(pats)
    assert len(w) == sum(max(0, len(p.y) - WINDOW + 1) for p in pats)
    p, end = w.index[7]
    assert w.labels[7] == pats[p].y[end]
    prep, _ = federated_preprocessor([w])
    w.prepare(prep)
    np.testing.assert_allclose(w.batch(np.array([7]))[0], prep.transform(w.raw(7)[None])[0], rtol=1e-6)


def test_federated_statistics_match_pooled(data):
    wa = WindowSet(load_hospital(str(data / "A")))
    wb = WindowSet(load_hospital(str(data / "B")))
    fed, log = federated_preprocessor([wa, wb])
    assert log["patient_rows_shared"] == 0
    vals, wts = [], []
    for w in (wa, wb):
        for x, c in w.row_weights():
            vals.append(x); wts.append(c)
    x, c = np.concatenate(vals), np.concatenate(wts)
    j = FEATURES.index("HR")
    ok = ~np.isnan(x[:, j]) & (c > 0)
    order = np.argsort(x[ok, j])
    cum = np.cumsum(c[ok][order])
    med = x[ok, j][order][np.searchsorted(cum, cum[-1] / 2)]
    assert abs(fed.median[j] - med) < 0.5
    filled = np.where(np.isnan(x[:, j]), fed.median[j], x[:, j])
    mean = np.average(filled, weights=c)
    std = np.sqrt(np.average((filled - mean) ** 2, weights=c))
    assert abs(fed.mean[j] - mean) < 1e-3 * abs(mean) and abs(fed.std[j] - std) < 1e-3 * std


def test_threshold_rule_keeps_recall():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(float)
    p = np.clip(y * 0.3 + rng.random(5000) * 0.5, 0, 1)
    thr = choose_threshold(y, p)
    assert at_threshold(y, p, thr)["sensitivity"] >= 0.5


def test_weighted_average():
    s1 = {"w": torch.tensor([0.0, 2.0])}
    s2 = {"w": torch.tensor([4.0, 2.0])}
    assert torch.allclose(average([s1, s2], [1, 3])["w"], torch.tensor([3.0, 2.0]))


def test_security_layer_detects_tampering():
    sec = SecurityLayer()
    key = sec.register("A")
    st = {"w": torch.ones(3)}
    sig = SecurityLayer.sign(key, st, 10)
    assert sec.verify("A", st, 10, sig)[0]
    assert not sec.verify("A", {"w": torch.ones(3) * 1.01}, 10, sig)[0]
    assert not sec.verify("A", st, 11, sig)[0]
    assert not sec.verify("Z", st, 10, sig)[0]


def test_orchestrator_rejects_bad_updates():
    g = {"w": torch.zeros(4)}
    good = lambda s: {"w": torch.tensor([1.0, 1.0, 0.9, 1.1]) * s}
    ups = [{"client": "A", "state": good(1.0), "n": 10}, {"client": "B", "state": good(1.1), "n": 10},
           {"client": "C", "state": good(0.9), "n": 10},
           {"client": "D", "state": good(30.0), "n": 10},                     # far too large
           {"client": "E", "state": good(-1.0), "n": 10},                     # opposite direction
           {"client": "F", "state": {"w": torch.tensor([float("nan"), 0, 0, 0])}, "n": 10},
           {"client": "G", "state": good(1.0), "n": 999}]                    # more than registered
    orch = OrchestratorAgent({c: 10 for c in "ABCDEFG"})
    acc, rep = orch.validate(ups, g)
    assert sorted(a["client"] for a in acc) == ["A", "B", "C"]
    assert {r["client"] for r in rep if r["status"] == "rejected"} == {"D", "E", "F", "G"}


def test_end_to_end_all_schemes(data, tmp_path):
    units = [HospitalUnit.from_folder("A", "Hospital A", str(data / "A")),
             HospitalUnit.from_folder("B", "Hospital B", str(data / "B"))]
    res = run(units, ["local", "fedavg", "fedprox", "centralized"], ["gru", "logistic_regression"],
              ["lightgbm"], rounds=2, artifacts=str(tmp_path / "art"), synthetic=True, log=lambda *a: None)
    assert set(res["schemes"]) == {"local", "fedavg", "fedprox", "centralized"}
    for scheme in ("fedavg", "fedprox"):
        fed = res["schemes"][scheme]
        assert set(fed) == {"A", "B"}
        for hid in ("A", "B"):
            assert abs(sum(fed[hid]["ensemble_weights"].values()) - 1) < 1e-3
            assert fed[hid]["test"][hid]["auroc"] > 0.6         # the synthetic signal is learnable
        hist = res["federation"][scheme]["history"]["gru"]["rounds"]
        assert len(hist) == 2 and hist[0]["accepted"] == 2
    pred = HospitalPredictor(str(tmp_path / "art" / "fedprox" / "hospital_B"))
    out = pred.predict(units[1].test.raw(0))
    assert 0 <= out["probability"] <= 1
    assert out["audit"]["hospital_id"] == "B" and len(out["audit"]["input_sha256"]) == 64


def test_attacked_client_is_left_out(data):
    units = [HospitalUnit.from_folder(h, h, str(data / h)) for h in ("A", "B")]
    units.append(HospitalUnit("C", "C", units[0].train_p + units[0].val_p + units[0].test_p))
    res = run(units, ["fedavg"], ["logistic_regression"], [], rounds=2, synthetic=True,
              attack_plan={"C": "scale"}, log=lambda *a: None)
    for r in res["federation"]["fedavg"]["history"]["logistic_regression"]["rounds"]:
        status = {c["client"]: c["status"] for c in r["clients"]}
        assert status == {"A": "accepted", "B": "accepted", "C": "rejected"}
