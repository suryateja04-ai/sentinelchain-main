import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedsepsis.agents import BLOCKED, ExplainabilityAgent, Orchestrator  # noqa: E402
from fedsepsis.config import FEATURES  # noqa: E402
from fedsepsis.experiment import run  # noqa: E402
from fedsepsis.inference import window_sha256  # noqa: E402
from fedsepsis.unit import HospitalUnit  # noqa: E402
from make_synthetic import write_hospital  # noqa: E402


@pytest.fixture(scope="module")
def unit_folder(tmp_path_factory):
    root = tmp_path_factory.mktemp("agents")
    write_hospital(str(root / "A"), 120, 0.0, 3, "p0")
    write_hospital(str(root / "B"), 120, 0.08, 4, "p1")
    units = [HospitalUnit.from_folder(h, h, str(root / h)) for h in ("A", "B")]
    run(units, ["fedavg"], ["logistic_regression", "gru"], ["lightgbm"], rounds=1,
        artifacts=str(root / "art"), synthetic=True, log=lambda *a: None)
    sep = next(p for p in units[0].test_p if p.y.max() > 0 and len(p.y) >= 16)
    return str(root / "art" / "fedavg" / "hospital_A"), sep


def test_three_agents_complete(unit_folder):
    folder, pat = unit_folder
    t = Orchestrator(folder).assess(pat.x, pat.pid)
    assert t["completed"] and [s["status"] for s in t["stages"]] == ["completed"] * 3
    assert [s["agent"] for s in t["stages"]] == ["Data Gathering and Preprocessing Agent",
                                                 "Sepsis Prediction Agent",
                                                 "Explainability and Clinical Guidance Agent"]
    pred = t["stages"][1]["output"]
    assert 0 <= pred["probability"] <= 1 and pred["alert"] == (pred["probability"] >= pred["threshold"])
    assert set(pred["operating_point"]) >= {"precision", "sensitivity", "specificity", "alerts_per_100"}
    assert abs(sum(m["weight"] for m in pred["models"]) - 1) < 1e-3
    assert t["audit"]["hospital_id"] == "A"
    assert t["audit"]["input_sha256"] == window_sha256(pat.x[-16:])


def test_guidance_never_uses_blocked_words(unit_folder):
    folder, pat = unit_folder
    t = Orchestrator(folder).assess(pat.x, pat.pid)
    text = " ".join(t["stages"][2]["output"]["guidance"]).lower()
    assert not any(w in text for w in BLOCKED)
    with pytest.raises(ValueError):
        ExplainabilityAgent.check_wording(["Start treatment now"])


def test_short_record_stops_after_agent_one(unit_folder):
    folder, pat = unit_folder
    t = Orchestrator(folder).assess(pat.x[:10], pat.pid)
    assert not t["completed"] and len(t["stages"]) == 1 and t["stages"][0]["status"] == "failed"
    assert t["audit"]["probability"] is None


def test_cleaning_flags_and_explanation(unit_folder):
    folder, pat = unit_folder
    rec = pat.x.copy()
    hr = FEATURES.index("HR")
    rec[-1, hr] = 999                               # impossible: must be set to missing
    rec[-2, hr] = 130                               # high but possible: must be flagged
    t = Orchestrator(folder).assess(rec, pat.pid)
    data = t["stages"][0]["output"]
    assert any(c["signal"] == "HR" for c in data["quality"]["cleaned"])
    flags = t["stages"][2]["output"]["flags"]
    assert any(f["signal"] == "HR" and f["direction"] == "high" and f["when"] == "1 h before the latest hour"
               for f in flags)
    factors = t["stages"][2]["output"]["factors"]
    observed = {FEATURES[j] for j in range(39) if not np.isnan(rec[-16:, j]).all()}
    assert all(f["signal"] in observed for f in factors)
