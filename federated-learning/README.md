# Federated hospital units

Privacy-preserving federated learning for the sepsis early-warning system. Every hospital has
its own unit that keeps its patient data, trains locally and runs predictions with models
personalised on its own patients. Hospitals improve a shared model through a central server:

    hospital units --signed updates--> Security Layer -> Orchestrator Agent
                                       -> Federated Aggregation (FedAvg / FedProx) -> Global Model
    Global Model --universal update--> every hospital unit, next round

The design is described in `docs/E6_Federated_Unit_Report.pdf`.

## Layout

| File | Role |
|---|---|
| `fedsepsis/config.py` | The 39 PhysioNet variables, window, split and threshold settings |
| `fedsepsis/data.py` | Loading a hospital's `.psv` files, cleaning, patient split, 16-hour windows, ICU-type clients |
| `fedsepsis/fedstats.py` | Federated median imputer and scaler from histograms and sums |
| `fedsepsis/models.py` | Logistic Regression, LSTM, GRU and Temporal CNN in PyTorch |
| `fedsepsis/unit.py` | `HospitalUnit`: local training, tree models, personalisation |
| `fedsepsis/security.py` | Security Layer: HMAC-SHA256 signing and verification |
| `fedsepsis/orchestrator.py` | Orchestrator Agent: validates and filters updates |
| `fedsepsis/server.py` | Central server rounds, aggregation, attack simulation |
| `fedsepsis/experiment.py` | Local, FedAvg, FedProx and centralised schemes; artifacts; results JSON |
| `fedsepsis/inference.py` | `HospitalPredictor`: loads a unit's models and scores windows |
| `fedsepsis/agents.py` | The three agents and the orchestrator that runs them and writes the audit record |
| `run_federated.py` | Command line for training |
| `assess.py` | Command line for one assessment through the three agents |
| `tests/` | Synthetic test data generator and tests |

## Three agents per hospital unit

1. **Data Gathering and Preprocessing Agent**: gathers the latest 16-hour window, fingerprints
   it (SHA-256), validates it, sets impossible values to missing, imputes and scales it, and
   measures the trend of key vitals.
2. **Sepsis Prediction Agent**: scores with every model, calibrates, combines with the
   hospital's ensemble weights, decides the alert with the hospital's threshold, and reports
   the hospital's precision, sensitivity, specificity and alerts per 100 windows.
3. **Explainability and Clinical Guidance Agent**: occlusion explanation, reference-range
   flags, fixed decision-support guidance (diagnosis, treatment and medication wording is
   blocked) and the clinician report.

The orchestrator runs them in order, stops if the data is unusable, and writes the audit record.

## Run

Put the PhysioNet 2019 folders `training_setA` and `training_setB` under `data/`, then:

    pip install -r requirements.txt
    python run_federated.py \
      --hospital A "Hospital A (PhysioNet set A)" data/training_setA \
      --hospital B "Hospital B (PhysioNet set B)" data/training_setB \
      --rounds 5 --artifacts artifacts --out results/federated_results.json

Options:

- `--split-by-icu` turns each hospital into one client per ICU type (non-IID clients).
- `--attack B scale` makes client B send a faulty update (`scale`, `signflip`, `noise`, `nan`,
  `tamper`) to show the Security Layer and Orchestrator Agent rejecting it.
- `--schemes fedavg fedprox` runs only the federated schemes; `--neural gru lstm` and
  `--trees lightgbm` choose models; `--limit 2000` uses the first 2,000 patients per
  hospital for a quick trial.

A GPU is used automatically when PyTorch finds one.

One assessment inside a hospital unit, printed as a JSON trace:

    python assess.py --unit artifacts/fedprox/hospital_A --patient data/training_setA/p000009.psv

## Tests

    python tests/make_synthetic.py           # optional: writes synthetic hospitals
    python -m pytest -q tests

The synthetic files contain random values with a built-in signal. They only check that the
code works; results on them say nothing about sepsis.
