"""Constants shared by every hospital unit.

The values follow the Phase-I pipeline: 39 PhysioNet variables, 16-hour windows,
patient-level 80/20 then 80/20 split with seed 42, FiO2 > 1000 treated as missing,
and the threshold rule "best F1 with recall >= 0.50".
"""

FEATURES = [
    # 8 vital signs
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    # 26 laboratory values
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
    # 5 demographic / admission values
    "Age", "Gender", "Unit1", "Unit2", "HospAdmTime",
]
LABEL = "SepsisLabel"
N_FEATURES = len(FEATURES)          # 39
WINDOW = 16                         # hours per window
FLAT = WINDOW * N_FEATURES          # 624 values for the flattened models
SEED = 42
SPLIT = (0.8, 0.8)                  # 80/20 test split, then 80/20 validation split
FIO2_MAX = 1000.0                   # recording errors above this are set to missing
LABEL_LEAD_HOURS = 6                # PhysioNet label turns 1 six hours before onset
MIN_RECALL = 0.50                   # threshold rule: best F1 with recall >= 0.50
HIST_BINS = 4096                    # bins for the federated median
assert N_FEATURES == 39 and FLAT == 624
