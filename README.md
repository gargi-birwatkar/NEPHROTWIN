# NEPHROTWIN — ML Core & Shared Contracts (BE1)

Machine learning core, feature engineering, and shared interfaces for **NephroTwin** — an in silico digital twin for acute-on-chronic kidney injury (AKI-on-CKD) early warning.

---

## 1. Quickstart & Installation

### Prerequisites

* Python 3.10+
* Git

### Setup

```bash
# 1. Clone repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd nephrotwin

# 2. Create and activate virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

```

---

## 2. Running the Pipeline

### Train Model & Export Artifacts

Trains the LightGBM classifier with patient-level cross-validation, applies isotonic probability calibration, and exports the model bundle to `ml/artefacts/v1/`:

```powershell
python -m ml.run_pipeline

```

**Outputs produced:**

* `ml/artefacts/v1/model.joblib`: Serialized `MLForecaster` instance.


* `ml/artefacts/v1/meta.json`: Model version, training source, and feature schema.



### Test Inference & What-If Simulation

Runs sample inference for a deteriorating patient profile (Sunita D.) and executes a counterfactual intervention (fluid resuscitation + stopping NSAIDs):

```powershell
python -m ml.test_interfernces

```

### Run Unit & Contract Tests

Verifies that feature extraction enforces chronological frame order, prevents temporal leakage, and handles missing sensor streams gracefully:

```powershell
pytest tests/ml/test_contracts.py

```

---

## 3. How BE2 Connects to This (The Handoff)

Backend 2 (BE2) builds the FastAPI routes, physiology engine, and counterfactual simulation sandbox. BE2 does **not** need to touch internal ML code under `ml/models/`.

BE2 interacts with BE1 solely through two components frozen in `00_SHARED_INTERFACES.md`:

### 1. Extracting Features from Raw Telemetry

BE2 receives patient vitals over the API and converts them into the validated feature vector using `extract_features`:

```python
from datetime import datetime, timezone
from contracts.interfaces import StaticProfile, TelemetryFrame
from ml.features.api import extract_features

# BE2 prepares static record + telemetry window
static = StaticProfile(
    patient_id="PATIENT_001",
    age_years=62.0,
    sex="F",
    baseline_creatinine_mg_dl=1.8,
    egfr_ml_min_1_73m2=34.0,
    uacr_category="A2",
    diabetes=True,
    hypertension=True,
    on_acei_arb=True,
    nsaid_exposure=True
)

now = datetime.now(timezone.utc)
telemetry = [
    TelemetryFrame(timestamp=now, map_mmhg=64.0, hr_bpm=86.0)
]

# extract_features is pure, deterministic, and leakage-free
features = extract_features(static, telemetry, prediction_time=now)

```

### 2. Loading the Model for Predictions & Sandbox Simulations

BE2 loads the exported `model.joblib` bundle and calls `predict` or `predict_trajectory` directly:

```python
import joblib
from contracts.interfaces import TrajectoryPerturbation

# BE2 loads model artifact
forecaster = joblib.load("ml/artefacts/v1/model.joblib")

# 1. Base prediction endpoint (/forecast)
result = forecaster.predict(features, patient_id=static.patient_id, prediction_time=now)
print(f"Risk 48h: {result.risk_48h * 100:.1f}% | Band: {result.risk_band}")

# 2. Counterfactual sandbox endpoint (/simulate)
perturbation = TrajectoryPerturbation(
    feature_overrides={"nsaid_exposure": 0.0},
    feature_deltas={"map_mean_24h": 10.0}
)
sim_result = forecaster.predict_trajectory(features, perturbation, static.patient_id, now)
print(f"Simulated Risk: {sim_result.risk_48h * 100:.1f}%")

```

---

## 4. Key Project Files

```
nephrotwin/
├── contracts/
│   ├── interfaces.py          # Frozen shared types and Forecaster protocol[cite: 1]
│   └── variables.yaml         # Plausible clinical ranges & risk thresholds[cite: 1]
├── ml/
│   ├── features/api.py        # extract_features implementation[cite: 1, 2]
│   ├── artefacts/v1/          # Exported model.joblib and meta.json[cite: 1, 2]
│   ├── run_pipeline.py        # Pipeline training script[cite: 2]
│   └── test_interfernces.py   # Test suite for inference and counterfactuals[cite: 1, 2]
└── tests/ml/
    └── test_contracts.py      # Contract adherence tests[cite: 2]

```
