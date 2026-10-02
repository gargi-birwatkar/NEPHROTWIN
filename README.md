# NEPHROTWIN — ML Core & Shared Contracts (BE1)

Machine learning core, feature engineering, and shared interfaces for **NephroTwin** — an in silico digital twin for acute-on-chronic kidney injury (AKI-on-CKD) early warning[cite: 1, 3].

Pre-trained model artifacts are versioned and stored under `ml/artefacts/v1/` so you can run inference immediately without retraining[cite: 1, 2].

---

## 1. Quickstart & Installation

### Setup
```bash
# 1. Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
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

## 2. Running Inference (Using Pre-Trained Artifacts)

You do **not** need to train the model to use it. The repository loads the saved `model.joblib` artifact directly.

### Run Inference & Counterfactual What-If Test

Loads `ml/artefacts/v1/model.joblib` to predict AKI risk for a deteriorating patient (Sunita D.) and tests an intervention (fluid resuscitation + stopping NSAIDs):

```powershell
python -m ml.test_interfernces

```

### Run Contract Adherence Tests

Verifies that feature extraction enforces chronological frame order, prevents temporal leakage, and handles missing sensor streams gracefully:

```powershell
pytest tests/ml/test_contracts.py

```

---

## 3. (Optional) Re-Training the Model

Run this **only** if you want to rebuild or update the weights using new data distributions:

```powershell
python -m ml.run_pipeline

```

This updates:

* `ml/artefacts/v1/model.joblib`

* `ml/artefacts/v1/meta.json`


---

## 4. How BE2 Connects to This (The Handoff)

Backend 2 (BE2) builds the FastAPI routes, physiology engine, and counterfactual simulation sandbox. BE2 does **not** need to touch internal ML code under `ml/models/`.

BE2 loads the pre-trained artifact using the contract in `00_SHARED_INTERFACES.md`:

### Step A: Load the Artifact Once on Startup

In BE2's FastAPI service:

```python
import joblib
from pathlib import Path

# Load pre-trained model bundle into memory (runs once at app startup)
MODEL_PATH = Path("ml/artefacts/v1/model.joblib")
forecaster = joblib.load(MODEL_PATH)

```

### Step B: Feature Extraction & Prediction

When a request hits BE2's endpoints:

```python
from datetime import datetime, timezone
from contracts.interfaces import StaticProfile, TelemetryFrame, TrajectoryPerturbation
from ml.features.api import extract_features

# 1. Convert incoming request data to interfaces
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
telemetry = [TelemetryFrame(timestamp=now, map_mmhg=64.0, hr_bpm=86.0)]

# 2. Extract features (pure function, no leakage)
features = extract_features(static, telemetry, prediction_time=now)

# 3. Base prediction (/forecast endpoint)
forecast = forecaster.predict(features, patient_id=static.patient_id, prediction_time=now)
print(f"Risk 48h: {forecast.risk_48h * 100:.1f}% | Band: {forecast.risk_band}")

# 4. Counterfactual What-If simulation (/simulate endpoint)
perturbation = TrajectoryPerturbation(
    feature_overrides={"nsaid_exposure": 0.0},
    feature_deltas={"map_mean_24h": 10.0}
)
sim_result = forecaster.predict_trajectory(features, perturbation, static.patient_id, now)
print(f"Simulated 48h Risk: {sim_result.risk_48h * 100:.1f}%")

```

---

## 5. Key Project Files

```
nephrotwin/
├── contracts/
│   ├── interfaces.py          # Frozen shared types and Forecaster protocol[cite: 1]
│   └── variables.yaml         # Plausible clinical ranges & risk thresholds[cite: 1]
├── ml/
│   ├── features/api.py        # extract_features implementation[cite: 1, 2]
│   ├── artefacts/v1/          # Pre-trained model.joblib and meta.json[cite: 1, 2]
│   ├── test_interfernces.py   # Test suite for inference and counterfactuals[cite: 1, 2]
│   └── run_pipeline.py        # Optional training script (for rebuilding weights)[cite: 2]
└── tests/ml/
    └── test_contracts.py      # Contract adherence tests[cite: 2]

```
