import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
from datetime import datetime, timezone, timedelta
from contracts.interfaces import StaticProfile, TelemetryFrame, TrajectoryPerturbation
from ml.features.api import extract_features

# 1. Load the frozen model bundle
forecaster = joblib.load("ml/artefacts/v1/model.joblib")

# 2. Patient Profile: Severe CKD Stage 3b + NSAID exposure + Diabetes
patient_id = "PATIENT_SUNITA_D"
now = datetime(2026, 10, 2, 20, 0, tzinfo=timezone.utc)
patient = StaticProfile(
    patient_id=patient_id,
    age_years=63.0,
    sex="F",
    baseline_creatinine_mg_dl=2.1,
    egfr_ml_min_1_73m2=28.0,
    uacr_category="A3",
    diabetes=True,
    hypertension=True,
    on_acei_arb=True,
    nsaid_exposure=True,
)

# 3. Deteriorating Telemetry: Sustained MAP depression (mean ~62 mmHg), tachycardia (88 bpm), low HRV (12 ms)
telemetry = []
for i in range(24):
    t = now - timedelta(hours=24 - i)
    is_night = (t.hour >= 22 or t.hour < 6)
    
    # Non-dipping or reverse dipping pattern
    map_val = 61.0 if is_night else 63.0
    telemetry.append(TelemetryFrame(
        timestamp=t,
        map_mmhg=map_val,
        hr_bpm=92.0,
        hrv_rmssd_ms=12.0,
    ))

# 4. Extract features & predict baseline
features = extract_features(patient, telemetry, now)
forecast = forecaster.predict(features, patient.patient_id, now)

# 5. Display Results
print("\n" + "=" * 65)
print(f"TRAINED ON DATASET  : {forecaster.meta.trained_on}")
print(f"INFERENCE PATIENT   : {forecast.patient_id}")
print(f"MODEL VERSION       : {forecast.model_version} (Stub: {forecaster.meta.is_stub})")
print("-" * 65)
print(f"Baseline 24h Risk   : {forecast.risk_24h * 100:.1f}% [{forecast.risk_24h_interval[0]*100:.1f}% - {forecast.risk_24h_interval[1]*100:.1f}%]")
print(f"Baseline 48h Risk   : {forecast.risk_48h * 100:.1f}% [{forecast.risk_48h_interval[0]*100:.1f}% - {forecast.risk_48h_interval[1]*100:.1f}%]")
print(f"Risk Band           : {forecast.risk_band.upper()}")
print(f"Low Confidence      : {forecast.low_confidence}")
print("\nTop Risk Drivers (TreeSHAP):")
for d in forecast.drivers:
    sym = "(+)" if d.direction == "increases_risk" else "(-)"
    print(f"  {sym} {d.display_name}: {d.rationale}")

# 6. Counterfactual Sandbox Intervention:
# Stop NSAID, eliminate hypotension, restore MAP to 78 mmHg, reduce HR to 74, recover HRV to 28
perturbation = TrajectoryPerturbation(
    feature_overrides={
        "nsaid_exposure": 0.0,
        "hypotension_burden_min": 0.0,
        "nocturnal_dip_pct": 12.0,
        "map_min_24h": 72.0,
    },
    feature_deltas={
        "map_mean_24h": 15.0,
        "hr_mean_24h": -18.0,
        "hrv_rmssd_mean_24h": 16.0,
    },
)
sim_result = forecaster.predict_trajectory(features, perturbation, patient.patient_id, now)

print("\n" + "-" * 65)
print("COUNTERFACTUAL SIMULATION (Sandbox Mode - In Silico Hypothesis):")
print(f"Revised 24h Risk    : {sim_result.risk_24h * 100:.1f}% [{sim_result.risk_24h_interval[0]*100:.1f}% - {sim_result.risk_24h_interval[1]*100:.1f}%]")
print(f"Revised 48h Risk    : {sim_result.risk_48h * 100:.1f}% [{sim_result.risk_48h_interval[0]*100:.1f}% - {sim_result.risk_48h_interval[1]*100:.1f}%] (Band: {sim_result.risk_band.upper()})")
print("=" * 65)