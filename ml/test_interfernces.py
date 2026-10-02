import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
from datetime import datetime, timezone, timedelta
from contracts.interfaces import StaticProfile, TelemetryFrame, TrajectoryPerturbation
from ml.features.api import extract_features

# 1. Load the frozen model bundle
forecaster = joblib.load("ml/artefacts/v1/model.joblib")
print(f"Loaded model version: {forecaster.meta.model_version} (Stub: {forecaster.meta.is_stub})")

# 2. Define a high-risk decompensating patient profile (Sunita D.)
now = datetime(2026, 10, 2, 20, 0, tzinfo=timezone.utc)
patient = StaticProfile(
    patient_id="PATIENT_SUNITA_D",
    age_years=63.0,
    sex="F",
    baseline_creatinine_mg_dl=1.9,
    egfr_ml_min_1_73m2=32.0,
    uacr_category="A3",
    diabetes=True,
    hypertension=True,
    on_acei_arb=True,
    nsaid_exposure=True,
)

# 3. Simulate 24 hours of deteriorating vitals (hypotension, loss of nocturnal dipping)
telemetry = []
for i in range(24):
    t = now - timedelta(hours=24 - i)
    is_night = (t.hour >= 22 or t.hour < 6)
    # Lost nocturnal dip: night MAP stays low / blunted around 62 mmHg
    map_val = 62.0 if is_night else 66.0
    telemetry.append(TelemetryFrame(
        timestamp=t,
        map_mmhg=map_val,
        hr_bpm=88.0,
        hrv_rmssd_ms=14.0,
    ))

# 4. Extract features & predict
features = extract_features(patient, telemetry, now)
forecast = forecaster.predict(features, patient.patient_id, now)

# 5. Display baseline forecast
print("\n" + "=" * 55)
print(f"Baseline Forecast for: {forecast.patient_id}")
print(f"Risk 24h: {forecast.risk_24h * 100:.1f}% [{forecast.risk_24h_interval[0]*100:.1f}% - {forecast.risk_24h_interval[1]*100:.1f}%]")
print(f"Risk 48h: {forecast.risk_48h * 100:.1f}% [{forecast.risk_48h_interval[0]*100:.1f}% - {forecast.risk_48h_interval[1]*100:.1f}%]")
print(f"Risk Band: {forecast.risk_band.upper()}")
print(f"Low Confidence: {forecast.low_confidence}")
print("\nTop Risk Drivers (TreeSHAP):")
for d in forecast.drivers:
    sym = "(+)" if d.direction == "increases_risk" else "(-)"
    print(f"  {sym} {d.display_name}: {d.rationale}")

# 6. Test counterfactual simulation (What-If: Stop NSAID + restore MAP by +8 mmHg)
perturbation = TrajectoryPerturbation(
    feature_overrides={"nsaid_exposure": 0.0, "hypotension_burden_min": 0.0, "nocturnal_dip_pct": 12.0},
    feature_deltas={"map_mean_24h": 12.0},
)
sim_result = forecaster.predict_trajectory(features, perturbation, patient.patient_id, now)

print("\n" + "-" * 55)
print("Counterfactual Simulation (Stop NSAID + Fluid Resuscitation + BP Restoration):")
print(f"Revised 24h Risk: {sim_result.risk_24h * 100:.1f}%")
print(f"Revised 48h Risk: {sim_result.risk_48h * 100:.1f}% (Band: {sim_result.risk_band.upper()})")
print("=" * 55)