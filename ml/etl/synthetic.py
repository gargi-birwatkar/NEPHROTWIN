import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

def generate_synthetic_cohort(n_patients: int = 400, days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generates synthetic static and telemetry records with multi-factorial clinical risk."""
    np.random.seed(42)
    start_time = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
    
    static_records = []
    telemetry_records = []

    for i in range(n_patients):
        pid = f"PATIENT_{i:04d}"
        age = float(np.random.normal(60, 10))
        sex = np.random.choice(["M", "F"])
        diabetes = bool(np.random.rand() < 0.40)
        hypertension = bool(np.random.rand() < 0.55)
        acei_arb = bool(diabetes and (np.random.rand() < 0.65))
        nsaid = bool(np.random.rand() < 0.30)
        
        baseline_creat = float(np.clip(np.random.normal(1.4, 0.5), 0.6, 3.8))
        egfr = float(np.clip(135.0 - (age * 0.7) - (baseline_creat * 22), 12.0, 95.0))
        
        # Clinical risk score determines decompensation probability
        risk_score = 0.0
        if egfr < 45.0: risk_score += 1.5
        if baseline_creat > 1.5: risk_score += 1.2
        if nsaid: risk_score += 1.0
        if acei_arb and nsaid: risk_score += 1.5  # "Double/Triple Whammy"
        if diabetes and hypertension: risk_score += 0.8
        
        prob_decomp = 1.0 / (1.0 + np.exp(-(risk_score - 2.5)))
        decompensates = bool(np.random.rand() < prob_decomp)

        static_records.append({
            "patient_id": pid, "age_years": age, "sex": sex,
            "baseline_creatinine_mg_dl": baseline_creat,
            "egfr_ml_min_1_73m2": egfr, "uacr_category": np.random.choice(["A1", "A2", "A3"]),
            "diabetes": diabetes, "hypertension": hypertension,
            "on_acei_arb": acei_arb, "nsaid_exposure": nsaid, "decompensates": decompensates
        })

        # Base hemodynamics
        mean_map = 82.0 - (12.0 if decompensates else 0.0)
        mean_hr = 72.0 + (14.0 if decompensates else 0.0)

        for h in range(days * 24):
            t = start_time + timedelta(hours=h)
            is_night = (t.hour >= 22 or t.hour < 6)
            
            # Non-dipping or reverse dipping if decompensating
            dip = 0.0 if decompensates else (10.0 if is_night else 0.0)
            map_val = np.random.normal(mean_map - dip, 4.0)
            hr_val = np.random.normal(mean_hr, 5.0)
            hrv_val = np.random.normal(16.0 if decompensates else 34.0, 4.0)

            telemetry_records.append({
                "patient_id": pid, "timestamp": t,
                "map_mmhg": float(map_val), "hr_bpm": float(hr_val),
                "hrv_rmssd_ms": float(max(4.0, hrv_val))
            })

    return pd.DataFrame(static_records), pd.DataFrame(telemetry_records)