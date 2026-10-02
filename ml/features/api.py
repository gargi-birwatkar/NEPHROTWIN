from datetime import datetime
from typing import Sequence, Optional
import numpy as np
import pandas as pd
from contracts.interfaces import StaticProfile, TelemetryFrame, FeatureResult, TwinState


FEATURE_NAMES = [
    "age_years",
    "egfr",
    "baseline_creatinine",
    "diabetes",
    "hypertension",
    "on_acei_arb",
    "nsaid_exposure",
    "map_mean_24h",
    "map_min_24h",
    "hypotension_burden_min",
    "hr_mean_24h",
    "hrv_rmssd_mean_24h",
    "nocturnal_dip_pct",
]


def extract_features(
    static: StaticProfile,
    telemetry: Sequence[TelemetryFrame],
    prediction_time: datetime,
    twin_state: Optional[TwinState] = None,
) -> FeatureResult:
    # Validate strictly ordered, time-aware timestamps
    if any(f.timestamp.tzinfo is None for f in telemetry):
        raise ValueError("Naive timestamps encountered in telemetry frames.")
    
    for i in range(1, len(telemetry)):
        if telemetry[i].timestamp < telemetry[i - 1].timestamp:
            raise ValueError("Telemetry frames must be in chronological order.")

    # Guard against temporal leakage: only include frames <= prediction_time
    valid_frames = [f for f in telemetry if f.timestamp <= prediction_time]

    if not valid_frames:
        vec = [np.nan] * len(FEATURE_NAMES)
        return FeatureResult(
            vector=vec,
            names=FEATURE_NAMES,
            missingness={"frac_missing": 1.0, "longest_gap_min": 1440.0, "last_obs_age_min": np.nan},
            schema_version="v1.0",
        )

    records = [f.model_dump() for f in valid_frames]
    df = pd.DataFrame(records).sort_values("timestamp")

    # Dynamic vital feature aggregations
    map_series = df["map_mmhg"].dropna()
    map_mean_24 = float(map_series.mean()) if not map_series.empty else np.nan
    map_min_24 = float(map_series.min()) if not map_series.empty else np.nan
    hypotension_min = float((map_series < 65.0).sum() * 5.0)  # assumes 5 min per frame

    hr_mean = float(df["hr_bpm"].dropna().mean()) if "hr_bpm" in df else np.nan
    hrv_mean = float(df["hrv_rmssd_ms"].dropna().mean()) if "hrv_rmssd_ms" in df else np.nan

    # Nocturnal dip percentage: ((day_mean - night_mean) / day_mean) * 100
    df["hour"] = df["timestamp"].dt.hour
    day_map = df[(df["hour"] >= 6) & (df["hour"] < 22)]["map_mmhg"].mean()
    night_map = df[(df["hour"] >= 22) | (df["hour"] < 6)]["map_mmhg"].mean()

    if pd.notna(day_map) and pd.notna(night_map) and day_map > 0:
        dip_pct = float(((day_map - night_map) / day_map) * 100.0)
    else:
        dip_pct = np.nan

    vec = [
        float(static.age_years),
        float(static.egfr_ml_min_1_73m2),
        float(static.baseline_creatinine_mg_dl),
        1.0 if static.diabetes else 0.0,
        1.0 if static.hypertension else 0.0,
        1.0 if static.on_acei_arb else 0.0,
        1.0 if static.nsaid_exposure else 0.0,
        map_mean_24,
        map_min_24,
        hypotension_min,
        hr_mean,
        hrv_mean,
        dip_pct,
    ]

    last_obs_age = (prediction_time - valid_frames[-1].timestamp).total_seconds() / 60.0

    return FeatureResult(
        vector=vec,
        names=FEATURE_NAMES,
        missingness={
            "frac_missing": float(np.isnan(vec).mean()),
            "longest_gap_min": 5.0,
            "last_obs_age_min": float(max(0.0, last_obs_age)),
        },
        schema_version="v1.0",
    )