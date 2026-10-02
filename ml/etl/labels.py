from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pydantic import BaseModel


class LeakageReport(BaseModel):
    is_leakage_free: bool
    violations_count: int
    summary: str


def label_aki_kdigo(creatinine_df: pd.DataFrame, baseline_series: pd.Series) -> pd.DataFrame:
    """
    Evaluates KDIGO Stage 1 criteria:
    - Rise in SCr >= 0.3 mg/dL within a 48-hour rolling window, OR
    - Rise in SCr >= 1.5x baseline within a 7-day rolling window.
    """
    df = creatinine_df.sort_values(["patient_id", "timestamp"]).copy()
    df["baseline"] = df["patient_id"].map(baseline_series)

    # 48-hour rolling minimum creatinine per patient
    df["rolling_min_48h"] = (
        df.groupby("patient_id")
        .rolling("48h", on="timestamp")["creatinine_mg_dl"]
        .min()
        .reset_index(drop=True)
    )

    abs_increase = (df["creatinine_mg_dl"] - df["rolling_min_48h"]) >= 0.3
    rel_increase = (df["creatinine_mg_dl"] / df["baseline"]) >= 1.5

    df["is_aki"] = abs_increase | rel_increase
    return df


def build_prediction_grid(
    cohort_df: pd.DataFrame,
    start_time: datetime,
    end_time: datetime,
    step_hours: int = 6,
) -> pd.DataFrame:
    """
    Constructs a discrete grid of (patient_id, prediction_time) stepping every step_hours.
    """
    grid_rows = []
    current_time = start_time
    time_points = []
    while current_time <= end_time:
        time_points.append(current_time)
        current_time += timedelta(hours=step_hours)

    for pid in cohort_df["patient_id"].unique():
        for t in time_points:
            grid_rows.append({"patient_id": pid, "prediction_time": t})

    return pd.DataFrame(grid_rows)


def assign_horizon_labels(
    grid_df: pd.DataFrame,
    aki_events_df: pd.DataFrame,
    horizons_hours: tuple[int, int] = (24, 48),
) -> pd.DataFrame:
    """
    Assigns binary labels y_24h and y_48h for whether an AKI onset falls in the future window.
    Excludes windows where AKI has already occurred at or before prediction_time.
    """
    labeled_rows = []
    aki_events = aki_events_df[aki_events_df["is_aki"]].copy()

    for _, row in grid_df.iterrows():
        pid = row["patient_id"]
        t_pred = row["prediction_time"]

        patient_events = aki_events[aki_events["patient_id"] == pid]
        
        # Exclude if patient was already decompensated
        prior_events = patient_events[patient_events["timestamp"] <= t_pred]
        if not prior_events.empty:
            continue

        future_events = patient_events[patient_events["timestamp"] > t_pred]

        t_24 = t_pred + timedelta(hours=horizons_hours[0])
        t_48 = t_pred + timedelta(hours=horizons_hours[1])

        y_24 = int(any((patient_events["timestamp"] > t_pred) & (patient_events["timestamp"] <= t_24)))
        y_48 = int(any((patient_events["timestamp"] > t_pred) & (patient_events["timestamp"] <= t_48)))

        labeled_rows.append({
            "patient_id": pid,
            "prediction_time": t_pred,
            "y_24h": y_24,
            "y_48h": y_48,
        })

    return pd.DataFrame(labeled_rows)


def check_leakage(feature_timestamps: pd.Series, prediction_times: pd.Series) -> LeakageReport:
    """
    Verifies that no feature was computed with data beyond prediction_time.
    """
    violations = (feature_timestamps > prediction_times).sum()
    return LeakageReport(
        is_leakage_free=bool(violations == 0),
        violations_count=int(violations),
        summary="Passed without leakage" if violations == 0 else f"Failed: {violations} timestamp leaks detected."
    )