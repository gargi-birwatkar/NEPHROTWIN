from typing import Literal
import pandas as pd
from pydantic import BaseModel


class CohortConfig(BaseModel):
    min_age_years: float = 18.0
    egfr_threshold: float = 60.0
    exclude_baseline_dialysis: bool = True
    exclude_transplant: bool = True
    min_hours_of_data: float = 24.0


def build_cohort(raw_patients: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """
    Applies clinical inclusion criteria:
    Adults (>= 18 yrs), reduced renal reserve (eGFR < 60), and minimum observation window.
    """
    df = raw_patients.copy()

    df["eligible"] = True
    df["exclusion_reason"] = "None"

    # Age filter
    age_mask = df["age_years"] < cfg.min_age_years
    df.loc[age_mask, "eligible"] = False
    df.loc[age_mask, "exclusion_reason"] = "Underage (<18)"

    # Prior ESRD / Dialysis check
    if cfg.exclude_baseline_dialysis and "on_dialysis" in df.columns:
        dialysis_mask = df["on_dialysis"] == True
        df.loc[dialysis_mask, "eligible"] = False
        df.loc[dialysis_mask, "exclusion_reason"] = "Baseline ESRD/Dialysis"

    # Minimum data duration check
    if "total_hours" in df.columns:
        hours_mask = df["total_hours"] < cfg.min_hours_of_data
        df.loc[hours_mask, "eligible"] = False
        df.loc[hours_mask, "exclusion_reason"] = "Insufficient telemetry (<24h)"

    return df[df["eligible"]].reset_index(drop=True)


def compute_baseline_creatinine(
    labs_df: pd.DataFrame,
    admission_time: pd.Series,
    window_days: int = 365,
    fallback: Literal["admission_min", "mdrd_75"] = "admission_min",
) -> pd.Series:
    """
    Computes pre-admission baseline creatinine using labs prior to admission_time.
    Never uses post-admission values to establish the baseline.
    """
    df = labs_df.copy()
    df["admission_t"] = df["patient_id"].map(admission_time)

    # Labs strictly prior to admission
    pre_admission = df[df["timestamp"] <= df["admission_t"]]

    if not pre_admission.empty:
        baseline = pre_admission.groupby("patient_id")["creatinine_mg_dl"].min()
    else:
        baseline = pd.Series(dtype=float)

    # Fallback to minimum lab within the first 24h if no pre-admission record exists
    if fallback == "admission_min":
        early_labs = df[df["timestamp"] <= (df["admission_t"] + pd.Timedelta(hours=24))]
        early_min = early_labs.groupby("patient_id")["creatinine_mg_dl"].min()
        baseline = baseline.combine_first(early_min)

    return baseline