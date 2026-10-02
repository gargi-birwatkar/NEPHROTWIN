from typing import Literal

def egfr_ckd_epi_2021(creatinine_mg_dl: float, age_years: float, sex: Literal["F", "M", "O"]) -> float:
    """Calculates CKD-EPI 2021 race-free eGFR."""
    kappa = 0.7 if sex == "F" else 0.9
    alpha = -0.241 if sex == "F" else -0.302
    gender_multiplier = 1.012 if sex == "F" else 1.0

    scr_k = creatinine_mg_dl / kappa
    egfr = (
        142.0
        * (min(scr_k, 1.0) ** alpha)
        * (max(scr_k, 1.0) ** -1.200)
        * (0.9938 ** age_years)
        * gender_multiplier
    )
    return float(round(egfr, 2))