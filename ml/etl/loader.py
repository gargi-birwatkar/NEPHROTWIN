import pandas as pd
from pathlib import Path

def load_uci_ckd(csv_path: Path = Path("ml/data/raw/uci_ckd.csv")) -> pd.DataFrame:
    """
    Cleans raw Apollo Hospitals CKD data into canonical types.
    Handles '?' strings, converts units, and extracts baseline features.
    """
    df = pd.read_csv(csv_path, na_values=["?", "\t?"])
    
    # Standardize column names
    rename_map = {
        "age": "age_years",
        "sc": "baseline_creatinine_mg_dl",
        "bp": "sbp_mmhg",
        "classification": "ckd_label"
    }
    df = df.rename(columns=rename_map)
    
    # Standardize binary target: 'ckd' -> 1, 'notckd' -> 0
    df["target"] = df["ckd_label"].astype(str).str.strip().apply(
        lambda x: 1 if "ckd" in x.lower() and "not" not in x.lower() else 0
    )
    
    # Fill or typecast numeric fields
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["baseline_creatinine_mg_dl"] = pd.to_numeric(df["baseline_creatinine_mg_dl"], errors="coerce")
    
    return df