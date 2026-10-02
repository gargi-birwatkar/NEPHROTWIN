from datetime import datetime
from typing import Sequence, Optional
import numpy as np
import pandas as pd


def rolling_window_stats(
    df: pd.DataFrame,
    col: str,
    windows_h: Sequence[int] = (6, 12, 24, 48),
    as_of: Optional[datetime] = None,
) -> dict[str, float]:
    """
    Computes rolling mean, std, min, max, and linear slope for a given telemetry variable.
    Strictly filters out any timestamps ahead of as_of to prevent leakage.
    """
    if df.empty or col not in df.columns:
        res = {}
        for w in windows_h:
            for stat in ["mean", "sd", "min", "max", "slope"]:
                res[f"{col}_{stat}_{w}h"] = np.nan
        return res

    df_clean = df.sort_values("timestamp").copy()
    if as_of:
        df_clean = df_clean[df_clean["timestamp"] <= as_of]

    res = {}
    for w in windows_h:
        cutoff = (as_of or df_clean["timestamp"].max()) - pd.Timedelta(hours=w)
        window_df = df_clean[df_clean["timestamp"] >= cutoff]
        vals = window_df[col].dropna()

        if vals.empty:
            for stat in ["mean", "sd", "min", "max", "slope"]:
                res[f"{col}_{stat}_{w}h"] = np.nan
            continue

        res[f"{col}_mean_{w}h"] = float(vals.mean())
        res[f"{col}_sd_{w}h"] = float(vals.std()) if len(vals) > 1 else 0.0
        res[f"{col}_min_{w}h"] = float(vals.min())
        res[f"{col}_max_{w}h"] = float(vals.max())

        # Linear slope (rate of change per hour)
        if len(vals) > 1:
            x = (window_df.loc[vals.index, "timestamp"] - cutoff).dt.total_seconds() / 3600.0
            slope = float(np.polyfit(x, vals.values, deg=1)[0])
            res[f"{col}_slope_{w}h"] = slope
        else:
            res[f"{col}_slope_{w}h"] = 0.0

    return res


def hypotension_burden(
    map_series: pd.Series,
    threshold_mmhg: float = 65.0,
    cadence_min: int = 5,
) -> dict[str, float]:
    """
    Measures duration and fraction of time spent in renal hypoperfusion (MAP < 65 mmHg).
    """
    valid = map_series.dropna()
    if valid.empty:
        return {"hypotension_burden_min": np.nan, "hypotension_burden_pct": np.nan}

    below_threshold = valid < threshold_mmhg
    minutes_below = float(below_threshold.sum() * cadence_min)
    pct_below = float(below_threshold.mean() * 100.0)

    return {
        "hypotension_burden_min": minutes_below,
        "hypotension_burden_pct": pct_below,
    }


def nocturnal_dip_pct(
    df: pd.DataFrame,
    col: str = "map_mmhg",
    night_start_hour: int = 22,
    night_end_hour: int = 6,
) -> Optional[float]:
    """
    Calculates nocturnal dipping percentage: ((day_mean - night_mean) / day_mean) * 100.
    In CKD, a loss of dipping (<10%) or reverse dipping signals autonomic strain.
    """
    if df.empty or col not in df.columns:
        return np.nan

    df_clean = df.dropna(subset=[col]).copy()
    if df_clean.empty:
        return np.nan

    hours = df_clean["timestamp"].dt.hour
    is_night = (hours >= night_start_hour) | (hours < night_end_hour)

    day_vals = df_clean.loc[~is_night, col]
    night_vals = df_clean.loc[is_night, col]

    if day_vals.empty or night_vals.empty:
        return np.nan

    day_mean = float(day_vals.mean())
    night_mean = float(night_vals.mean())

    if day_mean <= 0:
        return np.nan

    return float(((day_mean - night_mean) / day_mean) * 100.0)