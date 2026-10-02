import pytest
from datetime import datetime, timezone, timedelta
import numpy as np

from contracts.interfaces import StaticProfile, TelemetryFrame
from ml.features.api import extract_features, FEATURE_NAMES


@pytest.fixture
def sample_static():
    return StaticProfile(
        patient_id="TEST_001",
        age_years=64.0,
        sex="M",
        baseline_creatinine_mg_dl=1.4,
        egfr_ml_min_1_73m2=52.0,
        uacr_category="A2",
        diabetes=True,
        hypertension=True,
        on_acei_arb=True,
        nsaid_exposure=False,
    )


def test_empty_telemetry_returns_nan_without_raising(sample_static):
    """Verifies graceful degradation when no telemetry records exist."""
    now = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    res = extract_features(sample_static, [], now)

    assert len(res.vector) == len(FEATURE_NAMES)
    assert res.missingness["frac_missing"] == 1.0
    assert np.isnan(res.vector[FEATURE_NAMES.index("map_mean_24h")])


def test_naive_timestamp_raises_value_error(sample_static):
    """Verifies that naive (non-timezone-aware) timestamps are rejected."""
    naive_t = datetime(2026, 10, 2, 12, 0)
    frames = [TelemetryFrame(timestamp=naive_t, map_mmhg=80.0)]

    with pytest.raises(ValueError, match="Naive timestamps"):
        extract_features(sample_static, frames, datetime.now(timezone.utc))


def test_unsorted_frames_raise_value_error(sample_static):
    """Verifies chronological order enforcement."""
    t1 = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    t2 = t1 - timedelta(minutes=10)
    frames = [
        TelemetryFrame(timestamp=t1, map_mmhg=80.0),
        TelemetryFrame(timestamp=t2, map_mmhg=82.0),
    ]

    with pytest.raises(ValueError, match="chronological order"):
        extract_features(sample_static, frames, t1)


def test_no_temporal_leakage(sample_static):
    """Verifies frames beyond prediction_time are ignored."""
    t_pred = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    frames = [
        TelemetryFrame(timestamp=t_pred - timedelta(hours=1), map_mmhg=70.0),
        TelemetryFrame(timestamp=t_pred + timedelta(hours=1), map_mmhg=150.0),  # Leak
    ]

    res = extract_features(sample_static, frames, t_pred)
    idx = FEATURE_NAMES.index("map_mean_24h")
    # Mean must be 70.0, ignoring the 150.0 future reading
    assert res.vector[idx] == pytest.approx(70.0)