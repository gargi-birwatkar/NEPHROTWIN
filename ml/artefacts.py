from datetime import datetime
from typing import Sequence
import joblib
from contracts.interfaces import (
    ForecasterMeta,
    FeatureResult,
    ForecastResult,
    TrajectoryPerturbation,
    risk_band,
)
from ml.explain.explainer import CalibratedTreeExplainer
import numpy as np


class MLForecaster:
    def __init__(self, explainer: CalibratedTreeExplainer, meta: ForecasterMeta):
        self.explainer = explainer
        self.meta = meta

    def predict(
        self, features: FeatureResult, patient_id: str, prediction_time: datetime
    ) -> ForecastResult:
        p_24, int_24, drivers = self.explainer.explain_vector(features.vector)
        
        # Consistent scaling and strictly ordered intervals
        p_48 = float(np.clip(p_24 * 1.10, 0.0, 1.0))
        lo_48 = float(max(0.0, min(p_48, int_24[0] * 1.05)))
        hi_48 = float(min(1.0, max(p_48, int_24[1] * 1.05)))

        missing_ratio = features.missingness.get("frac_missing", 0.0)
        low_conf = missing_ratio > 0.35
        reasons = ["Excessive missing features in telemetry input."] if low_conf else []

        return ForecastResult(
            patient_id=patient_id,
            prediction_time=prediction_time,
            risk_24h=p_24,
            risk_48h=p_48,
            risk_24h_interval=(lo_48, hi_48) if p_24 == p_48 else (int_24[0], int_24[1]),
            risk_48h_interval=(lo_48, hi_48),
            risk_band=risk_band(p_48),
            low_confidence=low_conf,
            low_confidence_reasons=reasons,
            drivers=drivers,
            model_version=self.meta.model_version,
        )
    def predict_batch(
        self,
        features: Sequence[FeatureResult],
        patient_ids: Sequence[str],
        prediction_time: datetime,
    ) -> list[ForecastResult]:
        return [self.predict(f, pid, prediction_time) for f, pid in zip(features, patient_ids)]

    def predict_trajectory(
        self,
        base_features: FeatureResult,
        modified_telemetry_delta: TrajectoryPerturbation,
        patient_id: str,
        prediction_time: datetime,
    ) -> ForecastResult:
        vec = list(base_features.vector)
        for name, val in modified_telemetry_delta.feature_overrides.items():
            if name in base_features.names:
                vec[base_features.names.index(name)] = val
        for name, delta in modified_telemetry_delta.feature_deltas.items():
            if name in base_features.names:
                vec[base_features.names.index(name)] += delta

        perturbed = base_features.model_copy(update={"vector": vec})
        return self.predict(perturbed, patient_id, prediction_time)