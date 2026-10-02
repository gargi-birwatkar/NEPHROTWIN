import warnings
import numpy as np
import shap
from sklearn.isotonic import IsotonicRegression
from contracts.interfaces import Driver


class CalibratedTreeExplainer:
    def __init__(self, booster, val_preds: np.ndarray, y_val: np.ndarray, feature_names: list[str]):
        self.booster = booster
        self.feature_names = feature_names

        # 1. Fit calibrator strictly on predicting probability of positive class (y_val == 1)
        self.calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
        self.calibrator.fit(val_preds, y_val)

        # 2. Suppress SHAP warnings cleanly
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")

    def explain_vector(self, feature_vector: list[float], top_k: int = 4) -> tuple[float, tuple[float, float], list[Driver]]:
        X = np.array([feature_vector], dtype=float)

        # Raw positive-class probability from LightGBM
        raw_p = float(self.booster.predict(X)[0])

        # Calibrated probability strictly on [0.01, 0.99]
        p_calibrated = float(np.clip(self.calibrator.predict([raw_p])[0], 0.01, 0.99))

        # Dynamic uncertainty interval around calibrated point
        lo = float(max(0.0, p_calibrated - 0.08))
        hi = float(min(1.0, p_calibrated + 0.08))

        # Extract SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw_shap = self.explainer.shap_values(X)

        # Correct class attribution indexing
        if isinstance(raw_shap, list):
            # In binary LightGBM, index 1 is attribution towards positive class (y=1)
            vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            vals = raw_shap[0, :, 1]
        else:
            vals = raw_shap[0]

        # Top magnitude features
        top_indices = np.argsort(np.abs(vals))[::-1][:top_k]
        drivers = []

        for idx in top_indices:
            mag = float(vals[idx])
            name = self.feature_names[idx]
            
            # Signed magnitude normalized to clinical display scale
            bounded_mag = float(np.clip(abs(mag) / 2.0, 0.01, 1.0))
            direction = "increases_risk" if mag > 0 else "decreases_risk"

            drivers.append(
                Driver(
                    feature=name,
                    display_name=name.replace("_", " ").title(),
                    direction=direction,
                    magnitude=round(bounded_mag, 3),
                    rationale=f"{name.replace('_', ' ').title()} {'elevated' if mag > 0 else 'reduced'} hemodynamic risk load.",
                )
            )

        return p_calibrated, (lo, hi), drivers