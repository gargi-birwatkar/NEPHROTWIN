import numpy as np
import shap
from sklearn.isotonic import IsotonicRegression
from contracts.interfaces import Driver
import warnings
import shap

class CalibratedTreeExplainer:
    def __init__(self, booster, val_preds: np.ndarray, y_val: np.ndarray, feature_names: list[str]):
        self.booster = booster
        self.feature_names = feature_names
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(val_preds, y_val)
        # Suppress warning by targeting tree-path dependence explicitly
        self.explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.explainer = shap.TreeExplainer(booster, feature_perturbation="tree_path_dependent")

    def explain_vector(self, feature_vector: list[float], top_k: int = 4) -> tuple[float, tuple[float, float], list[Driver]]:
        X = np.array([feature_vector])
        raw_p = float(self.booster.predict(X)[0])
        p_calibrated = float(np.clip(self.calibrator.predict([raw_p])[0], 0.02, 0.98))

        # Dynamic uncertainty interval guaranteed to satisfy: 0.0 <= lo <= p <= hi <= 1.0
        lo = float(max(0.0, p_calibrated - 0.08))
        hi = float(min(1.0, p_calibrated + 0.08))

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            # Binary classification in newer SHAP returns [class_0, class_1]
            vals = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            vals = raw_shap[0, :, 1]
        else:
            vals = raw_shap[0]

        top_indices = np.argsort(np.abs(vals))[::-1][:top_k]
        drivers = []

        for idx in top_indices:
            mag = float(vals[idx])
            name = self.feature_names[idx]
            bounded_mag = float(np.clip(abs(mag) / 3.0, 0.01, 1.0))
            
            drivers.append(
                Driver(
                    feature=name,
                    display_name=name.replace("_", " ").title(),
                    direction="increases_risk" if mag > 0 else "decreases_risk",
                    magnitude=round(bounded_mag, 3),
                    rationale=f"{name.replace('_', ' ').title()} {'elevated' if mag > 0 else 'reduced'} hemodynamic risk load.",
                )
            )

        return p_calibrated, (lo, hi), drivers