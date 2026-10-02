import os
import json
import joblib
import numpy as np
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from contracts.interfaces import StaticProfile, TelemetryFrame, ForecasterMeta
from ml.etl.synthetic import generate_synthetic_cohort
from ml.features.api import extract_features, FEATURE_NAMES
from ml.explain.explainer import CalibratedTreeExplainer
from ml.artefacts import MLForecaster

def run():
    print("1. Generating synthetic Indian-phenotype cohort...")
    df_static, df_telemetry = generate_synthetic_cohort(n_patients=150, days=3)

    pred_time = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)
    X, y, groups = [], [], []

    print("2. Extracting feature vectors across cohort...")
    for _, row in df_static.iterrows():
        pid = row["patient_id"]
        static = StaticProfile(
            patient_id=pid,
            age_years=row["age_years"],
            sex=row["sex"],
            baseline_creatinine_mg_dl=row["baseline_creatinine_mg_dl"],
            egfr_ml_min_1_73m2=row["egfr_ml_min_1_73m2"],
            uacr_category=row["uacr_category"],
            diabetes=row["diabetes"],
            hypertension=row["hypertension"],
            on_acei_arb=row["on_acei_arb"],
            nsaid_exposure=row["nsaid_exposure"],
        )
        patient_telemetry = [
            TelemetryFrame(**f) for f in df_telemetry[df_telemetry["patient_id"] == pid].to_dict(orient="records")
        ]

        feat_res = extract_features(static, patient_telemetry, pred_time)
        X.append(feat_res.vector)
        y.append(1 if row["decompensates"] else 0)
        groups.append(pid)

    X, y, groups = np.array(X), np.array(y), np.array(groups)

    print("3. Training LightGBM with Patient-Level GroupKFold CV...")
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "verbosity": -1,
        "random_state": 42,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
    )

    val_preds = booster.predict(X_val)
    print(f"Validation AUROC: {roc_auc_score(y_val, val_preds):.3f}")
    print(f"Validation PR-AUC: {average_precision_score(y_val, val_preds):.3f}")

    print("4. Calibrating probabilities & building forecaster...")
    explainer = CalibratedTreeExplainer(booster, val_preds, y_val, FEATURE_NAMES)
    meta = ForecasterMeta(
        model_version="1.0.0",
        feature_schema_version="v1.0",
        feature_names=FEATURE_NAMES,
        trained_on="Synthetic-CKD-Cohort-v1",
        is_stub=False,
    )

    forecaster = MLForecaster(explainer, meta)

    out_dir = Path("ml/artefacts/v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(forecaster, out_dir / "model.joblib")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    print(f"Done! ML bundle ready and exported to {out_dir}/model.joblib")


if __name__ == "__main__":
    run()