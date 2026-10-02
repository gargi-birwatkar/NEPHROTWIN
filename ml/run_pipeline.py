import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import lightgbm as lgb
from datetime import timezone
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from contracts.interfaces import ForecasterMeta
from ml.etl.loader import load_apollo_ckd, fetch_vitaldb_dataset
from ml.features.api import extract_features, FEATURE_NAMES
from ml.explain.explainer import CalibratedTreeExplainer
from ml.artefacts import MLForecaster


def run():
    print("=" * 60)
    print("1. Inspecting Apollo Hospitals (UCI CKD) Priors...")
    apollo_csv = Path("ml/data/raw/uci_ckd.csv")
    has_apollo = apollo_csv.exists()
    
    dataset_description = "VitalDB (Real Human Telemetry)"
    if has_apollo:
        dataset_description += " + Apollo Hospitals India (UCI-CKD Static Priors)"
    
    print("\n" + "=" * 60)
    print(f"DATASET SOURCE: {dataset_description}")
    print("=" * 60)
    
    if apollo_csv.exists():
        df_apollo = load_apollo_ckd(apollo_csv)
        print(f"  - Apollo Patients: {len(df_apollo)}")
        print(f"  - Mean Baseline Creatinine: {df_apollo['baseline_creatinine_mg_dl'].mean():.2f} mg/dL")
        print(f"  - Diabetes Prevalence: {df_apollo['diabetes'].mean() * 100:.1f}%")
        print(f"  - Hypertension Prevalence: {df_apollo['hypertension'].mean() * 100:.1f}%")
    else:
        print("  - [Note] ml/data/raw/uci_ckd.csv not found; skipping Apollo prior stats.")

    print("\n2. Fetching real telemetry time-series from VitalDB...")
    statics, telemetries, y_list, pids = fetch_vitaldb_dataset(n_cases=397)

    print("\n3. Extracting feature vectors via shared contract...")
    X, y, groups = [], [], []

    for static, frames, label, pid in zip(statics, telemetries, y_list, pids):
        pred_time = frames[-1].timestamp
        feat_res = extract_features(static, frames, pred_time)
        X.append(feat_res.vector)
        y.append(label)
        groups.append(pid)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    print(f"  - Total feature matrix: {X.shape}")
    print(f"  - Class distribution (AKI=1): {int(np.sum(y == 1))} / {len(y)}")

    # Class balance adjustment
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    scale_pos = float(neg_count / max(1, pos_count))

    print("\n4. Training LightGBM on real physiological features...")
    gkf = GroupKFold(n_splits=3)
    train_idx, val_idx = next(gkf.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,        # Slower learning rate
        "num_leaves": 7,              # Shallow trees (prevents deep memorization)
        "max_depth": 3,               # Strict depth cap
        "min_child_samples": 25,      # Requires at least 25 patients per leaf
        "colsample_bytree": 0.7,      # Feature subsampling
        "subsample": 0.8,             # Row subsampling
        "reg_alpha": 1.0,             # L1 regularization
        "reg_lambda": 3.0,            # L2 regularization
        "scale_pos_weight": scale_pos,
        "verbosity": -1,
        "random_state": 42,
    }

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=60,           # Stop earlier (prevents over-boosting)
        valid_sets=[val_data],
    )
    val_preds = booster.predict(X_val)
    if len(np.unique(y_val)) > 1:
        print(f"  - Real Data Val AUROC: {roc_auc_score(y_val, val_preds):.3f}")
        print(f"  - Real Data Val PR-AUC: {average_precision_score(y_val, val_preds):.3f}")
    else:
        print("  - Val fold has single-class label; metrics skipped.")

    print("\n5. Packaging Calibrated MLForecaster...")
    explainer = CalibratedTreeExplainer(booster, val_preds, y_val, FEATURE_NAMES)
    meta = ForecasterMeta(
        model_version="1.1.0",
        feature_schema_version="v1.0",
        feature_names=FEATURE_NAMES,
        trained_on=dataset_description,
        is_stub=False,
    )
    forecaster = MLForecaster(explainer, meta)

    out_dir = Path("ml/artefacts/v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(forecaster, out_dir / "model.joblib")
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta.model_dump(), f, indent=2)

    print(f"\nBundle written to {out_dir}/model.joblib")


if __name__ == "__main__":
    run()