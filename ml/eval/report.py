import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from ml.etl.loader import fetch_vitaldb_dataset
from ml.features.api import extract_features, FEATURE_NAMES
from ml.explain.explainer import CalibratedTreeExplainer


def compute_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Computes clinical discrimination, accuracy, and calibration metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    
    # 1. Standard classification metrics
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    # 2. Sensitivity at fixed Specificity (target >= 0.85/0.90 to limit alert fatigue)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 3. Calibration Slope and Intercept (via logistic calibration line)
    eps = 1e-6
    clipped_probs = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(clipped_probs / (1 - clipped_probs)).reshape(-1, 1)
    
    lr = LogisticRegression()
    lr.fit(logits, y_true)
    cal_slope = float(lr.coef_[0][0])       # Ideal = 1.0 (Slope < 1 signals overconfidence)
    cal_intercept = float(lr.intercept_[0])  # Ideal = 0.0

    return {
        "Accuracy": acc,
        "AUROC": auroc,
        "PR-AUC": pr_auc,
        "F1-Score": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Brier-Score": brier,
        "Cal-Slope": cal_slope,
        "Cal-Intercept": cal_intercept,
    }


def evaluate_overfitting_and_performance(n_splits: int = 5):
    print("=" * 70)
    print("1. Loading Cohort & Extracting Clinical Features...")
    statics, telemetries, y_list, pids = fetch_vitaldb_dataset(n_cases=400)

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

    print(f"Cohort size: {X.shape[0]} patients, {X.shape[1]} features.")
    print(f"Prevalence (Decompensation / AKI): {y.mean() * 100:.1f}%\n")

    # Regularized parameters matching clinical guardrails
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 25,
        "colsample_bytree": 0.7,
        "subsample": 0.8,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "verbosity": -1,
        "random_state": 42,
    }

    gkf = GroupKFold(n_splits=n_splits)
    train_metrics_list = []
    val_metrics_list = []

    print(f"2. Running Patient-Level {n_splits}-Fold Cross Validation (Zero Patient Overlap)...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        booster = lgb.train(params, train_data, num_boost_round=60, valid_sets=[val_data])

        # Evaluate raw predictions
        train_preds = booster.predict(X_train)
        val_preds = booster.predict(X_val)

        # Calibrate using isotonic regression
        explainer = CalibratedTreeExplainer(booster, val_preds, y_val, FEATURE_NAMES)
        cal_val_probs = explainer.calibrator.predict(val_preds)
        cal_train_probs = explainer.calibrator.predict(train_preds)

        train_metrics_list.append(compute_clinical_metrics(y_train, cal_train_probs))
        val_metrics_list.append(compute_clinical_metrics(y_val, cal_val_probs))

    # Aggregate metrics across folds
    df_train = pd.DataFrame(train_metrics_list)
    df_val = pd.DataFrame(val_metrics_list)

    print("\n" + "=" * 70)
    print("                CROSS-VALIDATION EVALUATION REPORT                   ")
    print("=" * 70)
    print(f"{'Metric':<20} | {'Train (Mean ± SD)':<22} | {'Val (Mean ± SD)':<22} | {'Generalization Gap'}")
    print("-" * 75)

    for metric in df_val.columns:
        train_mean, train_sd = df_train[metric].mean(), df_train[metric].std()
        val_mean, val_sd = df_val[metric].mean(), df_val[metric].std()
        gap = train_mean - val_mean
        print(f"{metric:<20} | {train_mean:.3f} ± {train_sd:.3f}          | {val_mean:.3f} ± {val_sd:.3f}        | {gap:+.3f}")

    # 3. Automated Overfitting Diagnosis
    train_auc = df_train["AUROC"].mean()
    val_auc = df_val["AUROC"].mean()
    train_acc = df_train["Accuracy"].mean()
    val_acc = df_val["Accuracy"].mean()
    auc_gap = train_auc - val_auc
    cal_slope = df_val["Cal-Slope"].mean()

    print("\n" + "=" * 70)
    print("                     OVERFITTING DIAGNOSIS CHECK                     ")
    print("=" * 70)

    is_overfitted = False
    reasons = []

    if val_auc >= 0.98 or val_acc >= 0.98:
        is_overfitted = True
        reasons.append("⚠️ SUSPECTED LEAKAGE: Val AUROC/Accuracy is near 100%, indicating label contamination or trivial rules.")

    if auc_gap > 0.12:
        is_overfitted = True
        reasons.append(f"⚠️ HIGH GENERALIZATION GAP: Train AUROC exceeds Val AUROC by {auc_gap:.3f} (> 0.12 threshold). Model is memorizing train set.")

    if cal_slope < 0.65:
        reasons.append(f"⚠️ PROBABILITY OVERCONFIDENCE: Calibration slope is {cal_slope:.2f} (< 0.70). Predicted probabilities are too extreme.")

    if not is_overfitted:
        print("✅ MODEL IS PROPERLY GENERALIZED (HEALTHY FIT)")
        print(f"  - Validation Accuracy: {val_acc * 100:.1f}%")
        print(f"  - Validation AUROC:    {val_auc:.3f}")
        print(f"  - Generalization Gap:  {auc_gap:.3f} (within acceptable clinical tolerance < 0.10)")
        print(f"  - Brier Loss:          {df_val['Brier-Score'].mean():.3f} (closer to 0.00 is better)")
    else:
        print("❌ OVERFITTING OR MEMORIZATION DETECTED:")
        for r in reasons:
            print(f"  {r}")
            
    print("=" * 70)


if __name__ == "__main__":
    evaluate_overfitting_and_performance(n_splits=5)