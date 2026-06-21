"""
Gradient Boosting Model
==========================

XGBoost classifier on the engineered tabular features (SOFA components,
delta vitals, rolling stats, shock index, etc). This is the "fast, accurate
on tabular features" half of the ensemble -- it's excellent at picking up
threshold-like and interaction effects (e.g. "lactate is high AND rising"),
but treats each patient-hour independently and has no real memory of the
sequence shape, which is what the LSTM half is for.

Patient-level train/test split (NOT row-level) is used throughout, since
splitting individual hours of the same patient across train/test would leak
information and inflate performance -- a classic mistake in clinical time
series ML that this project deliberately avoids.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
MODELS_OUT = BASE

EXCLUDE_COLS = {"patient_id", "hour", "trajectory", "onset_hour", "label"}


def load_features():
    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    return df, feature_cols


def patient_level_split(df, test_size=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def train_gb_model():
    df, feature_cols = load_features()
    train_df, test_df = patient_level_split(df)

    X_train, y_train = train_df[feature_cols], train_df["label"]
    X_test, y_test = test_df[feature_cols], test_df["label"]

    # Class imbalance handling: scale_pos_weight ~ (negatives / positives)
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())

    base_model = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    base_model.fit(X_train, y_train)

    raw_probs = base_model.predict_proba(X_test)[:, 1]

    # --- Calibration ---
    # Gradient boosted trees are well known to produce poorly calibrated
    # probabilities (systematically push scores toward 0/1). We fit an
    # isotonic regression calibrator on a held-out fold of the *training*
    # patients (not the test set!) so the reported AUROC/AUPRC stay an
    # honest estimate of out-of-sample performance. Isotonic regression is
    # fit directly on (uncalibrated model score -> empirical positive rate)
    # pairs, which is the same idea CalibratedClassifierCV(method="isotonic")
    # implements internally, just done explicitly here for transparency.
    cal_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=7)
    fit_idx, cal_idx = next(cal_split.split(train_df, groups=train_df["patient_id"]))
    fit_df, cal_df = train_df.iloc[fit_idx], train_df.iloc[cal_idx]

    inner = XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight, eval_metric="aucpr",
        random_state=42, n_jobs=-1,
    )
    inner.fit(fit_df[feature_cols], fit_df["label"])

    cal_raw_probs = inner.predict_proba(cal_df[feature_cols])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(cal_raw_probs, cal_df["label"])

    test_raw_probs = inner.predict_proba(X_test)[:, 1]
    calibrated_probs = calibrator.predict(test_raw_probs)

    # --- Metrics ---
    auroc = roc_auc_score(y_test, calibrated_probs)
    auprc = average_precision_score(y_test, calibrated_probs)
    fpr, tpr, roc_thresh = roc_curve(y_test, calibrated_probs)
    prec, rec, pr_thresh = precision_recall_curve(y_test, calibrated_probs)

    print(f"[GB] Test AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")

    # Save model artifacts
    import joblib
    joblib.dump(base_model, os.path.join(MODELS_OUT, "gb_model.joblib"))
    joblib.dump(inner, os.path.join(MODELS_OUT, "gb_model_for_calibration.joblib"))
    joblib.dump(calibrator, os.path.join(MODELS_OUT, "gb_calibrator.joblib"))

    # Feature importance for the dashboard
    importances = pd.Series(base_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    results = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_train_patients": int(train_df["patient_id"].nunique()),
        "n_test_patients": int(test_df["patient_id"].nunique()),
        "roc_curve": {"fpr": fpr[::4].tolist(), "tpr": tpr[::4].tolist()},
        "pr_curve": {"precision": prec[::4].tolist(), "recall": rec[::4].tolist()},
        "top_features": importances.head(15).to_dict(),
        "test_probs": calibrated_probs.tolist(),
        "test_probs_raw": test_raw_probs.tolist(),
        "test_labels": y_test.tolist(),
        "test_patient_ids": test_df["patient_id"].tolist(),
        "test_hours": test_df["hour"].tolist(),
    }
    with open(os.path.join(MODELS_OUT, "gb_results.json"), "w") as f:
        json.dump(results, f)

    print(f"Saved model + results to {MODELS_OUT}")
    return results


if __name__ == "__main__":
    train_gb_model()
