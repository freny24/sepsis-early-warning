"""
Sepsis Early Warning System -- API
======================================

FastAPI backend serving:
  - Patient list + cohort summary stats
  - Per-patient vitals/labs time series + SOFA components
  - Per-patient model predictions (GB, LSTM, ensemble) over time
  - Model performance artifacts (ROC, PR, calibration curves, feature importance)
  - Alert-threshold operating-point table for the interactive threshold tuner

Run with:  uvicorn backend.api.app:app --reload --port 8000
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = FastAPI(
    title="Sepsis Early Warning System API",
    description=(
        "Portfolio demo API serving synthetic ICU patient data, SOFA scoring, "
        "and a Gradient Boosting + LSTM ensemble for early sepsis prediction. "
        "NOT for clinical use -- all patient data is synthetic."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load data once at startup
# ---------------------------------------------------------------------------
_patients_df = pd.read_parquet(os.path.join(DATA_DIR, "patients.parquet"))
_features_df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))

with open(os.path.join(MODELS_DIR, "ensemble_results.json")) as f:
    _ensemble_results = json.load(f)
with open(os.path.join(MODELS_DIR, "gb_results.json")) as f:
    _gb_results = json.load(f)
with open(os.path.join(MODELS_DIR, "lstm_results.json")) as f:
    _lstm_results = json.load(f)

_ensemble_preds = pd.read_parquet(os.path.join(MODELS_DIR, "ensemble_eval_predictions.parquet"))

# Build a quick lookup of which patients have ensemble predictions available
# (only the eval half of the test split has final calibrated ensemble scores)
_PREDICTED_PATIENT_IDS = set(_ensemble_preds["patient_id"].unique())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _patient_or_404(patient_id: str):
    row = _patients_df[_patients_df["patient_id"] == patient_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return row.iloc[0]


def _clean(obj):
    """Recursively replace NaN/inf with None so the response is valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "n_patients": int(_patients_df.shape[0])}


@app.get("/api/patients")
def list_patients(only_with_predictions: bool = True):
    """List all patients with summary info, for the dashboard's patient picker."""
    df = _patients_df.copy()
    if only_with_predictions:
        df = df[df["patient_id"].isin(_PREDICTED_PATIENT_IDS)]

    df = df.sort_values(["sepsis_label", "patient_id"], ascending=[False, True])
    records = df[["patient_id", "trajectory", "los_hours", "onset_hour", "age", "sepsis_label"]].to_dict(orient="records")
    return _clean({"count": len(records), "patients": records})


@app.get("/api/cohort/summary")
def cohort_summary():
    """High-level cohort stats for a dashboard header/overview panel."""
    df = _patients_df
    return _clean({
        "n_patients": int(df.shape[0]),
        "n_sepsis": int(df["sepsis_label"].sum()),
        "sepsis_prevalence": float(df["sepsis_label"].mean()),
        "trajectory_counts": df["trajectory"].value_counts().to_dict(),
        "median_los_hours": float(df["los_hours"].median()),
        "median_age": float(df["age"].median()),
    })


@app.get("/api/patients/{patient_id}/timeseries")
def patient_timeseries(patient_id: str):
    """Hourly vitals, labs, and SOFA components for one patient."""
    _patient_or_404(patient_id)
    df = _features_df[_features_df["patient_id"] == patient_id].sort_values("hour")
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No timeseries for {patient_id}")

    cols = [
        "hour", "heart_rate", "resp_rate", "temp_c", "sbp", "dbp", "map", "spo2",
        "wbc", "lactate", "creatinine", "platelets", "bilirubin", "procalcitonin",
        "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal", "sofa_resp", "sofa_total",
        "sofa_delta_from_baseline", "shock_index", "label",
    ]
    records = df[cols].to_dict(orient="records")

    patient_info = _patient_or_404(patient_id)
    return _clean({
        "patient_id": patient_id,
        "trajectory": patient_info["trajectory"],
        "onset_hour": patient_info["onset_hour"],
        "los_hours": int(patient_info["los_hours"]),
        "timeseries": records,
    })


@app.get("/api/patients/{patient_id}/predictions")
def patient_predictions(patient_id: str):
    """Per-hour GB / LSTM / ensemble probabilities for one patient (eval-set patients only)."""
    _patient_or_404(patient_id)
    df = _ensemble_preds[_ensemble_preds["patient_id"] == patient_id].sort_values("hour")
    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No model predictions available for {patient_id} (not in evaluation set)",
        )
    records = df[["hour", "gb_prob", "lstm_prob", "ensemble_prob", "label"]].to_dict(orient="records")
    return _clean({"patient_id": patient_id, "predictions": records})


@app.get("/api/model/performance")
def model_performance():
    """ROC, PR, calibration curves and headline metrics for all 3 models."""
    return _clean({
        "ensemble": {
            "auroc": _ensemble_results["auroc"],
            "auprc": _ensemble_results["auprc"],
            "roc_curve": _ensemble_results["roc_curve"],
            "pr_curve": _ensemble_results["pr_curve"],
            "calibration_curve": _ensemble_results["calibration_curve"],
            "weight_gb": _ensemble_results["ensemble_weight_gb"],
            "weight_lstm": _ensemble_results["ensemble_weight_lstm"],
        },
        "gradient_boosting": {
            "auroc": _gb_results["auroc"],
            "auprc": _gb_results["auprc"],
            "roc_curve": _gb_results["roc_curve"],
            "pr_curve": _gb_results["pr_curve"],
        },
        "lstm": {
            "auroc": _lstm_results["auroc"],
            "auprc": _lstm_results["auprc"],
            "roc_curve": _lstm_results["roc_curve"],
            "pr_curve": _lstm_results["pr_curve"],
            "train_losses": _lstm_results["train_losses"],
        },
    })


@app.get("/api/model/feature-importance")
def feature_importance():
    return _clean({"top_features": _gb_results["top_features"]})


@app.get("/api/model/operating-points")
def operating_points():
    """
    Full sensitivity/specificity/PPV/NPV/alert-rate table across thresholds,
    used by the dashboard's interactive alert-threshold slider.
    """
    return _clean({"operating_points": _ensemble_results["operating_points"]})


@app.get("/api/model/operating-point")
def operating_point_at(threshold: float = 0.5):
    """Single nearest operating point for a given threshold (for live slider feedback)."""
    points = _ensemble_results["operating_points"]
    closest = min(points, key=lambda p: abs(p["threshold"] - threshold))
    return _clean(closest)
