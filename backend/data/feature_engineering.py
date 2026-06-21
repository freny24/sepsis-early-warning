"""
Feature Engineering
=====================

Builds the feature set used by both the Gradient Boosting model (tabular,
one row per patient-hour) and the LSTM (sequences of these same features).

Feature families:
  1. Raw vitals/labs (forward-filled)
  2. SOFA components + total (from sofa_score.py)
  3. Delta features: change over the last 1h, 4h, 8h windows -- these are
     what let the model see deterioration trends rather than just absolute
     values, which is the core idea behind early-warning scoring (a HR of
     110 that was 80 four hours ago is a very different signal than a HR
     that's been steady at 110 all admission).
  4. Rolling statistics: 6h rolling mean/std, to capture variability
     (loss of heart-rate variability is itself a known early sepsis signal)
  5. Shock index (HR/SBP) -- a simple, well-known clinical deterioration ratio
  6. Time since admission (hour) -- weak but real signal, deterioration risk
     compounds over a long ICU stay
"""

import numpy as np
import pandas as pd

VITAL_COLS = ["heart_rate", "resp_rate", "temp_c", "sbp", "dbp", "map", "spo2"]
LAB_COLS = ["wbc", "lactate", "creatinine", "platelets", "bilirubin", "procalcitonin"]
SOFA_COLS = ["sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal", "sofa_resp", "sofa_total"]

DELTA_WINDOWS = [1, 4, 8]
ROLLING_WINDOW = 6


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df already SOFA-scored (output of sofa_score.compute_sofa),
    sorted by patient_id, hour, with labs forward/back-filled.
    Returns df with engineered feature columns appended.
    """
    df = df.sort_values(["patient_id", "hour"]).copy()
    grp = df.groupby("patient_id")

    # --- Delta features ---
    for col in VITAL_COLS + LAB_COLS:
        for w in DELTA_WINDOWS:
            df[f"{col}_delta_{w}h"] = grp[col].diff(w)

    # --- Rolling mean/std (vitals only -- labs too sparse pre-ffill for meaningful rolling std) ---
    for col in VITAL_COLS:
        df[f"{col}_roll_mean_{ROLLING_WINDOW}h"] = (
            grp[col].transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=2).mean())
        )
        df[f"{col}_roll_std_{ROLLING_WINDOW}h"] = (
            grp[col].transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=2).std())
        )

    # --- Shock index: HR / SBP, classic bedside deterioration ratio ---
    df["shock_index"] = df["heart_rate"] / df["sbp"].replace(0, np.nan)

    # --- Time in stay ---
    df["hour_in_stay"] = df["hour"]

    # Fill early-stay NaNs (no history yet for deltas/rolling) with 0 (=no change detected),
    # standard convention for these features at series start.
    engineered_cols = [c for c in df.columns if "_delta_" in c or "_roll_" in c]
    df[engineered_cols] = df[engineered_cols].fillna(0)

    return df


def make_prediction_labels(df: pd.DataFrame, patients_df: pd.DataFrame, horizon_hours: int = 6) -> pd.DataFrame:
    """
    Creates the EARLY-WARNING target: label = 1 if sepsis onset occurs within
    the next `horizon_hours` of this row's timestamp. This is the key framing
    that makes it an *early warning* system rather than a same-time
    classifier -- we want the model to fire before clinical recognition, not
    at the moment of it.

    Rows after onset are also labeled 1 (patient is septic), rows more than
    horizon_hours before onset are 0, and all rows for 'stable' patients are 0.

    A small per-patient jitter (+/- 2h) is applied to the effective horizon
    boundary. Clinical recognition of sepsis onset is itself a somewhat noisy
    label in real datasets (it depends on when a clinician orders cultures
    and charts a suspected-infection flag, not a a clean biological instant),
    so treating the boundary as exact would hand the model an unrealistically
    clean decision surface.
    """
    df = df.merge(patients_df[["patient_id", "trajectory", "onset_hour"]], on="patient_id", how="left")

    rng = np.random.default_rng(123)
    jitter_by_patient = {
        pid: rng.integers(-2, 3) for pid in df["patient_id"].unique()
    }

    def label_row(row):
        if pd.isna(row["onset_hour"]):
            return 0
        effective_horizon = horizon_hours + jitter_by_patient[row["patient_id"]]
        return int(row["hour"] >= row["onset_hour"] - effective_horizon)

    df["label"] = df.apply(label_row, axis=1)
    return df


if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    ts_df = pd.read_parquet(os.path.join(base, "timeseries_with_sofa.parquet"))
    patients_df = pd.read_parquet(os.path.join(base, "patients.parquet"))

    featured = build_features(ts_df)
    labeled = make_prediction_labels(featured, patients_df, horizon_hours=6)

    out_path = os.path.join(base, "features.parquet")
    labeled.to_parquet(out_path, index=False)
    print(f"Feature matrix shape: {labeled.shape}")
    print(f"Positive label rate: {labeled['label'].mean():.1%}")
    print(f"Wrote: {out_path}")
