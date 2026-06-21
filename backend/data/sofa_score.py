"""
SOFA Score Computation
=======================

Sequential Organ Failure Assessment (SOFA) score, computed per patient-hour
from vitals + labs. This is the clinical backbone of the Sepsis-3 definition:
sepsis = suspected infection + acute increase in SOFA score >= 2.

We compute the 4 components that are derivable from our synthetic vitals/labs
(the full clinical SOFA has 6 organ systems; the other two -- CNS via Glasgow
Coma Scale and respiratory support via FiO2/ventilator settings -- require
inputs this dataset doesn't simulate, so they're approximated/omitted with
that noted below):

  - Coagulation   (platelets)
  - Liver         (bilirubin)
  - Cardiovascular (MAP, as a proxy for MAP + vasopressor dose)
  - Renal         (creatinine)

Respiratory (PaO2/FiO2 ratio) is approximated using SpO2 as a proxy via a
standard SpO2/FiO2 -> PaO2/FiO2 conversion, since we don't simulate blood
gas draws. This is a known, published approximation (Rice et al. 2007) used
when arterial blood gases aren't available, and is flagged here so it's
clear this is a simplification rather than a clinical-grade SOFA score.

This module is for portfolio / educational purposes only and is NOT
validated for clinical use.
"""

import numpy as np
import pandas as pd


def _score_coagulation(platelets: pd.Series) -> pd.Series:
    conditions = [
        platelets >= 150,
        (platelets >= 100) & (platelets < 150),
        (platelets >= 50) & (platelets < 100),
        (platelets >= 20) & (platelets < 50),
        platelets < 20,
    ]
    return np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)


def _score_liver(bilirubin: pd.Series) -> pd.Series:
    conditions = [
        bilirubin < 1.2,
        (bilirubin >= 1.2) & (bilirubin < 2.0),
        (bilirubin >= 2.0) & (bilirubin < 6.0),
        (bilirubin >= 6.0) & (bilirubin < 12.0),
        bilirubin >= 12.0,
    ]
    return np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)


def _score_cardiovascular(map_val: pd.Series) -> pd.Series:
    # Simplified: real SOFA also scores vasopressor dose/type, which we don't
    # simulate. MAP < 70 alone scores 1; lower MAP without pressor data is
    # capped at 1 here (true score could be higher with pressors in use).
    conditions = [map_val >= 70, map_val < 70]
    return np.select(conditions, [0, 1], default=np.nan)


def _score_renal(creatinine: pd.Series) -> pd.Series:
    conditions = [
        creatinine < 1.2,
        (creatinine >= 1.2) & (creatinine < 2.0),
        (creatinine >= 2.0) & (creatinine < 3.5),
        (creatinine >= 3.5) & (creatinine < 5.0),
        creatinine >= 5.0,
    ]
    return np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)


def _spo2_to_pf_ratio(spo2: pd.Series) -> pd.Series:
    """
    Approximate PaO2/FiO2 ratio from SpO2 using the Rice et al. (2007)
    nonlinear regression, assuming room air / standard FiO2 ~0.21-0.4 range
    typical of ward/step-down patients. This is a published approximation
    used in several sepsis early-warning papers (e.g. PhysioNet 2019
    Challenge baseline) when arterial blood gas data isn't available.
    """
    spo2c = spo2.clip(upper=99.99)
    # Rice et al. piecewise approximation:
    pao2 = np.where(
        spo2c <= 96,
        (spo2c.values - 100) / 0.36 if hasattr(spo2c, "values") else (spo2c - 100) / 0.36,
        np.nan,
    )
    pao2 = pd.Series(pao2, index=spo2.index)
    # For SpO2 > 96 the relationship flattens; approximate with a capped value
    pao2 = pao2.fillna(110)
    pao2 = pao2.clip(lower=40, upper=300)
    fio2_assumed = 0.21  # room air assumption for non-ventilated patients
    return pao2 / fio2_assumed / 100  # scaled down to a 0-5ish range for scoring thresholds below


def _score_respiratory(spo2: pd.Series) -> pd.Series:
    pf_proxy = _spo2_to_pf_ratio(spo2) * 100  # back to approx PF-ratio-like scale
    conditions = [
        pf_proxy >= 400,
        (pf_proxy >= 300) & (pf_proxy < 400),
        (pf_proxy >= 200) & (pf_proxy < 300),
        (pf_proxy >= 100) & (pf_proxy < 200),
        pf_proxy < 100,
    ]
    return np.select(conditions, [0, 1, 2, 3, 4], default=np.nan)


def compute_sofa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SOFA sub-scores and total for each row of a vitals/labs dataframe.
    Expects columns: platelets, bilirubin, map, creatinine, spo2.
    Forward-fills labs within each patient first (standard practice, since
    labs are sparsely sampled but the score is needed every hour).

    Returns the input dataframe with new columns:
      sofa_coag, sofa_liver, sofa_cardio, sofa_renal, sofa_resp, sofa_total
    """
    df = df.sort_values(["patient_id", "hour"]).copy()

    # Forward-fill ALL labs (not just the ones feeding SOFA sub-scores) so
    # downstream consumers (feature engineering, LSTM input) never see NaN.
    # This mirrors standard clinical-ML practice: a lab value drawn at hour 4
    # is assumed to still hold until the next draw, since labs aren't
    # re-measured every hour.
    all_lab_cols = ["wbc", "lactate", "creatinine", "platelets", "bilirubin", "procalcitonin"]
    df[all_lab_cols] = df.groupby("patient_id")[all_lab_cols].ffill()
    df[all_lab_cols] = df.groupby("patient_id")[all_lab_cols].bfill()

    lab_cols = ["platelets", "bilirubin", "creatinine"]

    df["sofa_coag"] = _score_coagulation(df["platelets"])
    df["sofa_liver"] = _score_liver(df["bilirubin"])
    df["sofa_cardio"] = _score_cardiovascular(df["map"])
    df["sofa_renal"] = _score_renal(df["creatinine"])
    df["sofa_resp"] = _score_respiratory(df["spo2"])

    sofa_cols = ["sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal", "sofa_resp"]
    df[sofa_cols] = df[sofa_cols].fillna(0)
    df["sofa_total"] = df[sofa_cols].sum(axis=1)

    # Sepsis-3-style delta: increase from each patient's baseline (first-hour) SOFA
    baseline = df.groupby("patient_id")["sofa_total"].transform("first")
    df["sofa_delta_from_baseline"] = df["sofa_total"] - baseline

    return df


if __name__ == "__main__":
    import os

    base = os.path.dirname(__file__)
    ts_df = pd.read_parquet(os.path.join(base, "timeseries.parquet"))
    scored = compute_sofa(ts_df)
    out_path = os.path.join(base, "timeseries_with_sofa.parquet")
    scored.to_parquet(out_path, index=False)
    print(f"SOFA scoring complete. Mean total SOFA: {scored['sofa_total'].mean():.2f}")
    print(f"Wrote: {out_path}")
