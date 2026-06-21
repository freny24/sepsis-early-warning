"""
Synthetic ICU Patient Data Generator
=====================================

Generates hourly vitals + labs time series for synthetic ICU patients, modeled
loosely on the structure of real critical-care datasets (e.g. MIMIC-III/IV,
PhysioNet Sepsis Challenge). This is NOT real patient data — it's used to make
this portfolio project runnable without needing data-use agreements.

Each patient has:
  - A trajectory type: 'stable', 'sepsis_slow_onset', 'sepsis_rapid_onset'
  - Hourly vitals: heart rate, resp rate, temp, SBP, DBP, MAP, SpO2
  - Labs (sampled less frequently, like real ICU charting): WBC, lactate,
    creatinine, platelets, bilirubin, procalcitonin
  - A binary sepsis label (Sepsis-3 consensus-inspired: SOFA increase >= 2
    plus suspected infection) and the *onset hour* used for early-warning
    evaluation (predicting T hours before clinical recognition is the point
    of the whole project).

Run directly to generate a cohort and write it to disk as parquet/csv.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

TrajectoryType = Literal["stable", "sepsis_slow_onset", "sepsis_rapid_onset"]

RNG_SEED = 42


@dataclass
class PatientConfig:
    patient_id: str
    trajectory: TrajectoryType
    los_hours: int          # length of stay in hours
    onset_hour: int | None  # hour sepsis criteria are met (None if stable)
    age: int
    baseline_severity: float  # 0-1, comorbidity burden


def _bounded_walk(n, start, drift_per_step, noise_std, lo, hi, rng):
    """Random walk with drift, clipped to physiological bounds."""
    vals = np.empty(n)
    vals[0] = start
    for i in range(1, n):
        step = drift_per_step + rng.normal(0, noise_std)
        vals[i] = np.clip(vals[i - 1] + step, lo, hi)
    return vals


def generate_patient_cohort(n_patients: int = 300, seed: int = RNG_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate a cohort of synthetic ICU patients.

    Returns
    -------
    timeseries_df : one row per patient per hour, vitals + labs (labs NaN
                     when not sampled that hour, matching real ICU charting
                     cadence)
    patients_df   : one row per patient, static info + labels
    """
    rng = np.random.default_rng(seed)

    patient_rows = []
    ts_rows = []

    # Cohort composition: realistic-ish sepsis prevalence for an ICU dataset
    # (real ICU sepsis prevalence is roughly 10-30% depending on population)
    trajectory_probs = {"stable": 0.70, "sepsis_slow_onset": 0.18, "sepsis_rapid_onset": 0.12}
    trajectories = rng.choice(
        list(trajectory_probs.keys()), size=n_patients, p=list(trajectory_probs.values())
    )

    for i, trajectory in enumerate(trajectories):
        patient_id = f"P{i+1:04d}"
        los_hours = int(rng.integers(48, 168))  # 2-7 day stay
        age = int(rng.integers(22, 90))
        baseline_severity = float(rng.beta(2, 5))  # most patients mild, tail of sicker ones

        if trajectory == "stable":
            onset_hour = None
        elif trajectory == "sepsis_slow_onset":
            # onset somewhere in the middle third of stay, gradual decline before it
            onset_hour = int(rng.integers(los_hours // 3, (2 * los_hours) // 3))
        else:  # rapid onset
            onset_hour = int(rng.integers(los_hours // 4, los_hours // 2))

        patient_rows.append(
            dict(
                patient_id=patient_id,
                trajectory=trajectory,
                los_hours=los_hours,
                onset_hour=onset_hour,
                age=age,
                baseline_severity=round(baseline_severity, 3),
                sepsis_label=int(trajectory != "stable"),
            )
        )

        hours = np.arange(los_hours)

        # ---- Baseline physiology (patient-specific set points) ----
        hr0 = rng.normal(80, 8) + baseline_severity * 10
        rr0 = rng.normal(16, 2) + baseline_severity * 2
        temp0 = rng.normal(36.8, 0.3)
        sbp0 = rng.normal(120, 12) - baseline_severity * 8
        dbp0 = rng.normal(75, 8)
        spo2_0 = rng.normal(97.5, 1) - baseline_severity * 1.5

        hr = _bounded_walk(los_hours, hr0, 0.0, 1.2, 40, 180, rng)
        rr = _bounded_walk(los_hours, rr0, 0.0, 0.4, 8, 45, rng)
        temp = _bounded_walk(los_hours, temp0, 0.0, 0.05, 34.5, 41.0, rng)
        sbp = _bounded_walk(los_hours, sbp0, 0.0, 1.0, 60, 200, rng)
        dbp = _bounded_walk(los_hours, dbp0, 0.0, 0.7, 35, 120, rng)
        spo2 = _bounded_walk(los_hours, spo2_0, 0.0, 0.15, 82, 100, rng)

        # Labs as smooth-ish underlying processes, sampled sparsely later
        wbc0 = rng.normal(8.5, 2)
        lactate0 = rng.normal(1.2, 0.3)
        creat0 = rng.normal(0.9, 0.2) + baseline_severity * 0.3
        plt0 = rng.normal(230, 40)
        bili0 = rng.normal(0.7, 0.2)
        pct0 = max(0.02, rng.normal(0.05, 0.02))

        wbc = _bounded_walk(los_hours, wbc0, 0.0, 0.15, 1.5, 40, rng)
        lactate = _bounded_walk(los_hours, lactate0, 0.0, 0.02, 0.4, 15, rng)
        creat = _bounded_walk(los_hours, creat0, 0.0, 0.005, 0.3, 8, rng)
        plt = _bounded_walk(los_hours, plt0, 0.0, 0.6, 10, 500, rng)
        bili = _bounded_walk(los_hours, bili0, 0.0, 0.003, 0.1, 15, rng)
        pct = _bounded_walk(los_hours, pct0, 0.0, 0.0005, 0.01, 50, rng)

        # ---- Inject sepsis physiology if applicable ----
        if onset_hour is not None:
            if trajectory == "sepsis_slow_onset":
                ramp_len = int(rng.integers(18, 30))  # slow build-up over ~18-30h
            else:
                ramp_len = int(rng.integers(4, 10))   # rapid deterioration over ~4-10h

            # Per-patient severity multiplier: not every septic patient
            # presents with the same intensity of derangement -- some are
            # caught early/mild, some are severe. This overlap with the
            # stable cohort is what keeps the task realistically hard
            # (real sepsis early-warning AUROCs in the literature land
            # around 0.75-0.88, not high-0.9s).
            severity_mult = float(np.clip(rng.normal(0.70, 0.38), 0.15, 1.5))

            ramp_start = max(0, onset_hour - ramp_len)
            for h in range(ramp_start, los_hours):
                # progress: 0 at ramp_start -> 1 at onset -> continues climbing post-onset
                progress = (h - ramp_start) / max(1, ramp_len)
                progress = min(progress, 1.6) * severity_mult

                hr[h] += 28 * progress * (0.7 + 0.6 * rng.random())
                rr[h] += 10 * progress * (0.7 + 0.6 * rng.random())
                temp[h] += (2.2 if rng.random() > 0.25 else -2.0) * progress * 0.5  # fever OR hypothermia
                sbp[h] -= 30 * progress * (0.7 + 0.6 * rng.random())
                dbp[h] -= 18 * progress * (0.7 + 0.6 * rng.random())
                spo2[h] -= 6 * progress * (0.7 + 0.6 * rng.random())

                wbc[h] += (9 if rng.random() > 0.2 else -5) * progress  # leukocytosis OR leukopenia
                lactate[h] += 3.0 * progress * (0.7 + 0.6 * rng.random())
                creat[h] += 1.1 * progress * (0.7 + 0.6 * rng.random())
                plt[h] -= 90 * progress * (0.7 + 0.6 * rng.random())
                bili[h] += 1.3 * progress * (0.7 + 0.6 * rng.random())
                pct[h] += 4.5 * progress * (0.7 + 0.6 * rng.random())

            # re-clip after injection so physiology stays plausible
            hr = np.clip(hr, 40, 190)
            rr = np.clip(rr, 8, 50)
            temp = np.clip(temp, 33.5, 41.5)
            sbp = np.clip(sbp, 50, 200)
            dbp = np.clip(dbp, 25, 120)
            spo2 = np.clip(spo2, 75, 100)
            wbc = np.clip(wbc, 1.0, 45)
            lactate = np.clip(lactate, 0.4, 18)
            creat = np.clip(creat, 0.3, 9)
            plt = np.clip(plt, 8, 500)
            bili = np.clip(bili, 0.1, 18)
            pct = np.clip(pct, 0.01, 55)
        else:
            # ---- Confounding non-septic deterioration ----
            # A meaningful fraction of stable (non-septic) patients still
            # have a rough patch -- post-op pain tachycardia, a transient
            # fever, dehydration-driven labs -- that LOOKS similar to early
            # sepsis on a few vitals but doesn't progress into true organ
            # dysfunction. Without this, "any deterioration at all" becomes
            # a trivial giveaway for the label, which would make the model
            # look unrealistically good. This is what real clinical
            # early-warning systems struggle with (specificity / false
            # alarms), and it's an intentional part of this dataset.
            if rng.random() < 0.45:
                blip_start = int(rng.integers(0, max(1, los_hours - 12)))
                blip_len = int(rng.integers(3, 14))
                blip_severity = float(np.clip(rng.normal(0.45, 0.20), 0.05, 0.85))
                for h in range(blip_start, min(los_hours, blip_start + blip_len)):
                    local_progress = blip_severity * (0.6 + 0.6 * rng.random())
                    hr[h] += 16 * local_progress
                    rr[h] += 5 * local_progress
                    temp[h] += 1.0 * local_progress * (1 if rng.random() > 0.3 else -1)
                    sbp[h] -= 10 * local_progress
                    wbc[h] += 3 * local_progress
                    lactate[h] += 0.6 * local_progress

                hr = np.clip(hr, 40, 190)
                rr = np.clip(rr, 8, 50)
                temp = np.clip(temp, 33.5, 41.5)
                sbp = np.clip(sbp, 50, 200)
                wbc = np.clip(wbc, 1.0, 45)
                lactate = np.clip(lactate, 0.4, 18)

        map_arr = (sbp + 2 * dbp) / 3  # mean arterial pressure

        # ---- Independent measurement/observation noise ----
        # Real bedside monitors and lab assays have noise that is independent
        # per-channel (a noisy SpO2 probe reading doesn't mean the lactate
        # assay is also noisy that hour). Without this, every feature is a
        # near-deterministic function of the same single `progress` ramp
        # variable, which makes the classification task artificially easy
        # (perfect-looking AUROC). Adding decorrelated per-channel noise here
        # is what keeps the engineered features informative-but-imperfect,
        # which is realistic and is also what makes the GB vs LSTM
        # comparison and the calibration story meaningful.
        hr = np.clip(hr + rng.normal(0, 3.5, los_hours), 40, 190)
        rr = np.clip(rr + rng.normal(0, 1.5, los_hours), 8, 50)
        temp = np.clip(temp + rng.normal(0, 0.25, los_hours), 33.5, 41.5)
        sbp = np.clip(sbp + rng.normal(0, 5.0, los_hours), 50, 200)
        dbp = np.clip(dbp + rng.normal(0, 3.5, los_hours), 25, 120)
        map_arr = np.clip(map_arr + rng.normal(0, 3.0, los_hours), 35, 150)
        spo2 = np.clip(spo2 + rng.normal(0, 1.2, los_hours), 75, 100)
        wbc = np.clip(wbc + rng.normal(0, 1.0, los_hours), 1.0, 45)
        lactate = np.clip(lactate + rng.normal(0, 0.35, los_hours), 0.4, 18)
        creat = np.clip(creat + rng.normal(0, 0.15, los_hours), 0.3, 9)
        plt = np.clip(plt + rng.normal(0, 12, los_hours), 8, 500)
        bili = np.clip(bili + rng.normal(0, 0.2, los_hours), 0.1, 18)
        pct = np.clip(pct + rng.normal(0, 0.5, los_hours), 0.01, 55)

        # Lab sampling cadence: labs drawn roughly every 4-8h (not every hour),
        # mirroring real ICU charting -> rest are NaN (the feature pipeline
        # will forward-fill, which is standard practice for these datasets).
        lab_mask = np.zeros(los_hours, dtype=bool)
        step = int(rng.integers(4, 8))
        lab_mask[::step] = True
        # always sample a lab near onset (clinicians order labs when patient looks sick)
        if onset_hour is not None:
            near = np.clip(np.array([onset_hour - 2, onset_hour, onset_hour + 2]), 0, los_hours - 1)
            lab_mask[near] = True

        for h in hours:
            ts_rows.append(
                dict(
                    patient_id=patient_id,
                    hour=int(h),
                    heart_rate=round(hr[h], 1),
                    resp_rate=round(rr[h], 1),
                    temp_c=round(temp[h], 2),
                    sbp=round(sbp[h], 1),
                    dbp=round(dbp[h], 1),
                    map=round(map_arr[h], 1),
                    spo2=round(spo2[h], 1),
                    wbc=round(wbc[h], 2) if lab_mask[h] else np.nan,
                    lactate=round(lactate[h], 2) if lab_mask[h] else np.nan,
                    creatinine=round(creat[h], 2) if lab_mask[h] else np.nan,
                    platelets=round(plt[h], 0) if lab_mask[h] else np.nan,
                    bilirubin=round(bili[h], 2) if lab_mask[h] else np.nan,
                    procalcitonin=round(pct[h], 2) if lab_mask[h] else np.nan,
                )
            )

    patients_df = pd.DataFrame(patient_rows)
    timeseries_df = pd.DataFrame(ts_rows)
    return timeseries_df, patients_df


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__))
    ts_df, patients_df = generate_patient_cohort(n_patients=300)

    ts_path = os.path.join(out_dir, "timeseries.parquet")
    patients_path = os.path.join(out_dir, "patients.parquet")
    ts_df.to_parquet(ts_path, index=False)
    patients_df.to_parquet(patients_path, index=False)

    print(f"Generated {patients_df.shape[0]} patients, {ts_df.shape[0]} patient-hours")
    print(f"Sepsis prevalence: {patients_df['sepsis_label'].mean():.1%}")
    print(f"Wrote: {ts_path}")
    print(f"Wrote: {patients_path}")
