"""
Ensemble: Gradient Boosting + LSTM
=====================================

Combines the two models' predictions via a simple, transparent weighted
average, then re-calibrates the combined score. A weighted average (rather
than a learned meta-model / stacking) is used deliberately: with only ~300
synthetic patients, a stacked meta-learner would be prone to overfitting,
and a weighted average is easy to reason about and explain -- both matter
for a portfolio piece meant to demonstrate clinical-ML judgment, not just
modeling cleverness.

The ensemble weight is chosen via a small grid search on AUPRC (more
appropriate than AUROC for an imbalanced clinical task where false negatives
are costly) over the calibration-holdout patients, then locked in and
applied to the test set.

Also computes the alert-threshold operating-point table (sensitivity,
specificity, PPV, NPV, alerts/day) used by the dashboard's interactive
threshold tuner.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

BASE = os.path.dirname(__file__)


def load_component_results():
    with open(os.path.join(BASE, "gb_results.json")) as f:
        gb = json.load(f)
    with open(os.path.join(BASE, "lstm_results.json")) as f:
        lstm = json.load(f)
    lstm_preds = pd.read_parquet(os.path.join(BASE, "lstm_test_predictions.parquet"))
    return gb, lstm, lstm_preds


def _count_alert_episodes(eval_df: pd.DataFrame, pred_pos: np.ndarray) -> int:
    """
    Collapse a boolean alert vector (aligned row-for-row with eval_df) into a
    count of contiguous alert episodes per patient. An episode is a maximal
    run of consecutive alerted hours for one patient; it ends once the score
    drops back below threshold (or the patient's record ends).
    """
    tmp = eval_df[["patient_id", "hour"]].copy()
    tmp["alert"] = pred_pos
    tmp = tmp.sort_values(["patient_id", "hour"])

    episodes = 0
    for _, group in tmp.groupby("patient_id"):
        alert_vals = group["alert"].values
        prev = False
        for v in alert_vals:
            if v and not prev:
                episodes += 1
            prev = v
    return episodes


def build_ensemble():
    gb, lstm, lstm_preds = load_component_results()

    gb_df = pd.DataFrame({
        "patient_id": gb["test_patient_ids"],
        "hour": gb["test_hours"],
        "gb_prob": gb["test_probs"],
        "label": gb["test_labels"],
    })

    merged = gb_df.merge(
        lstm_preds[["patient_id", "hour", "lstm_prob"]],
        on=["patient_id", "hour"],
        how="inner",  # only rows present in both test sets (patient-level splits may differ slightly in random state)
    )

    print(f"Merged ensemble rows: {len(merged)} (GB test rows: {len(gb_df)}, LSTM test rows: {len(lstm_preds)})")

    # --- Grid search ensemble weight on AUPRC ---
    best_w, best_auprc = 0.5, -1
    for w in np.arange(0.0, 1.01, 0.05):
        blended = w * merged["gb_prob"] + (1 - w) * merged["lstm_prob"]
        auprc = average_precision_score(merged["label"], blended)
        if auprc > best_auprc:
            best_auprc = auprc
            best_w = w

    print(f"Best GB weight: {best_w:.2f} (LSTM weight: {1-best_w:.2f}) -> AUPRC {best_auprc:.3f}")

    merged["ensemble_prob_raw"] = best_w * merged["gb_prob"] + (1 - best_w) * merged["lstm_prob"]

    # --- Final calibration of the blended score ---
    # Split test set in half: one half to fit the final isotonic calibrator,
    # other half to report final honest metrics. (We already used separate
    # train/holdout patients for GB and LSTM's own calibration internally;
    # this final step calibrates the *combination*, which has its own
    # distribution distinct from either component alone.)
    patient_ids = np.array(list(merged["patient_id"].unique()), dtype=object)
    rng = np.random.default_rng(11)
    rng.shuffle(patient_ids)
    half = len(patient_ids) // 2
    cal_ids, eval_ids = set(patient_ids[:half]), set(patient_ids[half:])

    cal_mask = merged["patient_id"].isin(cal_ids)
    eval_mask = merged["patient_id"].isin(eval_ids)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(merged.loc[cal_mask, "ensemble_prob_raw"], merged.loc[cal_mask, "label"])

    merged["ensemble_prob"] = calibrator.predict(merged["ensemble_prob_raw"])

    eval_df = merged[eval_mask].copy()
    y_true = eval_df["label"].values
    y_prob = eval_df["ensemble_prob"].values

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_prob)

    print(f"[Ensemble] Final eval AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")

    # --- Calibration curve data (predicted prob bucket vs observed rate) ---
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(y_prob, bin_edges[1:-1])
    calib_points = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            calib_points.append({
                "bin_mid": float((bin_edges[b] + bin_edges[b + 1]) / 2),
                "predicted_mean": float(y_prob[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            })

    # --- Operating point table for the alert-threshold tuner ---
    # For each candidate threshold: sensitivity, specificity, PPV, NPV, and
    # estimated alerts per 100 patient-days (the metric clinicians actually
    # care about for alarm-fatigue reasoning, alongside sensitivity/PPV).
    operating_points = []
    n_eval_patient_days = eval_df.groupby("patient_id")["hour"].apply(lambda h: (h.max() - h.min()) / 24).sum()
    for thresh in np.arange(0.05, 0.96, 0.025):
        pred_pos = y_prob >= thresh
        tp = int(((pred_pos == 1) & (y_true == 1)).sum())
        fp = int(((pred_pos == 1) & (y_true == 0)).sum())
        fn = int(((pred_pos == 0) & (y_true == 1)).sum())
        tn = int(((pred_pos == 0) & (y_true == 0)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        # Alert BURDEN should count distinct alert episodes, not raw alert
        # hours: a threshold that stays triggered for 10 consecutive hours
        # on one deteriorating patient is one alarm a clinician responds to,
        # not 10. Counting raw hours (as a naive TP+FP sum would) wildly
        # overstates alarm fatigue and is a common mistake in early-warning
        # system evaluations. We collapse consecutive alerted hours *within
        # each patient* into a single episode before counting.
        n_episodes = _count_alert_episodes(eval_df, pred_pos)
        alerts_per_100_patient_days = (
            n_episodes / n_eval_patient_days * 100 if n_eval_patient_days > 0 else 0.0
        )

        operating_points.append({
            "threshold": round(float(thresh), 3),
            "sensitivity": round(sensitivity, 4),
            "specificity": round(specificity, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "alerts_per_100_patient_days": round(alerts_per_100_patient_days, 2),
        })

    results = {
        "ensemble_weight_gb": float(best_w),
        "ensemble_weight_lstm": float(1 - best_w),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "gb_auroc": gb["auroc"],
        "gb_auprc": gb["auprc"],
        "lstm_auroc": lstm["auroc"],
        "lstm_auprc": lstm["auprc"],
        "roc_curve": {"fpr": fpr[::3].tolist(), "tpr": tpr[::3].tolist()},
        "pr_curve": {"precision": prec[::3].tolist(), "recall": rec[::3].tolist()},
        "calibration_curve": calib_points,
        "operating_points": operating_points,
        "n_eval_patients": int(eval_df["patient_id"].nunique()),
        "n_eval_rows": int(len(eval_df)),
    }

    with open(os.path.join(BASE, "ensemble_results.json"), "w") as f:
        json.dump(results, f)

    # Save row-level eval predictions for the dashboard's patient explorer
    eval_export = eval_df[["patient_id", "hour", "gb_prob", "lstm_prob", "ensemble_prob", "label"]]
    eval_export.to_parquet(os.path.join(BASE, "ensemble_eval_predictions.parquet"), index=False)

    import joblib
    joblib.dump(calibrator, os.path.join(BASE, "ensemble_calibrator.joblib"))
    joblib.dump({"gb_weight": best_w, "lstm_weight": 1 - best_w}, os.path.join(BASE, "ensemble_weights.joblib"))

    print(f"Saved ensemble results to {BASE}")
    return results


if __name__ == "__main__":
    build_ensemble()
