"""
LSTM Sequence Model
======================

A small LSTM that consumes the full per-patient hourly sequence (vitals +
labs + SOFA, NOT the delta/rolling features which are GB-specific -- the
LSTM should learn temporal patterns itself rather than be fed pre-computed
trend features) and predicts sepsis-within-horizon at every hour.

This is the half of the ensemble responsible for catching patterns that
unfold *across* time -- e.g. a specific shape of decline -- that a row-
independent tabular model can't see directly, even with delta features,
because delta features only look a few fixed windows back rather than
learning an arbitrary temporal representation.

Padding/masking: patients have variable-length stays, so sequences are
padded to the cohort max length and a mask is used so padded steps don't
contribute to the loss.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")

SEQ_FEATURE_COLS = [
    "heart_rate", "resp_rate", "temp_c", "sbp", "dbp", "map", "spo2",
    "wbc", "lactate", "creatinine", "platelets", "bilirubin", "procalcitonin",
    "sofa_coag", "sofa_liver", "sofa_cardio", "sofa_renal", "sofa_resp", "sofa_total",
]

torch.manual_seed(42)
np.random.seed(42)


class SepsisLSTM(nn.Module):
    def __init__(self, n_features, hidden_size=48, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features) -> per-timestep logits
        out, _ = self.lstm(x)            # (batch, seq_len, hidden)
        logits = self.head(out).squeeze(-1)  # (batch, seq_len)
        return logits


def build_sequences(df, feature_cols, max_len=None):
    patient_ids = df["patient_id"].unique()
    if max_len is None:
        max_len = df.groupby("patient_id").size().max()

    n_feat = len(feature_cols)
    X = np.zeros((len(patient_ids), max_len, n_feat), dtype=np.float32)
    Y = np.zeros((len(patient_ids), max_len), dtype=np.float32)
    M = np.zeros((len(patient_ids), max_len), dtype=np.float32)  # mask: 1 = real timestep

    for i, pid in enumerate(patient_ids):
        pdf = df[df["patient_id"] == pid].sort_values("hour")
        n = min(len(pdf), max_len)
        X[i, :n, :] = pdf[feature_cols].values[:n]
        Y[i, :n] = pdf["label"].values[:n]
        M[i, :n] = 1.0

    return X, Y, M, patient_ids, max_len


def train_lstm_model(epochs=18, batch_size=16, lr=1e-3):
    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Scale features using train-set statistics only (no leakage)
    scaler = StandardScaler()
    scaler.fit(train_df[SEQ_FEATURE_COLS])
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[SEQ_FEATURE_COLS] = scaler.transform(train_df[SEQ_FEATURE_COLS])
    test_df_scaled[SEQ_FEATURE_COLS] = scaler.transform(test_df[SEQ_FEATURE_COLS])

    max_len = int(df.groupby("patient_id").size().max())
    X_train, Y_train, M_train, _, _ = build_sequences(train_df_scaled, SEQ_FEATURE_COLS, max_len)
    X_test, Y_test, M_test, test_pids, _ = build_sequences(test_df_scaled, SEQ_FEATURE_COLS, max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SepsisLSTM(n_features=len(SEQ_FEATURE_COLS)).to(device)

    # Positive class weighting in the loss to address imbalance, same spirit as scale_pos_weight in GB
    pos_rate = Y_train[M_train == 1].mean()
    pos_weight_val = (1 - pos_rate) / max(pos_rate, 1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_train_t = torch.tensor(X_train, device=device)
    Y_train_t = torch.tensor(Y_train, device=device)
    M_train_t = torch.tensor(M_train, device=device)

    n_samples = X_train_t.shape[0]
    train_losses = []

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb, mb = X_train_t[idx], Y_train_t[idx], M_train_t[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss_raw = criterion(logits, yb)
            # weight positives, mask padded steps
            weights = torch.where(yb == 1, pos_weight_val, 1.0) * mb
            loss = (loss_raw * weights).sum() / weights.sum().clamp(min=1.0)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(1, n_batches)
        train_losses.append(avg_loss)
        print(f"[LSTM] Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, device=device)
        logits_test = model(X_test_t)
        probs_test = torch.sigmoid(logits_test).cpu().numpy()

    mask_flat = M_test.astype(bool)
    y_true = Y_test[mask_flat]
    y_prob = probs_test[mask_flat]

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    print(f"[LSTM] Test AUROC: {auroc:.3f} | AUPRC: {auprc:.3f}")

    # Save model + scaler + results
    torch.save(model.state_dict(), os.path.join(BASE, "lstm_model.pt"))
    import joblib
    joblib.dump(scaler, os.path.join(BASE, "lstm_scaler.joblib"))
    joblib.dump({"max_len": max_len, "feature_cols": SEQ_FEATURE_COLS}, os.path.join(BASE, "lstm_meta.joblib"))

    # Flatten per-(patient,hour) predictions for downstream ensembling
    rows = []
    for i, pid in enumerate(test_pids):
        n_valid = int(M_test[i].sum())
        pdf = test_df[test_df["patient_id"] == pid].sort_values("hour").reset_index(drop=True)
        for h in range(n_valid):
            rows.append({
                "patient_id": pid,
                "hour": int(pdf.loc[h, "hour"]),
                "lstm_prob": float(probs_test[i, h]),
                "label": int(Y_test[i, h]),
            })
    lstm_preds_df = pd.DataFrame(rows)
    lstm_preds_df.to_parquet(os.path.join(BASE, "lstm_test_predictions.parquet"), index=False)

    results = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "train_losses": train_losses,
        "roc_curve": {"fpr": fpr[::4].tolist(), "tpr": tpr[::4].tolist()},
        "pr_curve": {"precision": prec[::4].tolist(), "recall": rec[::4].tolist()},
    }
    with open(os.path.join(BASE, "lstm_results.json"), "w") as f:
        json.dump(results, f)

    print(f"Saved LSTM model + results to {BASE}")
    return results


if __name__ == "__main__":
    train_lstm_model()
