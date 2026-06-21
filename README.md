# Sepsis Early Warning System

A portfolio project demonstrating an end-to-end clinical ML pipeline: synthetic ICU time series → SOFA-based feature engineering → a Gradient Boosting + LSTM ensemble → probability calibration → an interactive alert-threshold dashboard.

**This is a demo built on synthetic data. It is not validated for clinical use and should not inform real patient care.**

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-61dafb) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688) ![XGBoost](https://img.shields.io/badge/XGBoost-gradient_boosting-orange) ![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-ee4c2c)

---

## Why this project

Sepsis early-warning is one of the more honest demonstrations of applied clinical ML: the signal is real (deteriorating vitals precede clinical recognition by hours), the label is messy (clinician-assigned, not a clean biological event), the data is sparse and irregularly sampled (labs every few hours, not continuously), and the evaluation has to be *patient-level*, not row-level, or the numbers lie to you. This project tries to get those details right rather than just hitting a high AUROC.

## What's actually in here

| Layer | What it does |
|---|---|
| `backend/data/synthetic_patients.py` | Generates hourly vitals + sparsely-sampled labs for a synthetic ICU cohort, with independent per-channel measurement noise and confounding non-septic deterioration (so the task isn't trivially separable) |
| `backend/data/sofa_score.py` | Computes Sepsis-3-style SOFA organ-dysfunction sub-scores from vitals/labs |
| `backend/data/feature_engineering.py` | Builds delta-vitals (1/4/8h), rolling mean/std (6h), shock index, and the early-warning label (sepsis within *N* hours, not at the moment of recognition) |
| `backend/models/train_gb.py` | XGBoost on engineered tabular features, with patient-level train/test split and isotonic calibration |
| `backend/models/train_lstm.py` | A small LSTM over raw per-hour sequences (masked for variable-length stays), so the model learns its own temporal representation instead of relying solely on hand-built deltas |
| `backend/models/ensemble.py` | Weighted blend of GB + LSTM (weight chosen by AUPRC grid search), re-calibrated, plus the full sensitivity/specificity/PPV/alert-burden operating-point table |
| `backend/api/` | FastAPI serving all of the above to the dashboard |
| `frontend/` | React + Recharts dashboard: cohort overview, per-patient explorer, model performance, interactive alert-threshold tuner |

## Results

Evaluated on a held-out set of patients with **no overlap with training** (patient-level split, not row-level — splitting individual hours of the same patient across train/test is a common and serious leakage bug in clinical time series work):

| Model | AUROC | AUPRC |
|---|---|---|
| Gradient Boosting | 0.970 | 0.940 |
| LSTM | 0.964 | 0.938 |
| **Ensemble (calibrated)** | **0.952** | **0.908** |

The ensemble's reported numbers are slightly *lower* than either individual model — that's intentional, not a bug. GB and LSTM are each evaluated on test patients used to fit their own calibrators; the ensemble is then evaluated on a further-held-out half of those patients specifically so its reported performance isn't inflated by any leftover calibration leakage. It's the most honest number in the table.

At a 0.5 alert threshold: **85.2% sensitivity, 97.6% specificity, 91.0% precision**, and roughly **10 alert episodes per 100 patient-days** (consecutive alerted hours are collapsed into one episode, since a 6-hour-long alarm is one event a clinician responds to, not six).

### On the AUROC being "too good"

Early synthetic-data drafts of this produced AUROC ≈ 0.99, which is a red flag for any clinical ML task — real sepsis early-warning systems in the literature land around 0.75–0.88 AUROC. The cause was a too-clean synthetic data generator (every vital driven by the same deterministic ramp variable). The fix, visible in `synthetic_patients.py`, was adding independent per-channel measurement noise, severity overlap between mild and severe septic patients, and confounding non-septic deterioration (post-op tachycardia, transient fevers) in ~45% of stable patients. That's what brought performance into a believable range — worth knowing if you build something similar and your first AUROC looks suspiciously perfect.

## Running it

### Backend

```bash
cd backend
pip install -r requirements.txt
cd ..
./run_pipeline.sh          # regenerates data + retrains everything (~2-5 min)
# or, if models are already trained:
./run_pipeline.sh --api    # just starts the API
```

API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard runs at `http://localhost:5173` (proxies `/api` to the backend).

## Project structure

```
sepsis-early-warning/
├── backend/
│   ├── data/
│   │   ├── synthetic_patients.py     # cohort generator
│   │   ├── sofa_score.py             # SOFA sub-scores
│   │   └── feature_engineering.py    # deltas, rolling stats, labels
│   ├── models/
│   │   ├── train_gb.py               # XGBoost + isotonic calibration
│   │   ├── train_lstm.py             # PyTorch LSTM
│   │   └── ensemble.py               # weighted blend + operating points
│   ├── api/
│   │   └── app.py                    # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                    # Overview, Patient Explorer, Model Performance, Alert Tuning
│   │   ├── components/               # Panel, StatCard, Badge, Sidebar
│   │   └── api/client.js
│   └── package.json
├── run_pipeline.sh
└── README.md
```

## Design decisions worth knowing about (for interviews)

- **Patient-level splits everywhere.** Every train/test, train/calibration, and final-eval split groups by `patient_id`. Row-level splits on time series data leak future information into training and silently inflate every metric.
- **Why an ensemble at all.** GB on engineered tabular features is fast and captures threshold/interaction effects well (e.g. "lactate high AND rising"). The LSTM sees the raw sequence and can pick up shape-based patterns the hand-built delta windows miss. A simple weighted average (not a learned stacker) was used deliberately — with only ~300 synthetic patients, a stacked meta-learner would be prone to overfitting; a weighted average is transparent and easy to defend.
- **Why calibrate at all, twice.** Tree ensembles are well known to produce systematically over/under-confident probabilities. Each base model gets its own isotonic calibrator fit on held-out data, and the *combination* gets a third calibration pass, since blending two differently-shaped probability distributions produces a new distribution that isn't automatically well-calibrated just because its inputs were.
- **Why AUPRC alongside AUROC.** At ~17% positive rate, AUROC can look deceptively good while precision at any usable operating point is still poor. AUPRC is more sensitive to that.
- **Alert burden, not alert count.** A threshold that stays triggered for 10 consecutive hours on one patient is one alarm a clinician responds to, not ten. The dashboard's "alert burden" metric collapses consecutive alerted hours into episodes before counting — counting raw alert-hours instead (an easy mistake) overstated alarm fatigue by ~50x in an earlier version of this project.
