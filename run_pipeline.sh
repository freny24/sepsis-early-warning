#!/usr/bin/env bash
# Sepsis Early Warning System -- full pipeline runner
#
# Usage:
#   ./run_pipeline.sh        # regenerate data + retrain all models from scratch
#   ./run_pipeline.sh --api  # just start the API (assumes models already trained)
#
set -e

cd "$(dirname "$0")"

if [ "$1" == "--api" ]; then
    echo "Starting API only (skipping data generation + training)..."
else
    echo "==> [1/6] Generating synthetic patient cohort..."
    python3 backend/data/synthetic_patients.py

    echo "==> [2/6] Computing SOFA scores..."
    python3 backend/data/sofa_score.py

    echo "==> [3/6] Engineering features + labels..."
    python3 backend/data/feature_engineering.py

    echo "==> [4/6] Training Gradient Boosting model..."
    python3 backend/models/train_gb.py

    echo "==> [5/6] Training LSTM model..."
    python3 backend/models/train_lstm.py

    echo "==> [6/6] Building calibrated ensemble..."
    python3 backend/models/ensemble.py
fi

echo ""
echo "==> Starting API server on http://localhost:8000"
echo "    (In another terminal, run: cd frontend && npm install && npm run dev)"
echo ""
uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000
