# Frontend — Sepsis Early Warning Dashboard

React + Vite + Recharts dashboard for the Sepsis Early Warning System backend.

## Setup

```bash
npm install
npm run dev
```

Runs at `http://localhost:5173`. Requires the backend API running at `http://localhost:8000` (see `vite.config.js` for the proxy setup — `/api/*` requests are forwarded there automatically in dev).

## Pages

- **Cohort Overview** — synthetic cohort summary, trajectory breakdown, model comparison
- **Patient Explorer** — per-patient vitals, labs, SOFA components, and model risk scores over time
- **Model Performance** — ROC/PR curves, calibration plot, LSTM training loss, GB feature importance
- **Alert Threshold Tuning** — interactive sensitivity/specificity/PPV/alert-burden tradeoff explorer

## Build for production

```bash
npm run build
```

Outputs to `dist/`. Note: in production you'll want to point API calls at a real deployed backend URL instead of relying on the Vite dev proxy — update `src/api/client.js`'s `BASE_URL` accordingly.

## Stack

- React 18
- Vite 5
- Recharts 2 (all charts — ROC/PR curves, calibration scatter, SOFA stacked areas, threshold tradeoff lines)
- No CSS framework — design tokens are plain CSS custom properties in `src/styles/index.css`
