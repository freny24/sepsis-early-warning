import React, { useState, useEffect } from 'react'
import { api } from '../api/client'
import Panel from '../components/Panel'
import StatCard from '../components/StatCard'
import {
  LineChart, Line, ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend, Cell,
} from 'recharts'

function tooltipStyle() {
  return {
    background: 'var(--bg-panel-raised)', border: '1px solid var(--border-default)',
    borderRadius: 8, fontSize: 12, fontFamily: 'var(--font-mono)', padding: '8px 10px',
  }
}

export default function ModelPerformancePage() {
  const [perf, setPerf] = useState(null)
  const [featImp, setFeatImp] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.modelPerformance(), api.featureImportance()])
      .then(([p, f]) => { setPerf(p); setFeatImp(f) })
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <div style={{ padding: 32, color: 'var(--severity-critical)' }}>Error: {error}</div>
  if (!perf || !featImp) return <div style={{ padding: 32, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>Loading model results…</div>

  // Build combined ROC data for overlay
  const rocData = buildOverlayCurve(
    [
      { key: 'gb', curve: perf.gradient_boosting.roc_curve, xKey: 'fpr', yKey: 'tpr' },
      { key: 'lstm', curve: perf.lstm.roc_curve, xKey: 'fpr', yKey: 'tpr' },
      { key: 'ensemble', curve: perf.ensemble.roc_curve, xKey: 'fpr', yKey: 'tpr' },
    ],
    'fpr'
  )

  const prData = buildOverlayCurve(
    [
      { key: 'gb', curve: perf.gradient_boosting.pr_curve, xKey: 'recall', yKey: 'precision' },
      { key: 'lstm', curve: perf.lstm.pr_curve, xKey: 'recall', yKey: 'precision' },
      { key: 'ensemble', curve: perf.ensemble.pr_curve, xKey: 'recall', yKey: 'precision' },
    ],
    'recall'
  )

  const featureData = Object.entries(featImp.top_features)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)
    .map(([name, importance]) => ({ name: prettifyFeatureName(name), importance }))

  const lossData = perf.lstm.train_losses.map((loss, i) => ({ epoch: i + 1, loss }))

  return (
    <div style={{ padding: 'var(--space-8)' }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-teal)', letterSpacing: '0.06em', marginBottom: 8 }}>
        EVALUATION
      </div>
      <h1 style={{ fontSize: 26 }}>Model Performance</h1>
      <p style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 8, maxWidth: 680, lineHeight: 1.6 }}>
        All metrics computed on a held-out evaluation split with patient-level separation from training
        (no patient appears in both train and test) to avoid the leakage that's common when splitting
        time series by row instead of by patient.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)', marginTop: 'var(--space-6)' }}>
        <StatCard label="GB AUROC" value={perf.gradient_boosting.auroc.toFixed(3)} accentColor="var(--model-gb)" />
        <StatCard label="LSTM AUROC" value={perf.lstm.auroc.toFixed(3)} accentColor="var(--model-lstm)" />
        <StatCard label="Ensemble AUROC" value={perf.ensemble.auroc.toFixed(3)} accentColor="var(--model-ensemble)" />
        <StatCard
          label="Ensemble Weights"
          value={`${(perf.ensemble.weight_gb * 100).toFixed(0)}/${(perf.ensemble.weight_lstm * 100).toFixed(0)}`}
          unit="GB/LSTM"
          accentColor="var(--accent-teal)"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-5)', marginTop: 'var(--space-5)' }}>
        <Panel title="ROC Curves" subtitle="True positive rate vs. false positive rate">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={rocData} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <Tooltip contentStyle={tooltipStyle()} formatter={(v) => v?.toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-disabled)" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="gb" name="Gradient Boosting" stroke="var(--model-gb)" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="lstm" name="LSTM" stroke="var(--model-lstm)" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="ensemble" name="Ensemble" stroke="var(--model-ensemble)" strokeWidth={2.5} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Precision-Recall Curves" subtitle="More informative than ROC given ~17% positive rate">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={prData} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" />
              <XAxis dataKey="x" type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} label={{ value: 'Recall', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} label={{ value: 'Precision', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <Tooltip contentStyle={tooltipStyle()} formatter={(v) => v?.toFixed(3)} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="gb" name="Gradient Boosting" stroke="var(--model-gb)" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="lstm" name="LSTM" stroke="var(--model-lstm)" strokeWidth={2} dot={false} connectNulls />
              <Line type="monotone" dataKey="ensemble" name="Ensemble" stroke="var(--model-ensemble)" strokeWidth={2.5} dot={false} connectNulls />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-5)', marginTop: 'var(--space-5)' }}>
        <Panel title="Calibration Plot" subtitle="Predicted probability vs. observed event rate, post isotonic calibration">
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" />
              <XAxis
                dataKey="predicted_mean" type="number" domain={[0, 1]} name="Predicted"
                tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false}
                label={{ value: 'Mean Predicted Probability', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-tertiary)' }}
              />
              <YAxis
                dataKey="observed_rate" type="number" domain={[0, 1]} name="Observed"
                tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false}
                label={{ value: 'Observed Event Rate', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-tertiary)' }}
              />
              <Tooltip
                contentStyle={tooltipStyle()}
                formatter={(v, name) => [typeof v === 'number' ? v.toFixed(3) : v, name]}
                cursor={{ strokeDasharray: '3 3' }}
              />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-disabled)" strokeDasharray="4 4" />
              <Scatter data={perf.ensemble.calibration_curve} fill="var(--accent-teal)">
                {perf.ensemble.calibration_curve.map((_, i) => (
                  <Cell key={i} fill="var(--accent-teal)" />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
          <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 8, lineHeight: 1.6 }}>
            Points close to the diagonal mean predicted probabilities can be trusted at face value —
            e.g. among hours scored ~30% risk, roughly 30% actually went on to meet sepsis criteria.
          </div>
        </Panel>

        <Panel title="LSTM Training Loss" subtitle="Weighted BCE loss per epoch (masked for padded timesteps)">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={lossData} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" vertical={false} />
              <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle()} formatter={(v) => v.toFixed(4)} />
              <Line type="monotone" dataKey="loss" name="Train loss" stroke="var(--model-lstm)" strokeWidth={2} dot={{ r: 2.5 }} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <div style={{ marginTop: 'var(--space-5)' }}>
        <Panel title="Top Features (Gradient Boosting)" subtitle="By XGBoost gain-based importance">
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={featureData} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} />
              <YAxis dataKey="name" type="category" width={190} tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle()} formatter={(v) => v.toFixed(3)} />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {featureData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? 'var(--accent-teal)' : 'var(--model-gb)'} fillOpacity={i === 0 ? 1 : 0.65 - i * 0.03} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>
    </div>
  )
}

function buildOverlayCurve(series, xKey) {
  // Merge multiple (x[], y[]) curves into one array keyed by a shared x-grid
  // for recharts overlay rendering. Uses nearest-point lookup per series.
  const xGrid = Array.from(new Set(series.flatMap((s) => s.curve[xKey].map((v) => Math.round(v * 100) / 100)))).sort((a, b) => a - b)

  return xGrid.map((x) => {
    const row = { x }
    series.forEach((s) => {
      const xs = s.curve[xKey]
      const ys = s.curve[s.yKey]
      let bestIdx = 0
      let bestDist = Infinity
      for (let i = 0; i < xs.length; i++) {
        const d = Math.abs(xs[i] - x)
        if (d < bestDist) { bestDist = d; bestIdx = i }
      }
      row[s.key] = ys[bestIdx]
    })
    return row
  })
}

function prettifyFeatureName(name) {
  return name
    .replace(/_/g, ' ')
    .replace(/\bsofa\b/gi, 'SOFA')
    .replace(/\bhr\b/gi, 'HR')
    .replace(/\bwbc\b/gi, 'WBC')
    .replace(/\d+h\b/g, (m) => m)
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
