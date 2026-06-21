import React, { useState, useEffect } from 'react'
import { api } from '../api/client'
import Panel from '../components/Panel'
import StatCard from '../components/StatCard'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts'

const TRAJECTORY_COLORS = {
  stable: '#3b9e7a',
  sepsis_slow_onset: '#d4a72c',
  sepsis_rapid_onset: '#e5484d',
}
const TRAJECTORY_LABELS = {
  stable: 'Stable (no sepsis)',
  sepsis_slow_onset: 'Sepsis — Slow Onset',
  sepsis_rapid_onset: 'Sepsis — Rapid Onset',
}

function PageHeader() {
  return (
    <div style={{ padding: 'var(--space-8) var(--space-8) var(--space-6)' }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-teal)', letterSpacing: '0.06em', marginBottom: 8 }}>
        PORTFOLIO PROJECT · CLINICAL ML
      </div>
      <h1 style={{ fontSize: 26, color: 'var(--text-primary)' }}>Cohort Overview</h1>
      <p style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 8, maxWidth: 680, lineHeight: 1.6 }}>
        A synthetic ICU cohort used to demonstrate an early sepsis-warning pipeline: SOFA-based
        feature engineering, a Gradient Boosting + LSTM ensemble, probability calibration, and an
        interactive alert-threshold tuner. All patient data below is generated, not real.
      </p>
    </div>
  )
}

export default function OverviewPage() {
  const [summary, setSummary] = useState(null)
  const [perf, setPerf] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    Promise.all([api.cohortSummary(), api.modelPerformance()])
      .then(([s, p]) => { setSummary(s); setPerf(p) })
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <ErrorState message={error} />
  if (!summary || !perf) return <LoadingState />

  const trajectoryData = Object.entries(summary.trajectory_counts).map(([key, value]) => ({
    name: TRAJECTORY_LABELS[key] || key,
    key,
    value,
  }))

  return (
    <div>
      <PageHeader />

      <div style={{ padding: '0 var(--space-8) var(--space-8)', display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>

        {/* Headline stats */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-4)' }}>
          <StatCard
            label="Total Patients"
            value={summary.n_patients}
            sublabel={`Median LOS: ${summary.median_los_hours.toFixed(0)}h`}
            accentColor="var(--accent-teal)"
          />
          <StatCard
            label="Sepsis Cases"
            value={summary.n_sepsis}
            unit={`/ ${summary.n_patients}`}
            sublabel={`${(summary.sepsis_prevalence * 100).toFixed(1)}% prevalence`}
            accentColor="var(--severity-critical)"
          />
          <StatCard
            label="Ensemble AUROC"
            value={perf.ensemble.auroc.toFixed(3)}
            sublabel="Held-out evaluation set"
            accentColor="var(--model-ensemble)"
          />
          <StatCard
            label="Ensemble AUPRC"
            value={perf.ensemble.auprc.toFixed(3)}
            sublabel="More informative under class imbalance"
            accentColor="var(--model-ensemble)"
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.4fr', gap: 'var(--space-5)' }}>
          <Panel title="Patient Trajectories" subtitle="Distribution across the synthetic cohort">
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-5)' }}>
              <div style={{ width: 160, height: 160, flexShrink: 0 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={trajectoryData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={48}
                      outerRadius={76}
                      paddingAngle={3}
                      strokeWidth={0}
                    >
                      {trajectoryData.map((entry) => (
                        <Cell key={entry.key} fill={TRAJECTORY_COLORS[entry.key]} />
                      ))}
                    </Pie>
                    <Tooltip
                      contentStyle={{
                        background: 'var(--bg-panel-raised)', border: '1px solid var(--border-default)',
                        borderRadius: 8, fontSize: 12, fontFamily: 'var(--font-mono)',
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)', flex: 1 }}>
                {trajectoryData.map((d) => (
                  <div key={d.key} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ width: 9, height: 9, borderRadius: 3, background: TRAJECTORY_COLORS[d.key], flexShrink: 0 }} />
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', flex: 1 }}>{d.name}</span>
                    <span style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {d.value}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </Panel>

          <Panel title="Model Comparison" subtitle="AUROC / AUPRC by model on the held-out evaluation set">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart
                data={[
                  { name: 'Gradient Boosting', auroc: perf.gradient_boosting.auroc, auprc: perf.gradient_boosting.auprc },
                  { name: 'LSTM', auroc: perf.lstm.auroc, auprc: perf.lstm.auprc },
                  { name: 'Ensemble', auroc: perf.ensemble.auroc, auprc: perf.ensemble.auprc },
                ]}
                barGap={6}
                margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    background: 'var(--bg-panel-raised)', border: '1px solid var(--border-default)',
                    borderRadius: 8, fontSize: 12, fontFamily: 'var(--font-mono)',
                  }}
                  formatter={(v) => v.toFixed(3)}
                />
                <Bar dataKey="auroc" name="AUROC" fill="var(--model-gb)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="auprc" name="AUPRC" fill="var(--accent-teal)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Panel>
        </div>

        <Panel title="Pipeline Summary" subtitle="What's actually happening under the hood">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-5)' }}>
            <PipelineStep
              n="01"
              title="Synthetic Cohort"
              desc="Hourly vitals + sparsely-sampled labs generated with independent measurement noise and confounding non-septic deterioration, so the task isn't trivially separable."
            />
            <PipelineStep
              n="02"
              title="SOFA + Feature Engineering"
              desc="Sepsis-3-style SOFA sub-scores, delta vitals over 1/4/8h windows, 6h rolling mean/std, and shock index."
            />
            <PipelineStep
              n="03"
              title="GB + LSTM Ensemble"
              desc="XGBoost on tabular engineered features blended with an LSTM over raw sequences; weight chosen by AUPRC grid search."
            />
            <PipelineStep
              n="04"
              title="Calibration + Thresholding"
              desc="Isotonic regression calibrates the blended score; an operating-point table drives the interactive alert-threshold tuner."
            />
          </div>
        </Panel>
      </div>
    </div>
  )
}

function PipelineStep({ n, title, desc }) {
  return (
    <div>
      <div style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-teal)', fontWeight: 700, marginBottom: 8 }}>
        {n}
      </div>
      <div style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 12.5, color: 'var(--text-tertiary)', lineHeight: 1.6 }}>{desc}</div>
    </div>
  )
}

function LoadingState() {
  return (
    <div style={{ padding: 'var(--space-8)', color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
      Loading cohort data…
    </div>
  )
}

function ErrorState({ message }) {
  return (
    <div style={{ padding: 'var(--space-8)', color: 'var(--severity-critical)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
      Error: {message}
    </div>
  )
}
