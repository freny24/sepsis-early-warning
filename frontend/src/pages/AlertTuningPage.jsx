import React, { useState, useEffect, useMemo } from 'react'
import { api } from '../api/client'
import Panel from '../components/Panel'
import StatCard from '../components/StatCard'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend, ComposedChart, Bar,
} from 'recharts'

export default function AlertTuningPage() {
  const [points, setPoints] = useState(null)
  const [threshold, setThreshold] = useState(0.5)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.operatingPoints()
      .then((res) => setPoints(res.operating_points))
      .catch((e) => setError(e.message))
  }, [])

  const current = useMemo(() => {
    if (!points) return null
    return points.reduce((best, p) =>
      Math.abs(p.threshold - threshold) < Math.abs(best.threshold - threshold) ? p : best
    , points[0])
  }, [points, threshold])

  if (error) return <div style={{ padding: 32, color: 'var(--severity-critical)' }}>Error: {error}</div>
  if (!points) return <div style={{ padding: 32, color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>Loading operating points…</div>

  return (
    <div style={{ padding: 'var(--space-8)' }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent-teal)', letterSpacing: '0.06em', marginBottom: 8 }}>
        CLINICAL DECISION SUPPORT
      </div>
      <h1 style={{ fontSize: 26 }}>Alert Threshold Tuning</h1>
      <p style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 8, maxWidth: 720, lineHeight: 1.6 }}>
        Every early-warning system trades sensitivity against alert volume. A lower threshold catches
        more true sepsis cases earlier but fires more false alarms — a real cost in a unit already
        fighting alarm fatigue. Drag the slider to see how the operating point shifts.
      </p>

      <Panel style={{ marginTop: 'var(--space-6)' }} bodyStyle={{ padding: 'var(--space-6)' }}>
        <ThresholdSlider threshold={threshold} setThreshold={setThreshold} />

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 'var(--space-3)', marginTop: 'var(--space-6)' }}>
          <StatCard label="Sensitivity" value={(current.sensitivity * 100).toFixed(1)} unit="%" sublabel="True positive rate" accentColor="var(--severity-critical)" />
          <StatCard label="Specificity" value={(current.specificity * 100).toFixed(1)} unit="%" sublabel="True negative rate" accentColor="var(--model-gb)" />
          <StatCard label="PPV / Precision" value={(current.ppv * 100).toFixed(1)} unit="%" sublabel="Alert → really septic" accentColor="var(--accent-teal)" />
          <StatCard label="NPV" value={(current.npv * 100).toFixed(1)} unit="%" sublabel="No alert → really stable" accentColor="var(--severity-low)" />
          <StatCard label="Alert Burden" value={current.alerts_per_100_patient_days.toFixed(0)} unit="/100 pt-days" sublabel="Alarm fatigue proxy" accentColor="var(--severity-moderate)" />
        </div>
      </Panel>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-5)', marginTop: 'var(--space-5)' }}>
        <Panel title="Sensitivity vs. Specificity" subtitle="Classic operating-point tradeoff curve">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={points} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" />
              <XAxis dataKey="threshold" tickFormatter={(v) => v.toFixed(1)} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} label={{ value: 'Alert threshold', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={tooltipStyle()}
                formatter={(v) => `${(v * 100).toFixed(1)}%`}
                labelFormatter={(t) => `Threshold ${t.toFixed(2)}`}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine x={current.threshold} stroke="var(--accent-teal)" strokeWidth={1.5} />
              <Line type="monotone" dataKey="sensitivity" name="Sensitivity" stroke="var(--severity-critical)" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="specificity" name="Specificity" stroke="var(--model-gb)" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>

        <Panel title="Precision vs. Alert Burden" subtitle="How alarm volume scales as the threshold relaxes">
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={points} margin={{ top: 8, right: 16, left: -16, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" />
              <XAxis dataKey="threshold" tickFormatter={(v) => v.toFixed(1)} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} label={{ value: 'Alert threshold', position: 'insideBottom', offset: -4, fontSize: 11, fill: 'var(--text-tertiary)' }} />
              <YAxis yAxisId="left" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle()} labelFormatter={(t) => `Threshold ${t.toFixed(2)}`} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <ReferenceLine yAxisId="left" x={current.threshold} stroke="var(--accent-teal)" strokeWidth={1.5} />
              <Bar yAxisId="right" dataKey="alerts_per_100_patient_days" name="Alerts / 100 pt-days" fill="var(--severity-moderate)" fillOpacity={0.35} radius={[3, 3, 0, 0]} />
              <Line yAxisId="left" type="monotone" dataKey="ppv" name="Precision (PPV)" stroke="var(--accent-teal)" strokeWidth={2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      <Panel title="Confusion Matrix at Current Threshold" style={{ marginTop: 'var(--space-5)' }}>
        <ConfusionMatrix current={current} />
      </Panel>
    </div>
  )
}

function tooltipStyle() {
  return {
    background: 'var(--bg-panel-raised)', border: '1px solid var(--border-default)',
    borderRadius: 8, fontSize: 12, fontFamily: 'var(--font-mono)', padding: '8px 10px',
  }
}

function ThresholdSlider({ threshold, setThreshold }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 14 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-secondary)' }}>Alert Threshold</span>
        <span style={{ fontSize: 24, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-teal)' }}>
          {threshold.toFixed(2)}
        </span>
      </div>
      <input
        type="range"
        min={0.05}
        max={0.95}
        step={0.025}
        value={threshold}
        onChange={(e) => setThreshold(parseFloat(e.target.value))}
        style={{
          width: '100%',
          accentColor: 'var(--accent-teal)',
          height: 6,
          cursor: 'pointer',
        }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8 }}>
        <span style={{ fontSize: 11, color: 'var(--text-disabled)' }}>0.05 — High sensitivity, more alerts</span>
        <span style={{ fontSize: 11, color: 'var(--text-disabled)' }}>0.95 — High precision, fewer alerts</span>
      </div>
    </div>
  )
}

function ConfusionMatrix({ current }) {
  const cellStyle = (bg) => ({
    padding: '20px 16px',
    borderRadius: 'var(--radius-sm)',
    background: bg,
    textAlign: 'center',
  })

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: 10, maxWidth: 560 }}>
      <div />
      <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-tertiary)', fontWeight: 600, alignSelf: 'end', paddingBottom: 6 }}>
        ACTUAL: SEPSIS
      </div>
      <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-tertiary)', fontWeight: 600, alignSelf: 'end', paddingBottom: 6 }}>
        ACTUAL: STABLE
      </div>

      <div style={{ display: 'flex', alignItems: 'center', fontSize: 11, color: 'var(--text-tertiary)', fontWeight: 600, writingMode: 'vertical-rl', transform: 'rotate(180deg)', justifyContent: 'center' }}>
        PREDICTED: ALERT
      </div>
      <div style={cellStyle('var(--severity-low-bg)')}>
        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--severity-low)' }}>{current.tp}</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>True Positive</div>
      </div>
      <div style={cellStyle('var(--severity-critical-bg)')}>
        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--severity-critical)' }}>{current.fp}</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>False Positive</div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', fontSize: 11, color: 'var(--text-tertiary)', fontWeight: 600, writingMode: 'vertical-rl', transform: 'rotate(180deg)', justifyContent: 'center' }}>
        PREDICTED: NO ALERT
      </div>
      <div style={cellStyle('var(--severity-critical-bg)')}>
        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--severity-critical)' }}>{current.fn}</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>False Negative</div>
      </div>
      <div style={cellStyle('var(--severity-low-bg)')}>
        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--severity-low)' }}>{current.tn}</div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>True Negative</div>
      </div>
    </div>
  )
}
