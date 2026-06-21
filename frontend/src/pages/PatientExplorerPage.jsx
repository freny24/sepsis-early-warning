import React, { useState, useEffect, useMemo } from 'react'
import { api } from '../api/client'
import Panel from '../components/Panel'
import Badge, { riskVariant, riskLabel } from '../components/Badge'
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Legend, ComposedChart,
} from 'recharts'

export default function PatientExplorerPage() {
  const [patients, setPatients] = useState([])
  const [selectedId, setSelectedId] = useState(null)
  const [timeseries, setTimeseries] = useState(null)
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.listPatients(true)
      .then((res) => {
        setPatients(res.patients)
        if (res.patients.length > 0) setSelectedId(res.patients[0].patient_id)
      })
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!selectedId) return
    setLoading(true)
    Promise.all([api.patientTimeseries(selectedId), api.patientPredictions(selectedId)])
      .then(([ts, preds]) => {
        setTimeseries(ts)
        setPredictions(preds)
        setLoading(false)
      })
      .catch((e) => { setError(e.message); setLoading(false) })
  }, [selectedId])

  const mergedData = useMemo(() => {
    if (!timeseries || !predictions) return []
    const predByHour = Object.fromEntries(predictions.predictions.map((p) => [p.hour, p]))
    return timeseries.timeseries.map((row) => ({
      ...row,
      gb_prob: predByHour[row.hour]?.gb_prob ?? null,
      lstm_prob: predByHour[row.hour]?.lstm_prob ?? null,
      ensemble_prob: predByHour[row.hour]?.ensemble_prob ?? null,
    }))
  }, [timeseries, predictions])

  const latestPrediction = mergedData.length > 0
    ? [...mergedData].reverse().find((d) => d.ensemble_prob !== null)
    : null

  if (error) return <div style={{ padding: 32, color: 'var(--severity-critical)' }}>Error: {error}</div>

  return (
    <div style={{ display: 'flex' }}>
      <PatientList
        patients={patients}
        selectedId={selectedId}
        onSelect={setSelectedId}
      />

      <div style={{ flex: 1, padding: 'var(--space-6) var(--space-8)', minWidth: 0 }}>
        {loading || !timeseries ? (
          <div style={{ color: 'var(--text-tertiary)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>Loading patient data…</div>
        ) : (
          <>
            <PatientHeader timeseries={timeseries} latestPrediction={latestPrediction} />

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)', marginTop: 'var(--space-5)' }}>
              <Panel
                title="Sepsis Risk Probability Over Time"
                subtitle="Calibrated ensemble probability vs. individual model components"
              >
                <RiskChart data={mergedData} onsetHour={timeseries.onset_hour} />
              </Panel>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-5)' }}>
                <Panel title="Vital Signs" subtitle="Heart rate, respiratory rate, MAP, SpO₂">
                  <VitalsChart data={mergedData} onsetHour={timeseries.onset_hour} />
                </Panel>
                <Panel title="SOFA Score Components" subtitle="Stacked organ-system sub-scores over time">
                  <SofaChart data={mergedData} onsetHour={timeseries.onset_hour} />
                </Panel>
              </div>

              <Panel title="Key Labs" subtitle="Lactate, WBC, creatinine, platelets — forward-filled between draws">
                <LabsChart data={mergedData} onsetHour={timeseries.onset_hour} />
              </Panel>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

function PatientList({ patients, selectedId, onSelect }) {
  return (
    <div style={{
      width: 280, flexShrink: 0, borderRight: '1px solid var(--border-subtle)',
      height: '100vh', overflowY: 'auto', position: 'sticky', top: 0,
      background: 'var(--bg-inset)',
    }}>
      <div style={{ padding: 'var(--space-5) var(--space-5) var(--space-3)' }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
          Evaluation Cohort
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-disabled)', marginTop: 4 }}>{patients.length} patients with model scores</div>
      </div>
      <div style={{ padding: '0 var(--space-3) var(--space-5)' }}>
        {patients.map((p) => {
          const isActive = p.patient_id === selectedId
          return (
            <button
              key={p.patient_id}
              onClick={() => onSelect(p.patient_id)}
              style={{
                width: '100%', textAlign: 'left', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '10px 12px', borderRadius: 'var(--radius-sm)', border: 'none', marginBottom: 2,
                background: isActive ? 'var(--bg-panel-raised)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              <div>
                <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', fontWeight: 600, color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)' }}>
                  {p.patient_id}
                </div>
                <div style={{ fontSize: 11, color: 'var(--text-disabled)', marginTop: 2 }}>
                  Age {p.age} · {p.los_hours}h stay
                </div>
              </div>
              {p.sepsis_label ? (
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--severity-critical)', flexShrink: 0 }} />
              ) : (
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: 'var(--severity-low)', flexShrink: 0 }} />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function PatientHeader({ timeseries, latestPrediction }) {
  const isSeptic = timeseries.trajectory !== 'stable'
  const prob = latestPrediction?.ensemble_prob ?? 0

  return (
    <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
      <div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <h1 style={{ fontSize: 22, fontFamily: 'var(--font-mono)' }}>{timeseries.patient_id}</h1>
          <Badge variant={isSeptic ? 'critical' : 'low'} dot>
            {isSeptic ? (timeseries.trajectory === 'sepsis_rapid_onset' ? 'Rapid Onset Sepsis' : 'Slow Onset Sepsis') : 'Stable'}
          </Badge>
        </div>
        <div style={{ fontSize: 13, color: 'var(--text-tertiary)', marginTop: 6 }}>
          {timeseries.los_hours}h ICU stay
          {timeseries.onset_hour != null && ` · Clinical sepsis recognition at hour ${Math.round(timeseries.onset_hour)}`}
        </div>
      </div>

      {latestPrediction && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 14,
          background: 'var(--bg-panel)', border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)', padding: '12px 18px',
        }}>
          <div>
            <div style={{ fontSize: 10.5, color: 'var(--text-tertiary)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Latest Risk Score (hr {latestPrediction.hour})
            </div>
            <div style={{ fontSize: 22, fontFamily: 'var(--font-mono)', fontWeight: 700, marginTop: 2 }}>
              {(prob * 100).toFixed(1)}%
            </div>
          </div>
          <Badge variant={riskVariant(prob)}>{riskLabel(prob)}</Badge>
        </div>
      )}
    </div>
  )
}

function tooltipStyle() {
  return {
    background: 'var(--bg-panel-raised)', border: '1px solid var(--border-default)',
    borderRadius: 8, fontSize: 12, fontFamily: 'var(--font-mono)', padding: '8px 10px',
  }
}

function RiskChart({ data, onsetHour }) {
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
        <defs>
          <linearGradient id="ensembleFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--model-ensemble)" stopOpacity={0.35} />
            <stop offset="100%" stopColor="var(--model-ensemble)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" vertical={false} />
        <XAxis dataKey="hour" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} label={{ value: 'ICU hour', position: 'insideBottom', offset: -2, fontSize: 11, fill: 'var(--text-tertiary)' }} />
        <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle()} formatter={(v, name) => [v != null ? `${(v * 100).toFixed(1)}%` : '—', name]} labelFormatter={(h) => `Hour ${h}`} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {onsetHour != null && (
          <ReferenceLine
            x={Math.round(onsetHour)}
            stroke="#e5484d"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            label={{ value: 'Onset', fill: '#e5484d', fontSize: 11, position: 'top' }}
          />
        )}
        <ReferenceLine y={0.5} stroke="var(--text-disabled)" strokeDasharray="2 4" />
        <Area type="monotone" dataKey="ensemble_prob" name="Ensemble" stroke="var(--model-ensemble)" strokeWidth={2} fill="url(#ensembleFill)" connectNulls dot={false} />
        <Line type="monotone" dataKey="gb_prob" name="Gradient Boosting" stroke="var(--model-gb)" strokeWidth={1.5} dot={false} connectNulls strokeOpacity={0.85} />
        <Line type="monotone" dataKey="lstm_prob" name="LSTM" stroke="var(--model-lstm)" strokeWidth={1.5} dot={false} connectNulls strokeOpacity={0.85} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

function VitalsChart({ data, onsetHour }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" vertical={false} />
        <XAxis dataKey="hour" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} />
        <YAxis tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle()} labelFormatter={(h) => `Hour ${h}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {onsetHour != null && (
          <ReferenceLine
            x={Math.round(onsetHour)}
            stroke="#e5484d"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            label={{ value: 'Onset', fill: '#e5484d', fontSize: 11, position: 'top' }}
          />
        )}
        <Line type="monotone" dataKey="heart_rate" name="HR (bpm)" stroke="#e5484d" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="resp_rate" name="RR (/min)" stroke="#5b8def" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="map" name="MAP (mmHg)" stroke="#d4a72c" strokeWidth={1.5} dot={false} />
        <Line type="monotone" dataKey="spo2" name="SpO₂ (%)" stroke="#2dd4bf" strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

const SOFA_COMPONENT_COLORS = {
  sofa_resp: '#5b8def',
  sofa_coag: '#c081e8',
  sofa_liver: '#d4a72c',
  sofa_cardio: '#e08a3c',
  sofa_renal: '#e5484d',
}

function SofaChart({ data, onsetHour }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" vertical={false} />
        <XAxis dataKey="hour" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} />
        <YAxis tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} domain={[0, 'dataMax + 1']} allowDecimals={false} />
        <Tooltip contentStyle={tooltipStyle()} labelFormatter={(h) => `Hour ${h}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {onsetHour != null && (
          <ReferenceLine
            x={Math.round(onsetHour)}
            stroke="#e5484d"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            label={{ value: 'Onset', fill: '#e5484d', fontSize: 11, position: 'top' }}
          />
        )}
        <Area type="monotone" dataKey="sofa_resp" name="Respiratory" stackId="sofa" stroke="none" fill={SOFA_COMPONENT_COLORS.sofa_resp} />
        <Area type="monotone" dataKey="sofa_coag" name="Coagulation" stackId="sofa" stroke="none" fill={SOFA_COMPONENT_COLORS.sofa_coag} />
        <Area type="monotone" dataKey="sofa_liver" name="Liver" stackId="sofa" stroke="none" fill={SOFA_COMPONENT_COLORS.sofa_liver} />
        <Area type="monotone" dataKey="sofa_cardio" name="Cardiovascular" stackId="sofa" stroke="none" fill={SOFA_COMPONENT_COLORS.sofa_cardio} />
        <Area type="monotone" dataKey="sofa_renal" name="Renal" stackId="sofa" stroke="none" fill={SOFA_COMPONENT_COLORS.sofa_renal} />
      </AreaChart>
    </ResponsiveContainer>
  )
}

function LabsChart({ data, onsetHour }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: -16, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 6" stroke="var(--border-subtle)" vertical={false} />
        <XAxis dataKey="hour" type="number" domain={['dataMin', 'dataMax']} tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={{ stroke: 'var(--border-default)' }} tickLine={false} />
        <YAxis yAxisId="left" tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} label={{ value: 'mmol/L · mg/dL · 10⁹/L', angle: -90, position: 'insideLeft', fontSize: 10, fill: 'var(--text-disabled)' }} />
        <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11, fill: 'var(--text-tertiary)' }} axisLine={false} tickLine={false} label={{ value: 'Platelets ×10⁹/L', angle: 90, position: 'insideRight', fontSize: 10, fill: 'var(--text-disabled)' }} />
        <Tooltip contentStyle={tooltipStyle()} labelFormatter={(h) => `Hour ${h}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {onsetHour != null && (
          <ReferenceLine
            yAxisId="left"
            x={Math.round(onsetHour)}
            stroke="#e5484d"
            strokeWidth={1.5}
            strokeDasharray="4 3"
            label={{ value: 'Onset', fill: '#e5484d', fontSize: 11, position: 'top' }}
          />
        )}
        <Line yAxisId="left" type="monotone" dataKey="lactate" name="Lactate (mmol/L)" stroke="#e5484d" strokeWidth={1.5} dot={false} />
        <Line yAxisId="left" type="monotone" dataKey="wbc" name="WBC (×10⁹/L)" stroke="#5b8def" strokeWidth={1.5} dot={false} />
        <Line yAxisId="left" type="monotone" dataKey="creatinine" name="Creatinine (mg/dL)" stroke="#d4a72c" strokeWidth={1.5} dot={false} />
        <Line yAxisId="right" type="monotone" dataKey="platelets" name="Platelets (×10⁹/L)" stroke="#c081e8" strokeWidth={1.5} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}
