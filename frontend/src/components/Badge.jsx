import React from 'react'

const VARIANTS = {
  low: { color: 'var(--severity-low)', bg: 'var(--severity-low-bg)' },
  moderate: { color: 'var(--severity-moderate)', bg: 'var(--severity-moderate-bg)' },
  elevated: { color: 'var(--severity-elevated)', bg: 'var(--severity-elevated-bg)' },
  critical: { color: 'var(--severity-critical)', bg: 'var(--severity-critical-bg)' },
  neutral: { color: 'var(--text-secondary)', bg: 'var(--bg-inset)' },
  teal: { color: 'var(--accent-teal)', bg: 'var(--accent-teal-glow)' },
}

export default function Badge({ children, variant = 'neutral', dot = false }) {
  const v = VARIANTS[variant] || VARIANTS.neutral
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 9px',
      borderRadius: 999,
      fontSize: 11,
      fontWeight: 600,
      color: v.color,
      background: v.bg,
      fontFamily: 'var(--font-mono)',
      letterSpacing: '0.01em',
      whiteSpace: 'nowrap',
    }}>
      {dot && <span style={{ width: 5, height: 5, borderRadius: '50%', background: v.color }} />}
      {children}
    </span>
  )
}

export function riskVariant(prob) {
  if (prob >= 0.7) return 'critical'
  if (prob >= 0.4) return 'elevated'
  if (prob >= 0.2) return 'moderate'
  return 'low'
}

export function riskLabel(prob) {
  if (prob >= 0.7) return 'High Risk'
  if (prob >= 0.4) return 'Elevated'
  if (prob >= 0.2) return 'Moderate'
  return 'Low Risk'
}
