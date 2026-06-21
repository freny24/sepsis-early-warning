import React from 'react'

export default function StatCard({ label, value, unit, trend, accentColor = 'var(--accent-teal)', sublabel }) {
  return (
    <div className="fade-in-up" style={{
      background: 'var(--bg-panel)',
      border: '1px solid var(--border-subtle)',
      borderRadius: 'var(--radius-md)',
      padding: 'var(--space-4) var(--space-5)',
      position: 'relative',
      overflow: 'hidden',
    }}>
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: 2,
        background: accentColor, opacity: 0.7,
      }} />
      <div style={{
        fontSize: 11, fontWeight: 600, color: 'var(--text-tertiary)',
        textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 10,
      }}>
        {label}
      </div>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
        <span style={{ fontSize: 28, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', letterSpacing: '-0.02em' }}>
          {value}
        </span>
        {unit && <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>{unit}</span>}
      </div>
      {sublabel && (
        <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 6 }}>{sublabel}</div>
      )}
      {trend && (
        <div style={{ fontSize: 12, marginTop: 6, color: trend.positive ? 'var(--severity-low)' : 'var(--severity-critical)' }}>
          {trend.label}
        </div>
      )}
    </div>
  )
}
