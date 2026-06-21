import React from 'react'

export default function Panel({ title, subtitle, action, children, style, bodyStyle }) {
  return (
    <div
      className="fade-in-up"
      style={{
        background: 'var(--bg-panel)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--shadow-panel)',
        overflow: 'hidden',
        ...style,
      }}
    >
      {(title || action) && (
        <div style={{
          display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between',
          padding: 'var(--space-4) var(--space-5)',
          borderBottom: '1px solid var(--border-subtle)',
          gap: 'var(--space-3)',
        }}>
          <div>
            {title && <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{title}</div>}
            {subtitle && <div style={{ fontSize: 12, color: 'var(--text-tertiary)', marginTop: 3 }}>{subtitle}</div>}
          </div>
          {action}
        </div>
      )}
      <div style={{ padding: 'var(--space-5)', ...bodyStyle }}>
        {children}
      </div>
    </div>
  )
}
