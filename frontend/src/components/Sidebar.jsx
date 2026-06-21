import React from 'react'

const NAV_ICONS = {
  overview: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="9" rx="1.5" />
      <rect x="14" y="3" width="7" height="5" rx="1.5" />
      <rect x="14" y="12" width="7" height="9" rx="1.5" />
      <rect x="3" y="16" width="7" height="5" rx="1.5" />
    </svg>
  ),
  patient: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 12h4l2-7 4 14 2-7h6" />
    </svg>
  ),
  model: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 3" />
    </svg>
  ),
  alerts: (
    <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.3 3.3a2 2 0 0 1 3.4 0l8 13.86A2 2 0 0 1 20 20H4a2 2 0 0 1-1.7-2.84z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="16.5" x2="12.01" y2="16.5" />
    </svg>
  ),
}

export default function Sidebar({ pages, activePage, onNavigate }) {
  return (
    <aside style={{
      width: 248,
      flexShrink: 0,
      background: 'var(--bg-panel)',
      borderRight: '1px solid var(--border-subtle)',
      display: 'flex',
      flexDirection: 'column',
      padding: 'var(--space-5) var(--space-3)',
      position: 'sticky',
      top: 0,
      height: '100vh',
    }}>
      <div style={{ padding: '0 var(--space-3) var(--space-6)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: 'linear-gradient(135deg, var(--accent-teal), var(--accent-teal-dim))',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 0 0 1px rgba(45,212,191,0.3), 0 4px 12px -2px rgba(45,212,191,0.3)',
          }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#06120f" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12h4l2-7 4 14 2-7h6" />
            </svg>
          </div>
          <div style={{ fontSize: 14.5, fontWeight: 700, letterSpacing: '-0.01em', lineHeight: 1.15 }}>
            Sepsis Early<br />Warning System
          </div>
        </div>
        <div style={{
          fontSize: 10.5, fontFamily: 'var(--font-mono)', color: 'var(--text-tertiary)',
          marginTop: 10, display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <span className="pulse-dot" style={{
            width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-teal)', display: 'inline-block',
          }} />
          DEMO · SYNTHETIC DATA
        </div>
      </div>

      <nav style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {Object.entries(pages).map(([key, { label }]) => {
          const isActive = key === activePage
          return (
            <button
              key={key}
              onClick={() => onNavigate(key)}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '9px 12px',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: isActive ? 'var(--bg-panel-raised)' : 'transparent',
                color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                fontSize: 13.5,
                fontWeight: isActive ? 600 : 500,
                textAlign: 'left',
                cursor: 'pointer',
                transition: 'background 0.15s, color 0.15s',
                position: 'relative',
              }}
              onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--bg-inset)' }}
              onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
            >
              {isActive && (
                <span style={{
                  position: 'absolute', left: -12, top: '50%', transform: 'translateY(-50%)',
                  width: 3, height: 16, borderRadius: 2, background: 'var(--accent-teal)',
                }} />
              )}
              <span style={{ color: isActive ? 'var(--accent-teal)' : 'var(--text-tertiary)', display: 'flex' }}>
                {NAV_ICONS[key]}
              </span>
              {label}
            </button>
          )
        })}
      </nav>

      <div style={{ marginTop: 'auto', padding: 'var(--space-3)' }}>
        <div style={{
          fontSize: 10.5, color: 'var(--text-disabled)', lineHeight: 1.6,
          borderTop: '1px solid var(--border-subtle)', paddingTop: 'var(--space-3)',
        }}>
          Sepsis-3 inspired SOFA scoring · Gradient Boosting + LSTM ensemble · Not for clinical use
        </div>
      </div>
    </aside>
  )
}
