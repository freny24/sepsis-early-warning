import React, { useState, useEffect } from 'react'
import { api } from './api/client'
import Sidebar from './components/Sidebar'
import OverviewPage from './pages/OverviewPage'
import PatientExplorerPage from './pages/PatientExplorerPage'
import ModelPerformancePage from './pages/ModelPerformancePage'
import AlertTuningPage from './pages/AlertTuningPage'
import './styles/index.css'

const PAGES = {
  overview: { label: 'Cohort Overview', component: OverviewPage },
  patient: { label: 'Patient Explorer', component: PatientExplorerPage },
  model: { label: 'Model Performance', component: ModelPerformancePage },
  alerts: { label: 'Alert Threshold Tuning', component: AlertTuningPage },
}

export default function App() {
  const [activePage, setActivePage] = useState('overview')
  const [apiStatus, setApiStatus] = useState('checking')

  useEffect(() => {
    api.health()
      .then(() => setApiStatus('ok'))
      .catch(() => setApiStatus('error'))
  }, [])

  const ActiveComponent = PAGES[activePage].component

  if (apiStatus === 'checking') {
    return (
      <div style={{
        minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', fontSize: 14,
      }}>
        Connecting to API…
      </div>
    )
  }

  if (apiStatus === 'error') {
    return (
      <div style={{
        minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center',
        justifyContent: 'center', gap: 12, color: 'var(--text-secondary)', padding: 24, textAlign: 'center',
      }}>
        <div style={{ fontSize: 15, color: 'var(--severity-critical)', fontWeight: 600 }}>
          Could not reach the backend API
        </div>
        <div style={{ fontSize: 13, fontFamily: 'var(--font-mono)', maxWidth: 480, lineHeight: 1.6 }}>
          Make sure the FastAPI server is running:<br />
          <code style={{ color: 'var(--accent-teal)' }}>
            cd backend &amp;&amp; uvicorn api.app:app --reload --port 8000
          </code>
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <Sidebar pages={PAGES} activePage={activePage} onNavigate={setActivePage} />
      <main style={{ flex: 1, minWidth: 0 }}>
        <ActiveComponent />
      </main>
    </div>
  )
}
