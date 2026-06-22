const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api'

async function getJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API error ${res.status} on ${path}: ${body}`)
  }
  return res.json()
}

export const api = {
  health: () => getJSON('/api/health'),
  cohortSummary: () => getJSON('/api/cohort/summary'),
  listPatients: (onlyWithPredictions = true) =>
    getJSON(`/api/patients?only_with_predictions=${onlyWithPredictions}`),
  patientTimeseries: (patientId) => getJSON(`/api/patients/${patientId}/timeseries`),
  patientPredictions: (patientId) => getJSON(`/api/patients/${patientId}/predictions`),
  modelPerformance: () => getJSON('/api/model/performance'),
  featureImportance: () => getJSON('/api/model/feature-importance'),
  operatingPoints: () => getJSON('/api/model/operating-points'),
  operatingPointAt: (threshold) => getJSON(`/api/model/operating-point?threshold=${threshold}`),
}