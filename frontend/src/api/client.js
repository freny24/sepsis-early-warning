const BASE_URL = '/api'

async function getJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API error ${res.status} on ${path}: ${body}`)
  }
  return res.json()
}

export const api = {
  health: () => getJSON('/health'),
  cohortSummary: () => getJSON('/cohort/summary'),
  listPatients: (onlyWithPredictions = true) =>
    getJSON(`/patients?only_with_predictions=${onlyWithPredictions}`),
  patientTimeseries: (patientId) => getJSON(`/patients/${patientId}/timeseries`),
  patientPredictions: (patientId) => getJSON(`/patients/${patientId}/predictions`),
  modelPerformance: () => getJSON('/model/performance'),
  featureImportance: () => getJSON('/model/feature-importance'),
  operatingPoints: () => getJSON('/model/operating-points'),
  operatingPointAt: (threshold) => getJSON(`/model/operating-point?threshold=${threshold}`),
}
