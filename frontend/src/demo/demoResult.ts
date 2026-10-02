import type { EvaluationResult } from '../types/evaluation';

/** Development-only fixture. Production mode must use src/lib/api.ts. */
export const demoResult: EvaluationResult = {
  id: 'demo-sample-001',
  status: 'complete',
  overall_score: 82,
  summary: 'Delivery is strongest in pitch dynamics and clarity. The main opportunity is pacing around the central phrase.',
  duration_seconds: 38,
  participant_audio: { name: 'Evaluation Recording 001', duration_seconds: 38, format: 'demo/wav' },
  reference_audio: { name: 'Ideal Delivery', duration_seconds: 38, format: 'demo/wav' },
  evaluated_at: '2026-10-02T16:00:00Z',
  transcript: [
    { word: 'Every', start: 0.4, end: 0.8 },
    { word: 'generation', start: 0.82, end: 1.55 },
    { word: 'changes', start: 1.62, end: 2.12 },
    { word: 'the', start: 2.18, end: 2.36 },
    { word: 'way', start: 2.42, end: 2.72 },
    { word: 'we', start: 2.78, end: 2.96 },
    { word: 'communicate', start: 3.02, end: 3.88 },
    { word: 'better', start: 19.18, end: 19.62 },
    { word: 'ideas', start: 19.68, end: 20.08 },
    { word: 'to', start: 20.14, end: 20.3 },
    { word: 'travel', start: 20.36, end: 20.84 }
  ],
  flaws: [
    { id: 'demo-pause', type: 'Pause deviation', start: 8.2, end: 10.4, severity: 'medium', affected_text: 'The next generation', feature: 'Pause Duration', confidence: 0.88, explanation: 'The pause shortened relative to the aligned baseline, reducing separation before the next phrase.' },
    { id: 'demo-pace', type: 'Pacing deviation', start: 18.42, end: 21.1, severity: 'high', affected_text: 'better ideas to travel', feature: 'Speech Rate', confidence: 0.91, explanation: 'Speech rate increased substantially relative to the aligned baseline within this segment.' },
    { id: 'demo-energy', type: 'Energy deviation', start: 31.05, end: 33.2, severity: 'low', affected_text: 'one conversation at a time', feature: 'Energy', confidence: 0.79, explanation: 'Energy dropped below the reference contour across the closing phrase.' }
  ],
  rubric_scores: [
    { name: 'Pace', score: 68, weight: 0.25, status: 'attention', explanation: 'Two time-grounded pacing findings.' },
    { name: 'Pitch', score: 76, weight: 0.25, status: 'healthy', explanation: 'Voiced coverage is sufficient for comparison.' },
    { name: 'Energy', score: 84, weight: 0.25, status: 'healthy', explanation: 'One low-severity energy region.' },
    { name: 'Pause Pattern', score: 82, weight: 0.25, status: 'attention', explanation: 'One shortened pause is above threshold.' }
  ],
  notes: ['Demo data is enabled with VITE_DEMO_MODE=true.', 'Some flaw categories are experimental and should be validated against backend performance before production use.'],
  metadata: { source: 'frontend-demo', api_contract: 'EvaluationResult v1' }
};
