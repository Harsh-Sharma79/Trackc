const duration = 38;

const transcript = [
  ['Every', 0.4, 0.8], ['generation', 0.82, 1.55], ['changes', 1.62, 2.12], ['the', 2.18, 2.36], ['way', 2.42, 2.72], ['we', 2.78, 2.96], ['communicate', 3.02, 3.88],
  ['and', 4.12, 4.35], ['the', 4.4, 4.58], ['next', 4.65, 4.98], ['decade', 5.04, 5.66], ['will', 5.74, 5.98], ['reshape', 6.04, 6.7], ['that', 6.76, 6.98], ['conversation', 7.04, 7.82],
  ['The', 8.7, 9.02], ['next', 9.08, 9.35], ['generation', 9.42, 10.2], ['of', 10.26, 10.42], ['communication', 10.48, 11.36], ['will', 11.42, 11.66], ['be', 11.72, 11.9], ['defined', 11.96, 12.58], ['by', 12.64, 12.88], ['clarity', 12.94, 13.48], ['and', 13.54, 13.76], ['care', 13.82, 14.1],
  ['When', 15.4, 15.78], ['we', 15.84, 16.02], ['speak', 16.08, 16.5], ['with', 16.56, 16.82], ['intention', 16.88, 17.62], ['we', 17.7, 17.88], ['make', 17.94, 18.3], ['space', 18.36, 18.86], ['for', 18.92, 19.12], ['better', 19.18, 19.62], ['ideas', 19.68, 20.08], ['to', 20.14, 20.3], ['travel', 20.36, 20.84],
  ['A', 22.1, 22.28], ['measured', 22.34, 22.92], ['voice', 22.98, 23.32], ['keeps', 23.38, 23.72], ['attention', 23.78, 24.46], ['close', 24.52, 24.9], ['to', 24.98, 25.16], ['the', 25.22, 25.4], ['meaning', 25.46, 26.02], ['inside', 26.08, 26.5], ['each', 26.56, 26.78], ['phrase', 26.84, 27.36],
  ['That', 28.7, 29.02], ['is', 29.08, 29.24], ['how', 29.3, 29.58], ['we', 29.64, 29.84], ['build', 29.9, 30.32], ['trust', 30.38, 30.82], ['one', 30.88, 31.12], ['conversation', 31.18, 32.02], ['at', 32.08, 32.22], ['a', 32.28, 32.42], ['time', 32.48, 32.82]
].map(([word, start, end], index) => ({ id: `word-${index}`, word, start, end }));

const samples = [
  { id: 'SAMPLE-001', title: 'Interpretive Reading', baseline: 'Ideal Delivery', participant: 'Evaluation Recording 001', score: 82, status: 'Validated', flaw: 'Pacing deviation', severity: 'High' },
  { id: 'SAMPLE-002', title: 'Persuasive Oratory', baseline: 'Ideal Delivery', participant: 'Evaluation Recording 002', score: 88, status: 'Validated', flaw: 'Pause deviation', severity: 'Medium' },
  { id: 'SAMPLE-003', title: 'Declamation', baseline: 'Ideal Delivery', participant: 'Evaluation Recording 003', score: 91, status: 'Validated', flaw: 'Energy deviation', severity: 'Low' }
];

const flaws = [
  { id: 'flaw-1', type: 'Pause deviation', feature: 'Pause Duration', start: 8.2, end: 10.4, severity: 'Medium', affected: 'The next generation', baseline: '0.72 sec', participant: '0.31 sec', deviation: '-57.0%', threshold: '-35%', confidence: 0.88, explanation: 'The pause shortened relative to the aligned baseline, reducing the separation before the next phrase.' },
  { id: 'flaw-2', type: 'Pacing deviation', feature: 'Speech Rate', start: 18.42, end: 21.1, severity: 'High', affected: 'better ideas to travel', baseline: '3.8 syllables/sec', participant: '5.1 syllables/sec', deviation: '+34.2%', threshold: '+20%', confidence: 0.91, explanation: 'Speech rate increased substantially relative to the aligned baseline within this segment.' },
  { id: 'flaw-3', type: 'Energy deviation', feature: 'Energy', start: 31.05, end: 33.2, severity: 'Low', affected: 'one conversation at a time', baseline: '0.42', participant: '0.35', deviation: '-16.7%', threshold: '-12%', confidence: 0.79, explanation: 'Energy dropped below the baseline contour across the closing phrase.' }
];

const featureMeta = {
  'Speech Rate': { unit: 'syllables/sec', baseline: 3.8, participant: 5.1, deviation: '+34.2%', threshold: '+20%', color: '#5cc8ff' },
  'Pitch / F0': { unit: 'Hz', baseline: '146–221', participant: '153–207', deviation: '-18.1% range', threshold: '±15%', color: '#70c7a4' },
  'Energy': { unit: 'normalized', baseline: '0.42', participant: '0.35', deviation: '-16.7%', threshold: '-12%', color: '#d08b62' },
  'Pause Duration': { unit: 'seconds', baseline: '0.72', participant: '0.31', deviation: '-57.0%', threshold: '-35%', color: '#b0a0d8' },
  'MFCC': { unit: 'distance', baseline: '0.18', participant: '0.27', deviation: '+50.0%', threshold: '+30%', color: '#93a6b8' },
  'Vocal Clarity': { unit: 'index', baseline: '0.84', participant: '0.77', deviation: '-8.3%', threshold: '-10%', color: '#8ab6d6' }
};

const scores = [
  { name: 'Pacing', raw: '0.68', score: 68, weight: '25%', contribution: '17.0', evidence: '2 grounded findings' },
  { name: 'Pause Control', raw: '0.82', score: 82, weight: '20%', contribution: '16.4', evidence: '1 grounded finding' },
  { name: 'Pitch Dynamics', raw: '0.76', score: 76, weight: '15%', contribution: '11.4', evidence: '92% voiced coverage' },
  { name: 'Energy', raw: '0.84', score: 84, weight: '15%', contribution: '12.6', evidence: '1 low-severity finding' },
  { name: 'Vocal Clarity', raw: '0.79', score: 79, weight: '15%', contribution: '11.9', evidence: 'MFCC evidence valid' },
  { name: 'Delivery Consistency', raw: '0.72', score: 72, weight: '10%', contribution: '7.2', evidence: '3 aligned regions' }
];

const evaluations = [
  { recording: 'Evaluation Recording 001', baseline: 'Ideal Delivery', score: 82, flaws: 3, duration: '00:38', date: '2026-09-29', status: 'Complete' },
  { recording: 'Evaluation Recording 002', baseline: 'Ideal Delivery', score: 88, flaws: 1, duration: '00:34', date: '2026-09-28', status: 'Complete' },
  { recording: 'Evaluation Recording 003', baseline: 'Ideal Delivery', score: 91, flaws: 1, duration: '00:33', date: '2026-09-26', status: 'Complete' }
];

const pipeline = ['Audio Validation', 'Transcript Alignment', 'Feature Extraction', 'Speaker Normalization', 'Baseline Comparison', 'Temporal Grounding', 'Explanation', 'Scoring'];
const variants = ['Ideal', 'Near Perfect', 'Mild', 'Moderate', 'Strong', 'Severe'];

function signalValue(t, feature, participant = false) {
  const base = Math.sin(t * 0.54) * 0.19 + Math.sin(t * 1.3) * 0.08 + 0.52;
  const pacing = t >= 18.42 && t <= 21.1 ? (participant ? 0.23 : 0.02) : 0;
  const pause = t >= 8.2 && t <= 10.4 ? (participant ? -0.18 : 0.04) : 0;
  const energy = t >= 31.05 && t <= 33.2 ? (participant ? -0.16 : 0.04) : 0;
  const pitch = t >= 15.4 && t <= 17.6 ? (participant ? -0.12 : 0) : 0;
  if (feature === 'Speech Rate') return Math.max(0.12, 0.46 + Math.sin(t * 0.5) * 0.08 + pacing + pause * 0.1);
  if (feature === 'Pitch / F0') return Math.max(0.12, 0.52 + Math.sin(t * 0.35) * 0.15 + pitch);
  if (feature === 'Energy') return Math.max(0.1, base + energy);
  if (feature === 'Pause Duration') return Math.max(0.08, 0.46 + Math.sin(t * 0.72) * 0.1 + pause);
  if (feature === 'MFCC') return Math.max(0.1, 0.4 + Math.sin(t * 0.2) * 0.12 + pacing * 0.35);
  return Math.max(0.12, 0.62 + Math.sin(t * 0.4) * 0.12 + energy * 0.35);
}

function buildSeries(feature) {
  return Array.from({ length: 77 }, (_, index) => {
    const t = (index / 76) * duration;
    return { t, baseline: signalValue(t, feature, false), participant: signalValue(t, feature, true) };
  });
}

export { duration, transcript, samples, flaws, featureMeta, scores, evaluations, pipeline, variants, buildSeries };
