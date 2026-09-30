import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { buildSeries, duration, evaluations, featureMeta, flaws, pipeline, samples, scores, transcript, variants } from './data/mockData';
import './styles.css';

const features = Object.keys(featureMeta);
const navItems = [
  ['Overview', '/'], ['Analyze', '/analyze'], ['Dataset', '/dataset'], ['Evaluations', '/evaluations'], ['Methodology', '/methodology']
];

function formatTime(seconds) {
  const safe = Math.max(0, seconds);
  return `00:${safe.toFixed(2).padStart(5, '0')}`;
}

function routePath() { return window.location.pathname.replace(/\/$/, '') || '/'; }
function navigate(path) { window.history.pushState({}, '', path); window.dispatchEvent(new PopStateEvent('popstate')); }

function App() {
  const [path, setPath] = useState(routePath());
  const [selectedAnalysis, setSelectedAnalysis] = useState(samples[0]);
  const [selectedFeature, setSelectedFeature] = useState('Speech Rate');
  const [selectedFlaw, setSelectedFlaw] = useState(flaws[1]);
  const [currentTime, setCurrentTime] = useState(18.42);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedVariant, setSelectedVariant] = useState(0);
  const [config, setConfig] = useState({ thresholds: true, transcript: true, compact: false });

  useEffect(() => {
    const onPop = () => setPath(routePath());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  useEffect(() => {
    if (!isPlaying) return undefined;
    const timer = window.setInterval(() => {
      setCurrentTime((time) => {
        if (time >= duration) { setIsPlaying(false); return 0; }
        return Math.min(duration, time + 0.1);
      });
    }, 100);
    return () => window.clearInterval(timer);
  }, [isPlaying]);

  const selectFlaw = (flaw) => { setSelectedFlaw(flaw); setCurrentTime(flaw.start); };
  const openSample = (sample) => { setSelectedAnalysis(sample); navigate('/analyze'); };
  const page = path === '/analyze' ? <AnalyzePage {...{ selectedAnalysis, selectedFeature, setSelectedFeature, selectedFlaw, selectFlaw, currentTime, setCurrentTime, isPlaying, setIsPlaying, config }} />
    : path === '/dataset' ? <DatasetPage openSample={openSample} selectedVariant={selectedVariant} setSelectedVariant={setSelectedVariant} />
    : path === '/evaluations' ? <EvaluationsPage openSample={openSample} />
    : path === '/methodology' ? <MethodologyPage />
    : path === '/settings' ? <SettingsPage config={config} setConfig={setConfig} />
    : path === '/terms' ? <LegalPage type="terms" />
    : path === '/privacy' ? <LegalPage type="privacy" />
    : <OverviewPage {...{ selectedFeature, setSelectedFeature, selectedFlaw, selectFlaw, currentTime, setCurrentTime, isPlaying, setIsPlaying, selectedAnalysis, setSelectedAnalysis, openSample }} />;

  return <AppShell path={path} setPath={navigate}>{page}</AppShell>;
}

function AppShell({ path, setPath, children }) {
  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => setPath('/')} aria-label="Go to overview"><span className="brand-mark">C</span><span>CONTRASTIVE<br /><strong>SPEECH ANALYTICS</strong></span></button>
      <div className="topbar-meta"><span className="status-dot" /> LOCAL DEMONSTRATION <span className="divider" /> TRACK C / 2026</div>
      <button className="text-button" onClick={() => setPath('/settings')}>Settings</button>
    </header>
    <div className="shell-body">
      <aside className="sidebar" aria-label="Primary navigation">
        <div className="side-label">WORKSPACE</div>
        {navItems.map(([label, href]) => <button key={href} className={`nav-link ${path === href ? 'active' : ''}`} onClick={() => setPath(href)}><span className="nav-index">{String(navItems.findIndex((item) => item[1] === href) + 1).padStart(2, '0')}</span>{label}</button>)}
        <div className="side-label side-label-lower">UTILITY</div>
        <button className={`nav-link ${path === '/settings' ? 'active' : ''}`} onClick={() => setPath('/settings')}><span className="nav-index">06</span>Settings</button>
        <div className="sidebar-bottom"><div className="pipeline-mini"><span className="status-dot success" /> DEMO DATA ACTIVE</div><span className="muted">Mock services are isolated and replaceable.</span></div>
      </aside>
      <main className="main-content">{children}</main>
    </div>
    <Footer setPath={setPath} />
  </div>;
}

function Footer({ setPath }) { return <footer className="footer"><span>CONTRASTIVE SPEECH ANALYTICS / PROTOTYPE</span><nav><button onClick={() => setPath('/analyze')}>Analyze</button><button onClick={() => setPath('/dataset')}>Dataset</button><button onClick={() => setPath('/evaluations')}>Evaluations</button><button onClick={() => setPath('/methodology')}>Methodology</button><button onClick={() => setPath('/terms')}>Terms</button><button onClick={() => setPath('/privacy')}>Privacy</button></nav></footer>; }

function OverviewPage(props) {
  return <div className="page overview-page">
    <PageIntro eyebrow="CONTRASTIVE SPEECH ANALYTICS" title="Inspect the delivery, not just the transcript." description="Compare speech delivery against an aligned ideal, locate acoustic deviations in time, and inspect the evidence behind each evaluation." actions={<><button className="primary-button" onClick={() => navigate('/analyze')}>Analyze Speech <span>↗</span></button><button className="secondary-button" onClick={() => navigate('/dataset')}>Explore Dataset</button></>} />
    <div className="chain-strip"><span>DATA</span><b>→</b><span>ALIGN</span><b>→</b><span>ANALYZE</span><b>→</b><span>GROUND</span><b>→</b><span>EXPLAIN</span><b>→</b><span>SCORE</span></div>
    <AnalysisWorkspace compact={false} {...props} />
  </div>;
}

function PageIntro({ eyebrow, title, description, actions }) { return <section className="page-intro"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div><div className="intro-actions">{actions}</div></section>; }

function AnalyzePage({ selectedAnalysis, selectedFeature, setSelectedFeature, selectedFlaw, selectFlaw, currentTime, setCurrentTime, isPlaying, setIsPlaying, config }) {
  const [processing, setProcessing] = useState(false);
  return <div className="page analysis-page">
    <div className="analysis-head"><div><div className="eyebrow">ANALYSIS WORKSPACE / SAMPLE 001</div><h1>{selectedAnalysis.title}</h1><div className="identity-row"><span>BASELINE <strong>{selectedAnalysis.baseline}</strong></span><span>PARTICIPANT <strong>{selectedAnalysis.participant}</strong></span><span className="complete-label"><i /> ANALYSIS COMPLETE</span></div></div><div className="head-actions"><button className="secondary-button" onClick={() => setIsPlaying(!isPlaying)}>{isPlaying ? 'Pause' : 'Play'}</button><button className="secondary-button" onClick={() => setCurrentTime(0)}>Restart</button><button className="secondary-button" onClick={() => setProcessing(true)}>Export Report</button></div></div>
    {processing && <ProcessingBanner onDone={() => setProcessing(false)} />}
    <AnalysisWorkspace {...{ selectedFeature, setSelectedFeature, selectedFlaw, selectFlaw, currentTime, setCurrentTime, isPlaying, setIsPlaying, config }} />
  </div>;
}

function ProcessingBanner({ onDone }) { const [step, setStep] = useState(0); useEffect(() => { const timer = setInterval(() => setStep((current) => { if (current >= pipeline.length - 1) { clearInterval(timer); setTimeout(onDone, 500); return current; } return current + 1; }), 380); return () => clearInterval(timer); }, [onDone]); return <div className="processing-banner"><div><div className="eyebrow">MOCK PROCESSING / RUN-0142</div><strong>{pipeline[step]}</strong></div><div className="pipeline-row">{pipeline.map((item, index) => <span key={item} className={index < step ? 'done' : index === step ? 'current' : ''}>{String(index + 1).padStart(2, '0')} {item}</span>)}</div></div>; }

function AnalysisWorkspace({ selectedFeature, setSelectedFeature, selectedFlaw, selectFlaw, currentTime, setCurrentTime, isPlaying, setIsPlaying, config }) {
  return <section className="analysis-workspace">
    <div className="workspace-toolbar"><div className="toolbar-title"><span className="live-indicator" /> SYNCHRONIZED EVIDENCE VIEW</div><div className="toolbar-meta"><span>ALIGNMENT <strong>96%</strong></span><span>RUNTIME <strong>00:38.00</strong></span><span>RUBRIC <strong>v1.2</strong></span></div></div>
    <div className="workspace-grid">
      <aside className="finding-rail"><div className="panel-heading"><span>DETECTED FLAWS</span><span className="count-badge">03</span></div><div className="filter-row"><button className="filter-chip active">All</button><button className="filter-chip">High</button><button className="filter-chip">Medium</button></div>{flaws.map((flaw) => <FlawCard key={flaw.id} flaw={flaw} selected={selectedFlaw.id === flaw.id} onClick={() => selectFlaw(flaw)} />)}<div className="rail-note"><span className="mini-marker" /> Evidence regions are linked to aligned transcript units.</div></aside>
      <section className="evidence-canvas"><AudioPlayer {...{ currentTime, setCurrentTime, isPlaying, setIsPlaying }} /><Waveform {...{ currentTime, setCurrentTime, selectedFlaw, selectFlaw }} /><div className="feature-header"><div><div className="panel-heading">FEATURE COMPARISON</div><p className="muted">Participant and baseline values over aligned time.</p></div><select value={selectedFeature} onChange={(e) => setSelectedFeature(e.target.value)} aria-label="Select acoustic feature">{features.map((feature) => <option key={feature}>{feature}</option>)}</select></div><FeatureChart feature={selectedFeature} currentTime={currentTime} selectedFlaw={selectedFlaw} config={config} /><TranscriptViewer currentTime={currentTime} selectedFlaw={selectedFlaw} onSelectWord={(word) => setCurrentTime(word.start)} /></section>
      <EvidencePanel flaw={selectedFlaw} feature={selectedFeature} />
    </div>
    <ScorePanel />
  </section>;
}

function AudioPlayer({ currentTime, setCurrentTime, isPlaying, setIsPlaying }) { return <div className="audio-player"><button className="play-button" onClick={() => setIsPlaying(!isPlaying)} aria-label={isPlaying ? 'Pause audio' : 'Play audio'}>{isPlaying ? 'Ⅱ' : '▶'}</button><div className="audio-track"><div className="audio-progress" style={{ width: `${(currentTime / duration) * 100}%` }} /></div><span className="mono">{formatTime(currentTime)}</span><span className="muted mono">/ 00:38.00</span><input className="volume" type="range" min="0" max="1" step="0.05" defaultValue="0.72" aria-label="Volume" /><button className="text-button" onClick={() => setCurrentTime(0)}>RESTART</button></div>; }

function Waveform({ currentTime, setCurrentTime, selectedFlaw, selectFlaw }) { const bars = useMemo(() => Array.from({ length: 96 }, (_, i) => 18 + Math.abs(Math.sin(i * 0.38)) * 30 + Math.abs(Math.sin(i * 0.11)) * 17), []); const x = (currentTime / duration) * 100; return <div className="waveform-block"><div className="waveform-labels"><span><i className="legend-line baseline-line" /> BASELINE / IDEAL</span><span><i className="legend-line participant-line" /> PARTICIPANT</span></div><div className="waveform" role="slider" aria-label="Analysis timeline" aria-valuemin="0" aria-valuemax={duration} aria-valuenow={currentTime} tabIndex="0" onClick={(event) => { const rect = event.currentTarget.getBoundingClientRect(); setCurrentTime(((event.clientX - rect.left) / rect.width) * duration); }}><div className="wave-bars baseline-wave">{bars.map((height, i) => <span key={i} style={{ height: `${height}%` }} />)}</div><div className="wave-bars participant-wave">{bars.map((height, i) => <span key={i} style={{ height: `${Math.max(12, height * (i > 44 && i < 54 ? 1.25 : i > 78 && i < 84 ? 0.72 : 0.88))}%` }} />)}</div>{flaws.map((flaw) => <button key={flaw.id} className={`flaw-band ${selectedFlaw.id === flaw.id ? 'selected' : ''}`} style={{ left: `${(flaw.start / duration) * 100}%`, width: `${((flaw.end - flaw.start) / duration) * 100}%` }} onClick={(e) => { e.stopPropagation(); selectFlaw(flaw); }} aria-label={`Select ${flaw.type} from ${formatTime(flaw.start)} to ${formatTime(flaw.end)}`}><span>{flaw.id.replace('flaw-', '0')}</span></button>)}<div className="playhead" style={{ left: `${x}%` }} /></div><div className="time-scale">{[0, 5, 10, 15, 20, 25, 30, 38].map((tick) => <span key={tick} style={{ left: `${(tick / duration) * 100}%` }}>{formatTime(tick)}</span>)}</div></div>; }

function FeatureChart({ feature, currentTime, selectedFlaw, config }) { const series = useMemo(() => buildSeries(feature), [feature]); const path = (key) => series.map((point, i) => `${i ? 'L' : 'M'} ${(point.t / duration) * 100} ${100 - point[key] * 78}`).join(' '); const x = (currentTime / duration) * 100; const meta = featureMeta[feature]; return <div className="chart-wrap"><div className="chart-topline"><span className="chart-unit">{feature.toUpperCase()} / {meta.unit}</span><span className="chart-legend"><i className="legend-line baseline-line" /> Baseline <i className="legend-line participant-line" /> Participant <span className="threshold-key">threshold</span></span></div><div className="chart" aria-label={`${feature} comparison chart`}><div className="grid-lines"><span /><span /><span /><span /></div><svg viewBox="0 0 100 100" preserveAspectRatio="none"><path d={path('baseline')} className="chart-path baseline-path" /><path d={path('participant')} className="chart-path participant-path" /></svg><div className="chart-region" style={{ left: `${(selectedFlaw.start / duration) * 100}%`, width: `${((selectedFlaw.end - selectedFlaw.start) / duration) * 100}%` }} /><div className="chart-cursor" style={{ left: `${x}%` }} /></div><div className="chart-axis"><span>0.0</span><span>0.5</span><span>1.0</span></div><div className="chart-footer"><span>Selected region {formatTime(selectedFlaw.start)} - {formatTime(selectedFlaw.end)}</span><span>Threshold {meta.threshold}</span></div></div>; }

function TranscriptViewer({ currentTime, selectedFlaw, onSelectWord }) { return <div className="transcript-block"><div className="panel-heading">ALIGNED TRANSCRIPT <span className="muted">/ click a word to seek</span></div><div className="transcript-text">{transcript.map((word) => { const selected = word.start < selectedFlaw.end && word.end > selectedFlaw.start; const current = currentTime >= word.start && currentTime <= word.end; return <button key={word.id} className={`transcript-word ${selected ? 'selected' : ''} ${current ? 'current' : ''}`} onClick={() => onSelectWord(word)}>{word.word}</button>; })}</div></div>; }

function FlawCard({ flaw, selected, onClick }) { return <button className={`flaw-card ${selected ? 'selected' : ''}`} onClick={onClick}><div className="flaw-card-top"><span className={`severity ${flaw.severity.toLowerCase()}`}>{flaw.severity}</span><span className="mono">{formatTime(flaw.start)}</span></div><strong>{flaw.type}</strong><p>{flaw.explanation}</p><div className="flaw-card-meta"><span>{flaw.deviation}</span><span>{Math.round(flaw.confidence * 100)}% confidence</span></div></button>; }

function EvidencePanel({ flaw, feature }) { const meta = featureMeta[feature]; return <aside className="evidence-panel"><div className="panel-heading">SELECTED EVIDENCE <span className="selection-tag">{flaw.id.replace('flaw-', '0')}</span></div><div className="evidence-title"><span className="eyebrow">{flaw.feature.toUpperCase()}</span><h2>{flaw.type}</h2><div className="evidence-time mono">{formatTime(flaw.start)} - {formatTime(flaw.end)}</div></div><div className="affected-text"><span className="muted">AFFECTED TEXT</span><strong>“{flaw.affected}”</strong></div><div className="metric-table"><Metric label="Feature" value={flaw.feature} /><Metric label="Baseline" value={flaw.baseline} /><Metric label="Participant" value={flaw.participant} /><Metric label="Deviation" value={flaw.deviation} emphasis /><Metric label="Threshold" value={flaw.threshold} /><Metric label="Duration" value={`${(flaw.end - flaw.start).toFixed(2)} sec`} /><Metric label="Confidence" value={`${Math.round(flaw.confidence * 100)}%`} /></div><div className="severity-row"><span>SEVERITY</span><strong className={`severity ${flaw.severity.toLowerCase()}`}>{flaw.severity}</strong></div><div className="explanation"><div className="eyebrow">EXPLANATION</div><p>{flaw.explanation}</p><small>This is a measured acoustic or timing deviation. It does not infer why the speaker produced it.</small></div><div className="formula"><span>COMPARISON RULE</span><code>deviation = (participant - baseline) / baseline</code></div></aside>; }
function Metric({ label, value, emphasis }) { return <div className="metric"><span>{label}</span><strong className={emphasis ? 'metric-emphasis' : ''}>{value}</strong></div>; }

function ScorePanel() { return <section className="score-panel"><div className="score-overview"><div className="eyebrow">OVERALL EVALUATION</div><div className="score-number">82 <span>/ 100</span></div><p className="muted">Contrastive delivery score<br />Coverage: 94% / Rubric v1.2</p></div><div className="score-rows">{scores.map((score) => <div className="score-row" key={score.name}><div className="score-name"><strong>{score.name}</strong><span>{score.evidence}</span></div><span className="mono raw">{score.raw}</span><strong className="mono score-value">{score.score}</strong><span className="mono weight">{score.weight}</span><div className="score-bar"><span style={{ width: `${score.score}%` }} /></div><span className="mono contribution">+{score.contribution}</span></div>)}</div></section>; }

function DatasetPage({ openSample, selectedVariant, setSelectedVariant }) { return <div className="page"><PageIntro eyebrow="CONTRASTIVE DATASET" title="Paired delivery samples." description="Explore exact-transcript baseline and mirror recordings across a controlled flaw spectrum." actions={<button className="primary-button" onClick={() => navigate('/analyze')}>Analyze a sample <span>↗</span></button>} /><section className="dataset-section"><div className="section-heading"><div><div className="eyebrow">SAMPLE REGISTRY</div><h2>Validated contrastive pairs</h2></div><span className="muted mono">03 SAMPLES / SPEAKER-AWARE SPLIT</span></div><div className="dataset-table">{samples.map((sample) => <button className="dataset-row" key={sample.id} onClick={() => openSample(sample)}><span className="mono sample-id">{sample.id}</span><span><strong>{sample.title}</strong><small>Exact transcript pairing</small></span><span><strong>{sample.baseline}</strong><small>{sample.participant}</small></span><span><strong>{sample.flaw}</strong><small>{sample.severity} / {sample.score} score</small></span><span className="mono">{sample.status}</span><span>OPEN ↗</span></button>)}</div></section><section className="variant-section"><div className="section-heading"><div><div className="eyebrow">CONTROLLED FLAW SPECTRUM</div><h2>One text, six deliveries.</h2></div><span className="muted">Same transcript across quality variants</span></div><div className="variant-strip">{variants.map((variant, index) => <button key={variant} className={`variant ${selectedVariant === index ? 'selected' : ''}`} onClick={() => setSelectedVariant(index)}><span className="variant-index">0{index + 1}</span><strong>{variant}</strong><div className="variant-wave"><i style={{ width: `${28 + index * 11}%` }} /></div><small>{index === 0 ? 'Reference' : index === 1 ? 'Control' : `Deviation ${index * 18}%`}</small></button>)}</div><div className="variant-detail"><span className="eyebrow">SELECTED VARIANT / {variants[selectedVariant].toUpperCase()}</span><p>“Every generation changes the way we communicate and the next decade will reshape that conversation.”</p><span className="muted">Transcript ID T-014 · Baseline comparison remains aligned across all variants.</span></div></section></div>; }

function EvaluationsPage({ openSample }) { return <div className="page"><PageIntro eyebrow="EVALUATION HISTORY" title="Previous analyses." description="Review completed contrastive runs and reopen their evidence workspaces." actions={<button className="primary-button" onClick={() => navigate('/analyze')}>New analysis <span>↗</span></button>} /><section className="dataset-section"><div className="section-heading"><div><div className="eyebrow">RUN REGISTRY</div><h2>Recent evaluations</h2></div><span className="muted mono">LOCAL MOCK SERVICE</span></div><div className="evaluation-table"><div className="evaluation-header"><span>RECORDING</span><span>BASELINE</span><span>SCORE</span><span>FLAWS</span><span>DURATION</span><span>DATE</span><span>STATUS</span></div>{evaluations.map((evaluation, index) => <button className="evaluation-row" key={evaluation.recording} onClick={() => openSample(samples[index])}><span><strong>{evaluation.recording}</strong><small>run-014{index + 2}</small></span><span>{evaluation.baseline}</span><strong className="score-inline">{evaluation.score}</strong><span>{evaluation.flaws}</span><span className="mono">{evaluation.duration}</span><span className="mono">{evaluation.date}</span><span className="complete-label"><i /> {evaluation.status}</span></button>)}</div></section></div>; }

function MethodologyPage() { const stages = [['01', 'Contrastive Dataset', 'Exact transcript pairs provide an ideal reference and controlled mirror deliveries.'], ['02', 'Forced Alignment', 'Words and phonemes map to timestamps so comparison remains temporally grounded.'], ['03', 'Acoustic Features', 'FFT, MFCC, F0, energy, speech rate, pauses, and clarity proxies are extracted.'], ['04', 'Speaker Normalization', 'Relative comparison reduces penalties for natural pitch, energy, and timbre differences.'], ['05', 'Baseline Comparison', 'Participant values are compared with aligned baseline values and configured thresholds.'], ['06', 'Temporal Grounding', 'Each finding receives start, end, affected text, deviation, and confidence.'], ['07', 'Causal Explanation', 'Versioned templates translate measurable evidence into cautious human language.'], ['08', 'Rubric Scoring', 'Dimension scores combine configured evidence into a transparent overall result.']]; return <div className="page"><PageIntro eyebrow="METHOD / PIPELINE" title="From signal to evidence." description="The interface follows a reproducible sequence from paired audio through temporal grounding and configurable scoring." actions={<button className="secondary-button" onClick={() => navigate('/settings')}>View configuration</button>} /><section className="method-flow">{stages.map(([index, title, description], i) => <div className="method-row" key={title}><span className="method-index mono">{index}</span><div className="method-copy"><h2>{title}</h2><p>{description}</p></div><span className="method-arrow">{i < stages.length - 1 ? '↓' : '●'}</span></div>)}</section><div className="method-note"><div className="eyebrow">MODEL LIMITATION</div><p>The system reports measurable delivery differences. It does not infer emotion, intent, confidence, personality, or mental state from acoustic signals.</p></div></div>; }

function SettingsPage({ config, setConfig }) { return <div className="page"><PageIntro eyebrow="SETTINGS / CONFIGURATION" title="Make the rubric inspectable." description="These controls are backed by mock configuration today and are shaped for a future analysis API." actions={<button className="primary-button" onClick={() => alert('Configuration saved to the mock service.')}>Save configuration</button>} /><section className="settings-grid"><SettingGroup title="Rubric weights"><SettingRow label="Pacing" value="25%" /><SettingRow label="Pause control" value="20%" /><SettingRow label="Pitch dynamics" value="15%" /><SettingRow label="Energy" value="15%" /><SettingRow label="Vocal clarity" value="15%" /><SettingRow label="Consistency" value="10%" /></SettingGroup><SettingGroup title="Detection behavior"><Toggle label="Show threshold bands" checked={config.thresholds} onChange={() => setConfig({ ...config, thresholds: !config.thresholds })} /><Toggle label="Synchronized transcript" checked={config.transcript} onChange={() => setConfig({ ...config, transcript: !config.transcript })} /><Toggle label="Compact analytical layout" checked={config.compact} onChange={() => setConfig({ ...config, compact: !config.compact })} /></SettingGroup><SettingGroup title="Provenance"><SettingRow label="Rubric version" value="v1.2" /><SettingRow label="Alignment engine" value="mock-aligner 0.4" /><SettingRow label="Feature config" value="features-2026.09" /><SettingRow label="Data retention" value="session only" /></SettingGroup></section></div>; }
function SettingGroup({ title, children }) { return <div className="setting-group"><div className="panel-heading">{title}</div>{children}</div>; }
function SettingRow({ label, value }) { return <div className="setting-row"><span>{label}</span><strong className="mono">{value}</strong></div>; }
function Toggle({ label, checked, onChange }) { return <label className="toggle-row"><span>{label}</span><input type="checkbox" checked={checked} onChange={onChange} /><i className="toggle-ui" /></label>; }

function LegalPage({ type }) { const privacy = type === 'privacy'; return <div className="page legal-page"><div className="eyebrow">{privacy ? 'PRIVACY POLICY' : 'TERMS OF SERVICE'}</div><h1>{privacy ? 'Prototype data handling.' : 'Prototype terms.'}</h1><p className="lead">{privacy ? 'This prototype uses isolated mock data for the current demonstration. No uploaded audio is sent to a production analysis service from this interface.' : 'This prototype is provided for research demonstration and evaluation. It is not a certification tool, medical device, or substitute for a qualified coach or judge.'}</p><div className="legal-copy">{privacy ? <><h2>Audio and transcript handling</h2><p>The current demonstration uses deterministic mock recordings, transcripts, and analysis results. A future connected deployment must document upload processing, access controls, encryption, and deletion behavior before accepting real speech data.</p><h2>Analysis data</h2><p>Results shown here are generated from structured mock evidence. They should not be interpreted as a production evaluation of a speaker.</p><h2>Retention and controls</h2><p>Prototype data is session-scoped. Production retention, export, correction, and deletion controls remain implementation requirements for the backend.</p></> : <><h2>Prototype use</h2><p>Use this interface to inspect the product concept and controlled demonstration workflow. Do not upload sensitive personal data to an environment that has not been approved for that purpose.</p><h2>Evidence limitations</h2><p>Scores and findings are demonstrative. Acoustic deviations do not establish emotion, intent, personality, intelligence, or mental state.</p><h2>Changes</h2><p>These prototype terms may change as the connected product and data governance model are defined.</p></>}</div></div>; }

function SkeletonLoader() { return <div className="skeleton-loader"><span /><span /><span /></div>; }

createRoot(document.getElementById('root')).render(<App />);
