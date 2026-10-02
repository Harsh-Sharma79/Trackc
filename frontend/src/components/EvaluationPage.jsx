import { useEffect, useState } from 'react';
import { getCapabilities } from '../lib/api';
import { useEvaluation } from '../hooks/useEvaluation';

function prettyTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  return `${Math.floor(value / 60).toString().padStart(2, '0')}:${(value % 60).toFixed(2).padStart(5, '0')}`;
}

function displayWordCount(text) {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

export default function EvaluationPage() {
  const [participantAudio, setParticipantAudio] = useState(null);
  const [referenceAudio, setReferenceAudio] = useState(null);
  const [transcript, setTranscript] = useState('');
  const [capabilities, setCapabilities] = useState(null);
  const [capabilityError, setCapabilityError] = useState('');
  const { result, status, error, runEvaluation } = useEvaluation();

  useEffect(() => {
    let active = true;
    getCapabilities()
      .then((value) => { if (active) setCapabilities(value); })
      .catch((cause) => { if (active) setCapabilityError(cause.message || 'Could not connect to the evaluation API.'); });
    return () => { active = false; };
  }, []);

  async function submit(event) {
    event.preventDefault();
    if (!participantAudio || !transcript.trim()) return;
    try {
      await runEvaluation({
        participant_audio: participantAudio,
        reference_audio: referenceAudio || undefined,
        transcript: transcript.trim(),
      });
    } catch {
      // The hook stores a readable error for the form to render.
    }
  }

  function downloadResult() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${result.id || 'speech-evaluation'}-result.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const transcriptText = typeof result?.transcript === 'string'
    ? result.transcript
    : result?.transcript?.map((word) => word.word).join(' ') || '';
  const score = Number(result?.overall_score ?? 0);

  return <div className="page real-evaluation-page">
    <section className="page-intro evaluation-intro">
      <div><div className="eyebrow">LIVE SPEECH EVALUATION / BACKEND CONNECTED</div><h1>Evaluate a recording.</h1><p>Upload a participant recording and an optional ideal reference. The backend extracts speech features, scores delivery, and returns time-grounded findings.</p></div>
      <div className="api-health-pill"><span className={`status-dot ${capabilities ? 'success' : ''}`} />{capabilities ? 'API READY' : capabilityError ? 'API UNAVAILABLE' : 'CHECKING API'}</div>
    </section>

    <div className="evaluation-layout">
      <form className="evaluation-form" onSubmit={submit}>
        <div className="panel-heading">INPUT RECORDINGS</div>
        <label className="upload-field"><span>PARTICIPANT AUDIO <b>REQUIRED</b></span><input type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg" required onChange={(event) => setParticipantAudio(event.target.files?.[0] || null)} /><small>{participantAudio ? `${participantAudio.name} · ${(participantAudio.size / (1024 * 1024)).toFixed(1)} MB` : 'Choose the speaker recording to score.'}</small></label>
        <label className="upload-field"><span>IDEAL REFERENCE <b>OPTIONAL</b></span><input type="file" accept="audio/*,.wav,.mp3,.flac,.m4a,.ogg" onChange={(event) => setReferenceAudio(event.target.files?.[0] || null)} /><small>{referenceAudio ? `${referenceAudio.name} · ${(referenceAudio.size / (1024 * 1024)).toFixed(1)} MB` : 'Add a recording of the same text for contrastive scoring.'}</small></label>
        <label className="transcript-field"><span>TRANSCRIPT <b>REQUIRED</b></span><textarea value={transcript} onChange={(event) => setTranscript(event.target.value)} rows="7" maxLength="20000" placeholder="Enter the exact words spoken in the recording…" required /><small>{displayWordCount(transcript)} words · Audio uploads up to {Math.round((capabilities?.max_upload_bytes || 50 * 1024 * 1024) / (1024 * 1024))} MB each</small></label>
        {capabilityError && <div className="evaluation-alert error" role="alert">{capabilityError} Start the backend service and reload this page.</div>}
        {error && <div className="evaluation-alert error" role="alert">{error}</div>}
        <button className="primary-button evaluation-submit" type="submit" disabled={status === 'analyzing' || !participantAudio || !transcript.trim()}>{status === 'analyzing' ? <><span className="button-spinner" /> Analyzing audio…</> : <>Run evaluation <span>↗</span></>}</button>
        <p className="evaluation-footnote">{referenceAudio ? 'Contrastive mode: participant will be compared with the ideal reference.' : 'Standalone mode: scoring uses the backend absolute-range rubric.'}</p>
      </form>

      <section className="evaluation-output" aria-live="polite">
        {!result && <div className="empty-evaluation"><div className="empty-mark">01</div><div className="eyebrow">RESULTS WILL APPEAR HERE</div><h2>Grounded speech feedback.</h2><p>Run an evaluation to see the composite score, rubric breakdown, and time-stamped delivery deviations returned by the backend.</p><div className="result-steps"><span>01 Feature extraction</span><span>02 Reference comparison</span><span>03 Temporal grounding</span></div></div>}
        {result && <div className="result-stack">
          <div className="result-header"><div><div className="eyebrow">COMPLETED RUN / {result.id}</div><h2>Evaluation results</h2></div><button type="button" className="secondary-button" onClick={downloadResult}>Download JSON</button></div>
          <div className="result-summary-card"><div className="result-score"><strong>{score.toFixed(1)}</strong><span>/ 100</span></div><div><div className="eyebrow">COMPOSITE DELIVERY SCORE</div><p>{result.summary}</p><div className="result-meta">{prettyTime(result.duration_seconds)} duration <span>·</span> {result.metadata?.has_baseline ? 'contrastive reference used' : 'standalone scoring'} <span>·</span> {new Date(result.evaluated_at || Date.now()).toLocaleString()}</div></div></div>
          <section className="result-section"><div className="section-heading"><div><div className="eyebrow">RUBRIC</div><h2>Dimension scores</h2></div></div><div className="rubric-grid">{(result.rubric_scores || []).map((item) => <article className="rubric-card" key={item.name}><div><strong>{item.name}</strong><b>{Number(item.score).toFixed(0)}</b></div><div className="rubric-track"><i style={{ width: `${Math.max(0, Math.min(100, Number(item.score)))}%` }} /></div><p>{item.explanation || 'Backend rubric score.'}</p></article>)}</div></section>
          <section className="result-section"><div className="section-heading"><div><div className="eyebrow">TEMPORAL GROUNDING</div><h2>Detected findings <span className="result-count">{result.flaws?.length || 0}</span></h2></div></div>{result.flaws?.length ? <div className="real-flaw-list">{result.flaws.map((flaw) => <article className="real-flaw-card" key={flaw.id}><div className="real-flaw-top"><span className={`severity ${String(flaw.severity).toLowerCase()}`}>{flaw.severity}</span><span className="mono">{prettyTime(flaw.start)} – {prettyTime(flaw.end)}</span></div><h3>{flaw.type}</h3><p>{flaw.explanation}</p>{flaw.metadata && Object.keys(flaw.metadata).length > 0 && <details><summary>Acoustic evidence</summary><pre>{JSON.stringify(flaw.metadata, null, 2)}</pre></details>}</article>)}</div> : <div className="no-findings">No temporal flaw regions were detected in this run.</div>}</section>
          {result.notes?.length > 0 && <div className="evaluation-alert note"><strong>Alignment note</strong><ul>{result.notes.map((note) => <li key={note}>{note}</li>)}</ul></div>}
          <details className="transcript-result"><summary>Transcript used for this evaluation · {displayWordCount(transcriptText)} words</summary><p>{transcriptText}</p></details>
          <details className="transcript-result"><summary>Pipeline metadata</summary><pre>{JSON.stringify(result.metadata, null, 2)}</pre></details>
        </div>}
      </section>
    </div>
  </div>;
}
