# Product Requirements Document

## Contrastive Speech Analytics & Temporal Flaw Grounding

**Track:** C  
**Document status:** MVP specification  
**Primary artifact:** `prd.md`  
**Audience:** Engineering, data, ML/audio, design, demo, and evaluation teams

---

## 1. Product Overview

Contrastive Speech Analytics & Temporal Flaw Grounding is a reproducible speech-evaluation system that compares a participant’s delivery with an aligned, high-quality baseline delivery of the **same text**. The system extracts acoustic and temporal features, identifies measurable deviations, localizes those deviations to timestamps and transcript regions, explains the acoustic/mathematical basis for each finding, and presents the evidence in an interactive dashboard.

The product is designed to make spoken-performance feedback more consistent, inspectable, and actionable without making unsupported claims about a speaker’s intent, personality, confidence, or psychological state.

### 1.1 Product principles

1. **Contrastive before absolute:** A participant is evaluated against an aligned baseline whenever a valid baseline exists.
2. **Evidence before judgment:** Every flaw must be supported by measurable feature values, comparison values, deviation magnitude, and confidence.
3. **Temporal by default:** Findings must point to a time range and affected transcript region, not only produce an utterance-level score.
4. **Reproducible:** The same inputs, configuration, model/version, and preprocessing settings must produce the same or acceptably equivalent outputs.
5. **Configurable scoring:** Rubric weights and thresholds live in versioned configuration, not scattered application code.
6. **Quality over scale:** The custom contrastive dataset should contain carefully paired, high-quality examples rather than an artificially large quantity of weak data.
7. **Human-readable but technically grounded:** Explanations translate acoustic evidence into plain language while preserving the underlying mathematics.

---

## 2. Problem Statement

### 2.1 Current problem

Spoken-performance evaluation is often subjective and inconsistent. Different judges may apply different standards for pacing, pauses, pitch movement, vocal energy, clarity, and consistency. Even the same evaluator may score the same delivery differently depending on context, fatigue, or whether the evaluator can remember the intended reference delivery. Numeric scores without evidence do not tell a participant what happened or where it happened.

Ordinary speech-recognition datasets do not solve this problem. They are generally optimized for recognizing words, phonemes, or intent across varied speech. They do not necessarily contain:

- A strong baseline delivery and a flawed delivery of the exact same text.
- Controlled variation in delivery quality.
- Human-annotated flaw categories.
- Start and end timestamps for each flaw.
- Evidence describing how a delivery differs acoustically from a baseline.

A system trained or evaluated only on recognition accuracy may correctly transcribe a speech sample while missing pacing problems, poorly controlled pauses, monotone delivery, unstable energy, or reduced clarity.

### 2.2 Why paired recordings are required

The central unit of this product is a contrastive pair:

- **Baseline/ideal recording:** A high-quality delivery of a defined transcript.
- **Flawed mirror recording:** A delivery of the same transcript containing one or more controlled delivery flaws.

Exact transcript pairing reduces confounding from lexical content. The system can compare a participant’s delivery to a reference whose words and sequence are known, rather than treating differences in text, topic, or sentence structure as delivery errors. Alignment then allows feature differences to be interpreted in the relevant word or phrase context.

The baseline is not assumed to be universally perfect. It is a reference delivery for the selected text and evaluation configuration. The product must distinguish measured deviation from a normative or psychological judgment.

### 2.3 Temporal grounding

A useful result must answer **where** a problem occurs. Temporal grounding identifies the flaw’s start and end timestamps, affected words or segment, and the corresponding baseline and participant measurements. For example, “pacing score: 62” is insufficient; the system should identify the relevant interval and explain whether speech rate increased, a pause was shortened, or timing became inconsistent relative to the aligned reference.

### 2.4 Causal explainability

For this product, “causal explanation” means an evidence chain from a detected measurable deviation to an observed delivery effect:

1. An acoustic or temporal feature was measured.
2. The participant and baseline values were compared over an aligned region.
3. The deviation exceeded a configured threshold with adequate confidence.
4. The deviation was mapped to a flaw category.
5. The system generated a human-readable explanation tied to those values.

The system must not claim that a speaker is nervous, unprepared, dishonest, or lacking confidence unless such a claim is independently supported by an authorized process; those claims are outside this product.

---

## 3. Goals and Non-Goals

### 3.1 Goals

- Compare participant audio with an aligned baseline delivery of the same text.
- Validate transcript and alignment quality before scoring.
- Extract interpretable acoustic and temporal features.
- Detect delivery deviations and categorize them into configurable flaw types.
- Ground each finding to timestamps and transcript words/segments.
- Show baseline values, participant values, deviation magnitude, and confidence.
- Generate technically supported, human-readable explanations.
- Provide an interactive dashboard for playback, waveform inspection, time-series comparison, and transcript synchronization.
- Make data, preprocessing, feature extraction, scoring, and outputs reproducible.
- Demonstrate stress testing with controlled flaws and near-perfect examples.

### 3.2 Non-goals

- General-purpose automatic speech recognition as the primary product.
- Speaker identity, emotion, personality, intent, or psychological-state inference.
- A universal definition of an “ideal” human speaking style.
- Replacing judges or coaches in high-stakes decisions.
- A huge production-scale speech corpus.
- Real-time streaming analysis for the MVP.
- Unrelated features such as social networking, marketplace functionality, coaching subscriptions, or medical assessment.

---

## 4. Target Users

| User | Need | Primary product value |
|---|---|---|
| Speech competitors | Understand exactly where delivery deviated and how to improve | Evidence-backed, timestamped feedback |
| Students | Learn measurable delivery concepts through examples | Clear explanations paired with waveform and transcript evidence |
| Coaches | Review multiple dimensions of a delivery and compare progress | Repeatable rubric and detailed flaw cards |
| Judges/evaluators | Apply a more consistent evaluation framework | Transparent measurements and configurable scoring |

### 4.1 User assumptions

- Users can provide an audio file in a supported format.
- Users can provide or select the transcript associated with the recording.
- Users understand that a comparison is valid only when transcript and baseline matching are sufficiently reliable.
- Users may not understand acoustic terminology; the dashboard must explain terms without hiding the evidence.

---

## 5. Core User Journey

1. **Upload participant audio.** The user selects a supported audio file and sees basic metadata such as duration, sample rate, channels, and detected speech presence.
2. **Upload or paste transcript.** The user supplies the intended text. The system normalizes whitespace and punctuation for matching while preserving a display form.
3. **Select or retrieve matching baseline.** The user selects a baseline by exact transcript match or enters a baseline recording from the custom dataset.
4. **Validate transcript and alignment.** The system checks transcript compatibility, performs forced alignment, reports low-confidence or unmapped regions, and requires the user to acknowledge or correct blocking issues.
5. **Extract acoustic features.** The system processes both recordings with identical preprocessing and feature-extraction settings.
6. **Compare participant against baseline.** Features are compared over aligned words, phonemes, pauses, and larger segments.
7. **Detect deviation/stress regions.** Configured thresholds, smoothing, and minimum-duration rules identify candidate flaw regions.
8. **Generate causal explanations.** Each accepted finding is converted into an explanation containing measurements and a cautious interpretation.
9. **Display score and evidence.** The dashboard presents dimension scores, overall score, confidence/coverage, and flaw cards.
10. **Review waveform/time-series regions.** The user can jump to a finding, play both recordings, inspect overlays, and view the synchronized transcript.
11. **Export/share results.** The user can export a self-contained report or structured result artifact, subject to privacy settings.

---

## 6. Custom Contrastive Dataset

### 6.1 Dataset objective

The dataset must support detection, temporal localization, explanation, and stress testing of delivery flaws. It should be intentionally scoped and quality-controlled. The project must not invent a requirement for thousands or millions of recordings.

A practical MVP dataset may consist of a **small, carefully reviewed set of transcripts**, each with one or more high-quality baseline recordings and controlled mirror recordings spanning the flaw spectrum. The exact final count is a data-collection decision; acceptance depends on pairing quality, annotation quality, and reproducibility rather than a minimum volume target.

### 6.2 Recording entities

Each recording must have:

- A stable recording ID.
- A transcript ID and exact transcript text/version.
- A role: `baseline` or `flawed_mirror`.
- Speaker ID, stored as a pseudonymous identifier.
- Recording conditions and capture metadata.
- Audio format, sample rate, channel count, duration, and preprocessing status.
- Flaw labels, if applicable.
- Temporal annotations, if applicable.
- Dataset split assignment.
- Annotation and review status.
- Data and schema version.

### 6.3 Baseline/ideal recordings

A baseline recording is a reviewed delivery intended to represent a strong reference for the selected transcript. Baseline criteria should include:

- Exact transcript compatibility.
- Complete speech coverage with no unintended omissions.
- Understandable articulation and acceptable recording quality.
- Natural but deliberate pacing, pausing, pitch movement, energy dynamics, and clarity.
- No known target flaw in the region used for comparison.
- Human review by at least one qualified reviewer; important benchmark examples should receive a second review.

The label “ideal” is a dataset role, not a claim that the recording is objectively perfect.

### 6.4 Flawed mirror recordings

A flawed mirror recording uses the exact same transcript as its baseline and introduces a controlled delivery issue. Examples include:

- Increased or reduced speech rate in a defined region.
- A shortened, extended, or misplaced pause.
- Reduced pitch variation or excessive pitch movement.
- Flattened or unstable energy dynamics.
- Reduced vocal clarity or articulation quality.
- Inconsistent delivery across otherwise comparable regions.

A recording may contain one primary flaw for clear supervision. Multi-flaw examples may be included for stress testing only when each flaw has distinct annotations and the expected interaction is documented.

### 6.5 Flaw spectrum

For each flaw category, examples should span:

- **Severe:** Clearly measurable and perceptible deviation.
- **Moderate:** Detectable deviation requiring comparison evidence.
- **Mild:** Small but meaningful deviation near the decision boundary.
- **Near-perfect/control:** A recording expected not to trigger the target flaw.

This spectrum enables threshold calibration and false-positive testing without pretending that a single binary label captures delivery quality.

### 6.6 Exact transcript pairing

Pairing must be validated at the normalized text level. The dataset must retain both:

- The original display transcript, including punctuation and formatting.
- A canonical comparison transcript with deterministic normalization rules.

A baseline and mirror are valid pairs only if their canonical transcript matches exactly, unless an explicit exception is recorded and excluded from exact-pair evaluation.

### 6.7 Temporal labels

Each annotated flaw should include:

- `start_time_seconds`.
- `end_time_seconds`.
- Affected word indices and/or phoneme indices.
- Flaw category and severity.
- Annotation rationale.
- Annotator ID or pseudonym.
- Review status and disagreement notes.

Temporal boundaries should be based on the audio/alignment evidence available to the annotator. Boundary uncertainty may be recorded when an exact boundary cannot be determined.

### 6.8 Speaker metadata

Store only metadata required for analysis and fairness review, such as pseudonymous speaker ID, language/accent information when voluntarily provided and relevant, recording conditions, and capture device class. Do not collect unnecessary sensitive personal information. Speaker metadata must not be used to make unsupported quality judgments.

### 6.9 Flaw taxonomy

The initial taxonomy is:

| Category | Primary evidence |
|---|---|
| Pacing deviation | Speech rate or local timing differs from baseline |
| Pause-control deviation | Pause duration, location, or consistency differs |
| Pitch-variation deviation | F0 range, contour, or local variation differs |
| Energy/volume-dynamics deviation | Loudness/energy contour or dynamics differs |
| Vocal-clarity deviation | Spectral/articulation proxy or signal quality differs |
| Delivery-consistency deviation | Repeated aligned regions show unstable timing or feature behavior |

A category must not be emitted if its required feature coverage is missing.

### 6.10 Train/validation/test split

Splits must be performed by speaker, not by randomly scattering recordings from the same speaker across splits. Where possible, paired baseline and mirror recordings remain in the same split to prevent leakage. The split manifest must be versioned and record the random seed and assignment logic.

- **Train:** Used for calibration or model fitting, if applicable.
- **Validation:** Used to tune thresholds, smoothing, and rubric weights.
- **Test:** Held out until evaluation and demo validation.

Near-duplicate takes, recordings from the same session, and derived segments must not cross split boundaries.

### 6.11 Reproducibility requirements for data

The repository must include:

- Dataset manifest with stable IDs and checksums.
- Schema definition.
- Transcript normalization rules.
- Split manifest and random seed.
- Annotation guidelines and label definitions.
- Data version identifier.
- A documented handling policy for unavailable or restricted audio.
- A small distributable sample or synthetic fixture sufficient to run the pipeline when full audio cannot be redistributed.

---

## 7. Audio Preprocessing and Alignment

### 7.1 Deterministic preprocessing

Participant and baseline audio must use the same preprocessing pipeline and configuration. The pipeline should document sample-rate conversion, channel handling, amplitude normalization policy, silence handling, and any denoising or filtering. Preprocessing must not erase the very delivery differences the system is intended to measure.

Every processed artifact must retain the original file checksum and preprocessing configuration ID.

### 7.2 Forced alignment

Forced alignment maps transcript units to audio timestamps using the transcript as a constraint. The implementation may align words and phonemes using an appropriate alignment engine, provided the engine, model/version, language configuration, and settings are recorded.

The output must include, where available:

- Token/word text.
- Canonical token index.
- Phoneme sequence and indices.
- Start and end timestamps.
- Alignment confidence.
- Unmapped or low-confidence status.

Alignment must be performed for participant and baseline using compatible transcript units. The system must identify alignment failures rather than silently scoring invalid regions.

### 7.3 Alignment validation rules

A run is valid for scoring only when:

- The canonical transcripts match.
- Required speech regions are aligned.
- The percentage of unmapped or low-confidence units is below a configurable limit.
- Audio duration and timestamp ranges are internally consistent.
- Both recordings have sufficient usable speech for the requested dimensions.

Invalid or low-confidence regions may be displayed for inspection but must be excluded from scoring or explicitly marked as unavailable.

---

## 8. Feature Extraction

Features are extracted over consistent frames and aggregated over aligned words, phonemes, pauses, and flaw windows. The exact window size, hop size, frequency range, normalization, and aggregation method must be versioned.

| Feature | What it measures | Why it matters | Contribution to flaw detection |
|---|---|---|---|
| FFT / short-time spectrum | Distribution of signal energy across frequencies over short windows | Captures spectral shape and changes in articulation or voice quality | Provides the basis for spectral summaries, vocal-clarity proxies, and optional spectral comparisons |
| MFCC | Compact representation of the short-time spectral envelope on a perceptual scale | Captures timbral and articulation-related differences while reducing dimensionality | Supports clarity-related comparison and local acoustic deviation detection |
| Pitch / F0 | Estimated fundamental frequency over voiced frames | Represents vocal pitch and its contour over time | Detects reduced pitch variation, unexpected contour changes, and pitch instability relative to baseline |
| Energy / loudness | Signal energy or perceptually related amplitude measure | Describes vocal intensity and dynamic emphasis | Detects flattened, excessive, or unstable volume/energy dynamics |
| Speech rate | Spoken units per unit time, such as words/second or syllables/second | Pacing affects intelligibility, emphasis, and timing | Detects local acceleration/deceleration against the aligned baseline |
| Pause duration | Duration of silence or low-energy intervals between speech units | Pauses shape phrasing and emphasis | Detects shortened, extended, missing, misplaced, or inconsistent pauses |
| Vocal clarity | A defined combination of measurable signal/alignment proxies, such as spectral and articulation consistency | Gives a user-facing dimension for intelligibility-related evidence | Detects regions whose acoustic evidence differs from the baseline, without claiming a medical or psychological diagnosis |
| Optional spectral features | Spectral centroid, bandwidth, roll-off, zero-crossing rate, spectral flux, or similar documented statistics | Adds interpretable detail when needed | May improve discrimination of timbral, articulation, or signal changes; optional features cannot be required for the MVP unless validated |

### 8.1 Feature computation rules

- Use identical preprocessing and extraction parameters for both recordings.
- Preserve frame-level values before aggregation.
- Use robust aggregation where appropriate, such as median and interquartile range, rather than only a single mean.
- Mark undefined values, such as unvoiced F0, explicitly.
- Do not impute missing values without recording the method and its effect on confidence.
- Normalize only with a documented policy. Baseline-relative and speaker-normalized values must not be mixed silently.

### 8.2 Baseline-relative comparison

For a feature value `x_p` from the participant and aligned baseline value `x_b`, a basic signed deviation is:

`d = x_p - x_b`

A relative deviation can be computed as:

`r = (x_p - x_b) / max(|x_b|, epsilon)`

where `epsilon` is a configured numerical stabilizer. For features where direction is not meaningful, use an absolute or robust standardized deviation. The UI must identify whether a displayed deviation is signed, absolute, percentage, or standardized.

### 8.3 Aggregation and smoothing

Frame-level deviations may be smoothed to avoid one-frame spikes. Smoothing window, aggregation function, minimum region duration, and merge-gap rules must be configurable and recorded in the run configuration. Smoothing must not shift displayed timestamps without accounting for the transformation.

---

## 9. Temporal Grounding

Every emitted flaw finding must have a structured temporal representation:

```json
{
  "finding_id": "stable-id",
  "category": "pacing_deviation",
  "start_time_seconds": 18.42,
  "end_time_seconds": 21.10,
  "affected_units": ["emphasized", "phrase"],
  "baseline_value": 3.10,
  "participant_value": 4.02,
  "deviation_magnitude": 0.92,
  "deviation_unit": "words_per_second",
  "relative_deviation_percent": 29.7,
  "confidence": 0.91,
  "alignment_confidence": 0.96,
  "evidence_feature": "speech_rate",
  "severity": "moderate",
  "explanation_template_id": "pacing-compression-v1"
}
```

### 9.1 Required grounding fields

- **Flaw start timestamp:** First timestamp in the detected region.
- **Flaw end timestamp:** Last timestamp in the detected region.
- **Affected words/segment:** Transcript units overlapping the region.
- **Baseline value:** Reference measurement using the displayed unit.
- **Participant value:** Participant measurement using the same unit.
- **Deviation magnitude:** Difference or standardized distance, with direction where meaningful.
- **Confidence:** Confidence in the finding, incorporating detection evidence and data quality.

### 9.2 Grounding behavior

- Findings must be sortable by time and category.
- Clicking a finding must seek playback to its start timestamp.
- The waveform, feature plot, and transcript must highlight the same region.
- Overlapping findings must be represented without hiding evidence; they may be grouped when configured.
- Findings below minimum duration, threshold, or confidence must remain available as diagnostic candidates only if the UI clearly labels them as non-scored.

---

## 10. Causal Explainability

### 10.1 Explanation contract

Every scored finding must generate an explanation with this structure:

1. **Time and category:** `00:18.42–00:21.10 — Pacing deviation`.
2. **Measured comparison:** `Speech rate increased from 3.10 to 4.02 words/second (+29.7%) relative to the aligned baseline.`
3. **Observed delivery consequence:** `The interval was compressed relative to the reference, including the pause before the emphasized phrase.`
4. **Evidence and confidence:** The feature, units, threshold, alignment confidence, and detection confidence.
5. **Limitation, when relevant:** A note that the system reports an acoustic/timing deviation and does not infer intent or psychological state.

### 10.2 Explanation rules

- Values must be derived from the stored result, not regenerated inconsistently in the UI.
- The explanation must not claim more than the measured feature supports.
- Psychological, moral, medical, or personality claims are prohibited.
- If multiple features contribute, list each feature and its contribution.
- If evidence is insufficient, return an uncertainty message rather than a confident causal statement.
- Templates and their versions must be stored with the run.

### 10.3 Example

> **00:18.42–00:21.10 — Pacing deviation**  
> Speech rate increased from **3.10** to **4.02 words/second** (**+29.7%**) relative to the aligned baseline. The participant compressed the interval containing the pause before the emphasized phrase. Detection confidence is **0.91** and alignment confidence is **0.96**. This finding describes a measurable timing difference; it does not infer why the difference occurred.

---

## 11. Scoring Rubric

### 11.1 Scoring principles

Scores must be transparent, reproducible, and based on measurable acoustic deviations. The rubric is a configuration artifact with a version, not a set of hardcoded constants distributed throughout the application.

### 11.2 Dimensions

The MVP rubric should include:

| Dimension | Example measurable inputs |
|---|---|
| Pacing | Local speech-rate deviation and timing consistency |
| Pause control | Pause duration/location deviation and pause consistency |
| Pitch variation | F0 range, contour variation, and voiced-frame coverage |
| Energy/volume dynamics | Energy contour, dynamic range, and aligned emphasis differences |
| Vocal clarity | Validated MFCC/spectral/articulation proxies and confidence |
| Delivery consistency | Variation of aligned deviations across repeated or comparable segments |

### 11.3 Configurable rubric model

A versioned rubric configuration should define, at minimum:

- Dimension IDs and display names.
- Feature inputs.
- Directionality of each feature.
- Threshold bands for mild, moderate, and severe deviations.
- Minimum duration and minimum evidence requirements.
- Dimension weights.
- Overall score range and missing-data behavior.
- Confidence and coverage rules.
- Explanation template mappings.

A conceptual dimension score may be computed as a weighted penalty from normalized deviations:

`dimension_score = max(0, 100 - 100 * weighted_penalty)
`

The implementation must document the normalization and clipping rules. Overall score is a weighted combination of available dimension scores, with a visible coverage indicator. Missing or invalid dimensions must not silently receive a perfect score.

### 11.4 Score interpretation

The UI should present scores as indicators of deviation relative to the selected baseline and rubric configuration. It must display the rubric version and avoid presenting the result as an objective measure of human worth or universal speaking quality.

---

## 12. Dashboard Requirements

### 12.1 Primary views

1. **Input and setup:** Audio upload, transcript input, baseline selection, and validation messages.
2. **Processing status:** Ordered pipeline steps, progress, warnings, and errors.
3. **Overview:** Overall score, dimension score cards, evidence coverage, alignment status, and top findings.
4. **Evidence workspace:** Synchronized waveform, baseline-vs-participant feature overlays, flaw-region highlighting, and timeline.
5. **Finding detail:** Flaw category, timestamps, affected transcript, values, deviation, confidence, mathematical evidence, explanation, and limitations.
6. **Transcript synchronization:** Word/segment highlighting that follows playback and can seek audio.
7. **Export/share:** Downloadable result package or report with configuration/version metadata.

### 12.2 Required controls

- Play/pause, seek, skip to finding, playback speed, and baseline/participant audio selection.
- Timeline zoom and region navigation.
- Feature selector and baseline/participant overlay toggle.
- Category and severity filters.
- Toggle for scored findings versus diagnostic candidates.
- Expandable mathematical evidence.
- Clear display of processing and alignment confidence.

### 12.3 Visualization requirements

- Waveform must use time in seconds on the x-axis.
- Feature plots must share a meaningful time axis with the waveform.
- Highlighted regions must align across waveform, plots, timeline, and transcript.
- Baseline and participant must be visually distinguishable using more than color alone.
- Tooltips must show units, values, and timestamp.
- Visualizations must remain usable for short and long recordings.

---

## 13. Functional Requirements

### 13.1 Inputs and validation

- **FR-01:** Accept participant audio in documented supported formats.
- **FR-02:** Accept transcript by paste and file input where supported.
- **FR-03:** Allow baseline selection by exact transcript match.
- **FR-04:** Validate file readability, duration, channels, sample rate, and speech presence.
- **FR-05:** Normalize and compare transcripts deterministically.
- **FR-06:** Block or warn on transcript mismatch, insufficient audio, or invalid alignment.

### 13.2 Processing

- **FR-07:** Run the same versioned preprocessing and feature extraction configuration on both recordings.
- **FR-08:** Produce word/phoneme alignment timestamps and confidence.
- **FR-09:** Extract required MVP features and preserve intermediate results.
- **FR-10:** Compare aligned features and calculate deviations.
- **FR-11:** Detect findings using configured thresholds, duration rules, smoothing, and confidence rules.
- **FR-12:** Generate a structured result object containing scores, findings, evidence, warnings, and provenance.

### 13.3 Results and interaction

- **FR-13:** Display overall and dimension scores with rubric version.
- **FR-14:** Display every scored finding with the required temporal and evidence fields.
- **FR-15:** Synchronize playback, waveform, feature plots, timeline, and transcript.
- **FR-16:** Generate human-readable explanations from versioned templates.
- **FR-17:** Permit users to inspect diagnostic candidates and excluded regions.
- **FR-18:** Export a human-readable report and machine-readable result artifact.
- **FR-19:** Preserve an analysis run ID and input/configuration provenance.

---

## 14. Non-Functional Requirements

### 14.1 Reproducibility and determinism

- Same input checksums, dataset version, code version, environment, and configuration must produce the same result within documented floating-point tolerance.
- Random seeds must be explicit.
- All model, alignment, feature, rubric, and explanation versions must be recorded.

### 14.2 Performance

- The MVP should provide visible processing progress and avoid unbounded waits.
- For a demo-length recording, processing should complete within a practical interactive-demo timeframe on the documented environment.
- Performance targets must be measured and documented rather than claimed without benchmark evidence.

### 14.3 Reliability

- A failed stage must report the stage, reason, and remediation.
- Partial intermediate outputs may be retained for debugging but must not be presented as valid final scores.
- The system must distinguish warnings from blocking errors.

### 14.4 Maintainability

- Separate ingestion, validation, alignment, feature extraction, comparison, scoring, explanation, and presentation layers.
- Use typed schemas or equivalent validation for inputs and results.
- Keep rubric and pipeline settings in versioned configuration.
- Provide tests for core numerical transforms and end-to-end fixtures.

### 14.5 Observability

- Record pipeline stage status, duration, configuration ID, and error category.
- Avoid storing raw audio or transcript content in logs unless explicitly enabled for a local development run.
- Provide enough metadata to reproduce a failed analysis.

---

## 15. Reproducibility

The repository must include:

- A clear setup and run guide.
- Dependency lockfile or pinned environment specification.
- Dataset schema, manifest, split manifest, and version identifiers.
- Sample or synthetic fixture data for an end-to-end run.
- Deterministic transcript normalization and preprocessing code.
- Feature extraction configuration.
- Alignment engine/model/version configuration.
- Rubric configuration and explanation template versions.
- Fixed random seeds where randomness exists.
- Automated tests and a documented test command.
- A single documented command or workflow that reproduces the demo result.
- Output checksums or a result manifest for the reference run.

The demo should show the configuration/provenance panel so evaluators can see that the result is not a manually authored screenshot.

---

## 16. Accessibility

- All important information must be available in text, not only by color or chart shape.
- Use sufficient contrast and visible keyboard focus.
- Provide keyboard-accessible audio controls and finding navigation.
- Do not use color as the only distinction between participant and baseline.
- Charts should include accessible labels, legends, units, and textual summaries.
- Avoid flashing or rapidly animated regions.
- Transcript highlighting must have a non-color indicator.
- Error and warning messages must be clear and associated with the relevant input.

---

## 17. Error Handling

| Condition | Required behavior |
|---|---|
| Unsupported audio format | Explain supported formats and identify the rejected file |
| Corrupt or unreadable file | Stop processing and provide a recoverable error |
| Empty or missing transcript | Block alignment and request transcript input |
| Transcript mismatch | Show normalized comparison and require a matching baseline or correction |
| Low alignment confidence | Mark affected regions, exclude them from scoring when configured, and show coverage impact |
| Insufficient speech/silence-heavy audio | Warn or block affected dimensions with an explanation |
| Missing feature values | Exclude the affected evidence, reduce coverage, and never silently impute a perfect result |
| Baseline unavailable | Allow only a clearly labeled non-contrastive diagnostic mode if implemented; otherwise block the comparison |
| Processing failure | Show failed stage, run ID, actionable message, and retry option |
| Export failure | Preserve the result in the dashboard and provide a retryable export error |

A failure must never produce an apparently valid score with hidden missing evidence.

---

## 18. Privacy and Security

- Treat audio, transcripts, and speaker metadata as sensitive user data.
- Minimize collection and retention.
- Use pseudonymous IDs in dataset and result artifacts where possible.
- Do not expose raw audio through guessable public URLs.
- Restrict access to uploaded data and exported results according to the deployment environment.
- Keep secrets out of source control and logs.
- Provide deletion/cleanup behavior for temporary processing artifacts.
- Document whether data is stored locally, in a controlled service, or only for the session.
- Do not use recordings for a new purpose without appropriate authorization.

---

## 19. Dataset Governance

- Maintain annotation guidelines for baseline quality, flaw categories, temporal boundaries, and severity.
- Track annotator identity using pseudonyms and retain review history.
- Record disagreements and adjudication decisions.
- Document consent and permitted use for contributed audio.
- Provide data cards describing collection conditions, limitations, intended use, and known biases.
- Prevent speaker leakage across train/validation/test splits.
- Do not claim broad fairness or generalization from a small dataset.
- Exclude or flag recordings with conditions that invalidate the intended comparison.
- Version every change to labels, transcripts, splits, and metadata.

---

## 20. Demo Requirements

The demo must show a complete, reproducible path using a prepared paired example:

1. Load or upload participant audio.
2. Show the exact transcript and selected matching baseline.
3. Validate alignment and display confidence/coverage.
4. Show processing status through feature extraction and comparison.
5. Display the score overview and rubric version.
6. Select at least one flaw card and jump to its timestamp.
7. Show waveform, baseline-vs-participant feature overlay, affected transcript words, and timeline highlighting.
8. Expand mathematical evidence with baseline value, participant value, deviation, units, and confidence.
9. Show the human-readable causal explanation and its limitation.
10. Export or display the machine-readable result/provenance artifact.

The demo should include at least one clear flaw, one near-perfect/control example, and a visible stress-test or validation result. It should not rely on manually edited output that the pipeline could not reproduce.

---

## 21. Expected Deliverables

1. Production-grade `prd.md`.
2. Reproducible source repository with documented setup.
3. Contrastive dataset manifest, schema, annotation guidance, and permitted sample/fixture data.
4. Audio preprocessing and forced-alignment pipeline.
5. Feature extraction module for FFT, MFCC, pitch/F0, energy, speech rate, pause duration, vocal clarity, and documented optional spectral features.
6. Alignment-aware comparison and temporal flaw-detection module.
7. Configurable scoring rubric and versioned explanation templates.
8. Interactive dashboard with required upload, validation, visualization, playback, transcript, scoring, and evidence views.
9. Machine-readable result schema and human-readable export.
10. Automated tests, reproducibility instructions, and reference run metadata.
11. Demo script or walkthrough covering the evaluation criteria.

---

## 22. Hackathon Evaluation Mapping

| Official criterion | Weight | Product functionality and evidence |
|---|---:|---|
| Causal Explainability & Temporal Grounding | 25% | Every scored flaw has timestamps, affected transcript units, baseline/participant values, deviation magnitude, confidence, mathematical evidence, and a cautious human-readable explanation. Selecting a flaw synchronizes waveform, plots, transcript, timeline, and playback. |
| Data Engineering & Stress Testing | 30% | Custom exact-transcript baseline/mirror pairs; controlled flaw spectrum from severe to near-perfect; temporal labels; speaker-aware splits; schema, manifest, validation, leakage controls, and stress-test cases. |
| Feature Extraction | 20% | Reproducible FFT, MFCC, pitch/F0, energy, speech-rate, pause, and vocal-clarity extraction, with documented measurement meaning, units, comparison logic, missing-value handling, and optional spectral features. |
| Visualization & Dashboard | 15% | Upload and setup flow; processing status; waveform; baseline/participant overlays; flaw-region highlighting; timeline; score cards; flaw cards; mathematical evidence; causal explanation; synchronized transcript; playback controls. |
| Reproducibility & Code Quality | 10% | Versioned configuration, deterministic preprocessing, pinned environment, split manifest, fixtures, tests, run IDs, provenance, reference command, modular architecture, and documented limitations. |

The implementation should prioritize these capabilities directly. It must not introduce unrelated features merely to increase apparent complexity.

---

## 23. MVP Scope

The MVP includes:

- File-based participant and baseline audio input.
- Transcript paste/upload and exact matching validation.
- Forced alignment with confidence and invalid-region handling.
- Required acoustic/temporal feature extraction.
- Baseline-relative comparison.
- Configurable flaw thresholds and scoring weights.
- At least the six rubric dimensions listed above.
- Timestamped flaw findings with affected transcript units.
- Mathematical evidence and versioned causal explanations.
- Interactive dashboard with synchronized waveform, time-series overlay, timeline, transcript, score cards, flaw cards, and playback.
- Exportable JSON result and human-readable report.
- Small, high-quality contrastive dataset or fixture with speaker-aware splits and controlled flaw spectrum.
- Tests and a reproducible reference run.

---

## 24. Stretch Scope

Stretch work is allowed only after MVP evidence is complete:

- More robust phoneme-level comparison where alignment quality supports it.
- Multi-flaw decomposition with explicit interaction handling.
- Coach-facing comparison across multiple sessions using the same rubric.
- Improved uncertainty calibration and annotator disagreement visualization.
- Additional validated spectral features.
- Batch analysis of a small set of recordings.
- Controlled user annotations that can be exported for dataset governance.
- More advanced baseline retrieval when exact transcript matching remains deterministic and auditable.

Stretch features must not compromise the core requirements for evidence, temporal grounding, reproducibility, privacy, or clarity.

---

## 25. Explicitly Out of Scope

- General conversational speech recognition product capabilities.
- Real-time streaming coaching.
- Psychological, emotional, personality, confidence, honesty, or intent inference.
- Medical, clinical, or diagnostic claims about voice.
- Automated judge replacement or high-stakes certification.
- Large-scale data collection requirement unrelated to benchmark quality.
- Social features, leaderboards, payments, subscriptions, or marketplace workflows.
- Unreviewed black-box scores without feature evidence.
- A universal “perfect speaker” model detached from transcript-matched baselines.
- Silent fallback to non-contrastive scoring when the baseline is missing.

---

## 26. MVP Acceptance Criteria

The MVP is accepted when all of the following are true:

1. A user can upload participant audio, provide a transcript, select an exact-transcript baseline, and start an analysis.
2. The system rejects or clearly flags transcript mismatch and unusable alignment before presenting a valid comparison score.
3. Participant and baseline audio pass through the same documented preprocessing and feature-extraction configuration.
4. The run produces FFT/spectral basis data, MFCC, pitch/F0, energy, speech rate, pause duration, and vocal-clarity evidence, or explicitly marks a dimension unavailable with a coverage explanation.
5. At least one controlled flaw in the reference fixture is detected and localized within a documented tolerance of the annotated region.
6. A near-perfect/control example does not generate an unjustified scored flaw above the configured threshold, or the false positive is surfaced transparently in validation results.
7. Every scored finding includes start time, end time, affected transcript region, baseline value, participant value, deviation magnitude/unit, confidence, feature name, and severity.
8. The dashboard synchronizes a selected finding across waveform, feature plot, timeline, transcript, and playback.
9. Every scored finding has a human-readable explanation generated from versioned data/templates and contains no unsupported psychological claim.
10. Overall and dimension scores are computed from a versioned configurable rubric; changing the configuration changes the result predictably and is visible in provenance.
11. Missing alignment or feature evidence reduces coverage or blocks the relevant score rather than silently awarding a perfect score.
12. The demo can be rerun from documented setup instructions with the supplied fixture and produces the same result within documented numerical tolerance.
13. Dataset manifests, schemas, split logic, annotation definitions, and data limitations are included.
14. Automated tests cover transcript normalization, alignment/result schema validation, feature comparison, thresholding, scoring, explanation generation, and at least one end-to-end fixture.
15. The deliverables map directly to the five official evaluation criteria without relying on unrelated features.

---

## 27. Open Implementation Decisions to Record

The engineering implementation may choose specific libraries and models, but must record the choice and version for:

- Audio decoding and resampling.
- Forced-alignment engine and acoustic/language model.
- FFT/MFCC/F0 implementation.
- Voice activity and pause-detection method.
- Vocal-clarity proxy definition.
- Dashboard framework and charting library.
- Storage and temporary-artifact policy.
- Numerical tolerance for reproducibility.

These are implementation choices, not invitations to expand product scope. Any choice that materially changes interpretation must be reflected in the schema, rubric, documentation, and evaluation results.
