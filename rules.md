# Engineering Constitution

## Contrastive Speech Analytics & Temporal Flaw Grounding

**Document:** `rules.md`  
**Status:** Mandatory engineering policy  
**Audience:** Every coding agent, engineer, reviewer, and contributor

This document defines the non-negotiable engineering rules for the project. Every coding agent must read and follow it before modifying the repository. If a proposed implementation conflicts with this document, the conflict must be resolved explicitly in the relevant project documentation; it must never be hidden through an implementation shortcut.

---

## 1. Source of Truth

The project documents have the following authority hierarchy:

1. The official **Track C problem statement** is the highest-level product requirement.
2. [`prd.md`](prd.md) defines product behavior, scope, requirements, and acceptance criteria.
3. [`architecture.md`](architecture.md) defines system structure and technical boundaries.
4. [`design.md`](design.md) defines UI/UX behavior and visual interaction requirements.
5. [`tasks.md`](tasks.md) defines implementation priorities and sequencing.
6. This file, [`rules.md`](rules.md), defines engineering constraints that apply across the repository.
7. Existing code is evidence of current behavior, not permission to preserve a contradiction.

### Source-of-truth rules

- Read the relevant source documents before making a change.
- Never silently contradict the problem statement, PRD, architecture, design, or task plan.
- If two documents conflict, stop and document the conflict before implementing a material interpretation.
- Update documentation when an approved architecture or product behavior changes.
- Do not introduce a feature solely because it is technically interesting or easy to demo.

---

## 2. Core Product Principle

> The system must compare a participant’s speech delivery against an aligned baseline delivery of the **same text**.

This is a contrastive speech-analysis system, not a generic speech-quality classifier.

### Required behavior

- A baseline must be transcript-compatible with the participant recording.
- Exact transcript pairing is the default validity requirement.
- Alignment must map comparable words, phonemes, pauses, and segments before meaningful comparison.
- Feature values must be compared over corresponding regions whenever the data supports it.
- The UI and documentation must make the baseline comparison visible.

### Prohibited substitutions

Do not replace the core product with:

- A generic speech-quality classifier.
- A transcription-accuracy dashboard presented as delivery analysis.
- An utterance-level score with no aligned baseline.
- A collection of unrelated audio visualizations.
- A black-box “good/bad speaker” label without measurable evidence.

If a non-contrastive diagnostic mode is ever implemented, it must be clearly labeled, must not impersonate the core comparison, and must not silently replace a missing baseline.

---

## 3. Data Quality and Provenance

Prefer clean, aligned, contrastive data over large, noisy datasets.

### Dataset rules

- Every flawed sample must correspond to an exact baseline transcript.
- Preserve original transcript text and canonical normalized text.
- Preserve provenance for every recording and derived artifact.
- Preserve flaw categories, severity, annotation rationale, and timestamps.
- Preserve stable IDs and checksums where applicable.
- Never fabricate speaker metadata, recording conditions, labels, timestamps, or collection history.
- Record missing or uncertain metadata explicitly as missing or uncertain.
- Keep baseline and mirror recordings traceable to the same transcript/version.
- Keep paired recordings together in the relevant dataset split unless a documented evaluation protocol requires otherwise.
- Prevent speaker/session leakage across train, validation, and test splits.
- Version changes to data, annotations, transcripts, schemas, and split manifests.

### Annotation rules

- Define flaw categories before labeling examples.
- Use documented temporal boundaries.
- Record annotation disagreement instead of silently choosing a convenient label.
- Do not label a flaw when the evidence is inadequate.
- Use near-perfect/control samples to test false positives.
- Do not imply that a baseline is universally perfect; it is a reference delivery for the selected text and rubric.

### Data minimization

Collect and retain only the data needed for the product and evaluation. Speaker metadata must be pseudonymous wherever possible and must not become a proxy for unsupported quality judgments.

---

## 4. Speaker Agnosticism

The system must distinguish delivery deviations from natural speaker characteristics.

### Required safeguards

- Normalize or otherwise account for speaker-dependent characteristics where the analysis requires it.
- Do not penalize a user merely because their natural pitch, energy, accent, timbre, or voice differs from the baseline speaker.
- Use aligned, relative, robust, or speaker-aware comparisons where justified and documented.
- Keep accent and voice characteristics separate from target delivery flaws unless the product requirements explicitly define a measurable, valid, and fair criterion.
- Make normalization methods and their limitations visible in provenance and documentation.
- Test examples with different speakers whenever data permits.

### Prohibited claims

Do not present a speaker’s identity, accent, pitch range, vocal timbre, or energy baseline as a flaw by itself. Do not use demographic or identity attributes to infer delivery quality.

---

## 5. Temporal Grounding

Every important detected flaw must be grounded in time and evidence.

### Required fields

A scored finding must include, at minimum:

- Start timestamp.
- End timestamp.
- Affected word, phoneme, or transcript segment where available.
- Feature evidence.
- Baseline value.
- Participant value.
- Deviation measurement and unit.
- Confidence and/or severity where applicable.
- Alignment confidence or coverage impact where relevant.

### Required behavior

- Findings must be sortable and navigable by time.
- Selecting a finding must seek playback to its start timestamp.
- Waveform, feature plot, timeline, and transcript highlighting must refer to the same region.
- Low-confidence or incomplete regions must be marked and excluded from scoring when configured.
- Boundary transformations caused by smoothing or aggregation must be documented.
- Overlapping findings must be represented honestly rather than hidden.

### Prohibited feedback

Do not produce vague feedback such as:

> “Your delivery needs improvement.”

A user-facing finding must identify what changed, where it changed, how it was measured, and how confident the system is.

---

## 6. Explainability and Evidence

Every explanation must be traceable to measurable evidence.

### Acceptable explanation pattern

> **00:21.40–00:24.10 — Pacing deviation**  
> Speech rate increased **31%** from the aligned baseline in this region. The participant compressed the timing around the affected phrase. Detection confidence: **0.89**.

### Explanation rules

- Use values from the stored analysis result.
- State the feature, comparison, direction, magnitude, unit, region, and confidence.
- Identify the baseline and participant measurements.
- Use versioned explanation templates.
- State limitations when evidence is incomplete or ambiguous.
- Explain mathematical terms in plain language without removing the underlying evidence.
- If several features contribute, list them separately or explain their combined rule.
- If evidence is insufficient, express uncertainty instead of inventing a conclusion.

### Prohibited inference

Do not infer or state that a speaker is:

- Nervous.
- Confident or unconfident.
- Intelligent or unintelligent.
- Lazy, unprepared, dishonest, or careless.
- Emotionally distressed.
- Possessing a particular personality or mental state.

Acoustic deviations are delivery evidence. They are not proof of a speaker’s internal state, intent, character, or ability.

---

## 7. Scoring

Scores must be deterministic, reproducible, configurable, and traceable.

### Scoring rules

- Store weights, thresholds, dimension definitions, and missing-data behavior in versioned configuration.
- Never hide scoring logic inside UI components.
- Never duplicate scoring formulas across frontend, backend, notebooks, or scripts.
- Every score must be traceable to measurable feature values and detected findings.
- Record the rubric version with every analysis result.
- Record the configuration and relevant code/model versions with every reproducible run.
- Make score direction and units explicit.
- Do not silently treat missing evidence as a zero penalty or perfect score.
- Display coverage and confidence alongside scores.
- Use deterministic calculations and explicit rounding rules.
- Keep a score breakdown available for inspection.

### Configuration rules

A rubric configuration should define:

- Dimensions and display names.
- Feature inputs and directionality.
- Threshold bands.
- Minimum duration/evidence requirements.
- Dimension weights.
- Overall-score aggregation.
- Missing-data and low-confidence behavior.
- Explanation-template mappings.

Changing a rubric must be a deliberate, versioned change. Do not edit a production result’s score without preserving the old configuration and provenance.

---

## 8. Feature Extraction

Feature extraction must be modular, reusable, and independent of presentation code.

### Required module boundaries

At minimum, keep these responsibilities separable:

- Audio ingestion and validation.
- Deterministic preprocessing.
- Alignment.
- Frame-level feature extraction.
- Word/segment aggregation.
- Baseline-relative comparison.
- Deviation detection.
- Scoring.
- Explanation generation.
- Result serialization.
- Dashboard presentation.

### Feature rules

- Do not duplicate signal-processing logic across components.
- Use identical documented extraction settings for participant and baseline audio.
- Preserve intermediate feature values needed for evidence and debugging.
- Mark undefined or missing values explicitly.
- Do not silently replace missing values with favorable values.
- Document units, windows, hop sizes, normalization, aggregation, and smoothing.
- Keep feature implementations testable without requiring the frontend.
- Do not add a feature to the rubric without documenting what it measures, why it matters, and how it supports the claimed flaw category.

---

## 9. Frontend Rules

The frontend is an evidence viewer and interaction layer, not a scientific-computation layer.

### Required behavior

- Consume explicit, structured analysis results.
- Render scores, findings, evidence, timestamps, confidence, and explanations from the result contract.
- Keep loading, error, empty, and success states distinct.
- Support desktop and mobile layouts.
- Make baseline-versus-participant comparison visually clear.
- Make timeline and evidence navigation primary interactions.
- Synchronize playback, waveform, feature overlays, timeline, and transcript.
- Provide accessible labels, controls, focus states, and text alternatives.

### Prohibited behavior

- Do not implement FFT, MFCC, pitch, scoring, thresholding, or other scientific calculations in UI components.
- Do not hardcode analysis results into visual components.
- Do not duplicate backend formulas in browser code merely to redraw a score.
- Do not hide missing evidence behind a polished empty chart.
- Do not use decorative UI to obscure the analytical workflow.

### Responsive and state rules

Every major view must define:

- Loading state.
- Empty state.
- Recoverable error state.
- Blocking validation state.
- Partial-data or low-confidence state.
- Successful result state.

Charts must remain interpretable on narrow screens. If a chart cannot be fully interactive on mobile, provide a readable summary and an accessible finding list.

---

## 10. API and Data Contracts

Define explicit request and response schemas for all boundaries.

### Contract rules

- Do not rely on undocumented object shapes.
- Validate requests at the boundary.
- Validate responses before rendering or persisting them.
- Version schemas when a breaking change is introduced.
- Use stable IDs for runs, findings, recordings, transcripts, and configurations.
- Include units and timestamp conventions in schemas.
- Represent missing, invalid, and unavailable values explicitly.
- Keep error responses structured and understandable.
- Keep provenance fields available in analysis responses.

### Minimum analysis result contract

An analysis result should expose, directly or through documented references:

- Run ID and status.
- Input artifact IDs/checksums.
- Transcript and baseline IDs.
- Alignment status and confidence/coverage.
- Feature/configuration/model versions.
- Rubric version.
- Overall score and dimension breakdown.
- Findings with temporal and evidence fields.
- Warnings, exclusions, and errors.
- Reproducibility/provenance metadata.

Do not make the frontend reconstruct scientific meaning from raw undocumented arrays.

---

## 11. Data Validation

Validate inputs before expensive processing and validate outputs before presentation.

### Required input validation

- Audio type and supported codec/container.
- File readability and size limits.
- Audio duration and channel structure.
- Sample rate and conversion behavior.
- Speech presence and excessive silence.
- Transcript availability and canonical normalization.
- Exact baseline transcript compatibility.
- Alignment validity and coverage.

### Required result validation

- Timestamp ordering and non-negative values.
- Finding intervals within audio bounds.
- Affected transcript units that exist in the alignment.
- Feature values with valid units and expected ranges.
- Score ranges and dimension weights.
- Required provenance fields.
- Correct representation of missing features.
- Schema compatibility before rendering or export.

Invalid results must be rejected, marked invalid, or shown as diagnostic-only. They must not be silently converted into valid-looking scores.

---

## 12. Error Handling and Diagnostics

Errors must be understandable to users while preserving developer diagnostics.

### User-facing rules

- Explain what failed.
- Identify the affected input or pipeline stage.
- Explain whether the user can fix the problem.
- Offer a retry or correction path where appropriate.
- Distinguish blocking errors, warnings, and informational notices.
- Never display a successful score after a blocking stage failed.

### Developer-facing rules

- Log a stable error category and run ID.
- Include useful diagnostics without leaking secrets or unnecessary raw audio/transcript data.
- Preserve stack traces in appropriate development/controlled logs.
- Record stage duration and configuration ID where useful.
- Do not use logs as the only source of user-facing explanations.

---

## 13. Performance and Caching

Avoid unnecessary repeated audio processing.

### Performance rules

- Cache expensive feature extraction when the source checksum and configuration match.
- Cache alignment results only when transcript, audio, alignment engine, model, and settings match.
- Invalidate caches when relevant inputs or versions change.
- Avoid recalculating the same feature in multiple pipeline stages.
- Make processing progress visible for long-running analysis.
- Prefer bounded, measurable performance targets over unsupported guarantees.
- Keep cache keys deterministic and inspectable.
- Do not trade away evidence, reproducibility, or correctness for an unmeasured speed improvement.

Cached data must preserve provenance and must never make an analysis appear newer or more authoritative than its source configuration.

---

## 14. Reproducibility

A result must be reproducible from documented inputs and versions.

### Required controls

- Pin dependencies or use a lockfile.
- Version preprocessing, alignment, feature, scoring, and explanation configurations.
- Record dataset and split versions.
- Record model and engine versions.
- Use deterministic processing wherever possible.
- Set and record random seeds when randomness exists.
- Record input checksums and output metadata.
- Provide a documented reference command or workflow.
- Keep a fixture or permitted sample that can exercise the complete pipeline.
- Document numerical tolerances for comparisons.

A reproducibility failure must be investigated and documented; it must not be hidden by manually editing output artifacts.

---

## 15. Code Quality

Code must be understandable, focused, and maintainable.

### Required standards

- Prefer small, focused modules.
- Use meaningful names for variables, functions, types, files, and configuration keys.
- Keep one source of truth for each calculation and business rule.
- Remove dead code and unused dependencies.
- Avoid duplicated logic.
- Replace unexplained magic numbers with named, documented configuration.
- Keep functions and components responsible for one coherent concern.
- Validate external input at boundaries.
- Make units and coordinate systems explicit.
- Use environment variables for deployment configuration.
- Never hardcode absolute paths.
- Do not commit secrets, API keys, tokens, or private credentials.
- Preserve working behavior unless the change is required by the product or a verified defect.

### Change-size rule

Make the smallest coherent change that satisfies the requirement. Do not rewrite working modules merely for stylistic reasons. Refactor only when it reduces risk, removes duplication, enables a required behavior, or fixes a verified defect.

---

## 16. UI Rules

The interface must prioritize the analytical workflow:

1. Evidence.
2. Timeline.
3. Comparison.
4. Explanation.
5. Score interpretation.

### UI requirements

- Make the selected baseline and participant recording explicit.
- Keep flaw regions visually connected to their evidence.
- Display units, timestamps, and confidence.
- Let users navigate from a finding to playback and back.
- Show score breakdowns rather than only a single overall number.
- Expose alignment and coverage limitations.
- Keep explanations next to the measurements that support them.
- Use consistent terminology from the PRD and schemas.

### UI prohibitions

- Avoid decorative UI that hides the analytical workflow.
- Do not use a large score as a substitute for evidence.
- Do not imply certainty when alignment or feature confidence is low.
- Do not use ambiguous labels such as “bad voice” or “poor speaker.”
- Do not present unsupported psychological interpretations.

---

## 17. Accessibility

Accessibility is a product requirement, not a final polish step.

### Required behavior

- Keyboard-accessible playback, seeking, tabs, filters, and finding navigation.
- Readable contrast for text, controls, charts, and overlays.
- Visible focus states.
- Explicit labels for controls and form inputs.
- Text alternatives or summaries for important chart information.
- Do not rely only on color to identify flaws or distinguish baseline from participant.
- Provide patterns, labels, icons with text, or line styles in addition to color.
- Ensure transcript highlighting has a non-color indicator.
- Make errors and validation messages associated with their relevant controls.
- Avoid flashing or rapidly changing visual effects.
- Ensure responsive layouts do not hide required evidence on mobile.

---

## 18. Security and Privacy

Treat audio, transcripts, and speaker metadata as sensitive data.

### Security rules

- Validate uploaded files before decoding or processing.
- Enforce documented size, type, and duration limits.
- Never expose secrets in source code, logs, client bundles, or screenshots.
- Never commit API keys or credentials.
- Use environment variables or the approved secret-management mechanism.
- Restrict access to uploaded audio, transcripts, and exports.
- Avoid guessable public URLs for private artifacts.
- Sanitize filenames and user-provided text where required.
- Document temporary-file cleanup and data retention behavior.
- Do not permanently store audio unless explicitly required and documented.
- Delete or expire temporary processing artifacts according to the documented policy.
- Do not use recordings for a new purpose without appropriate authorization.

### Privacy language

The product may describe measurable acoustic and timing differences. It must not imply that those measurements reveal a person’s mental state, personality, or intent.

---

## 19. Testing

Critical logic must have automated tests before it is treated as complete.

### Required test coverage

- Transcript normalization.
- Exact transcript matching.
- Audio and input validation.
- Alignment output validation.
- Speaker/feature normalization.
- Feature extraction.
- Missing-feature handling.
- Deviation calculation.
- Temporal grounding and timestamp bounds.
- Thresholding and region merging.
- Scoring and configurable weights.
- Explanation generation.
- Result schema validation.
- Export serialization.
- At least one end-to-end paired fixture.

### Test principles

- Test both positive findings and near-perfect/control cases.
- Test boundary conditions around thresholds and minimum durations.
- Test invalid, missing, and low-confidence data.
- Test deterministic repeatability with the same fixture and configuration.
- Test that UI components render structured results without performing scientific calculations.
- Do not rely only on screenshots or manually inspected demo output.

---

## 20. Agent Behavior and Change Workflow

Before modifying code, every coding agent must:

1. Read `memory.md` if it exists.
2. Read `prd.md`.
3. Read `architecture.md` if it exists.
4. Read `design.md` if it exists.
5. Read `tasks.md` if it exists.
6. Inspect the existing implementation and relevant tests.
7. Identify the source-of-truth requirement being implemented.
8. Make the smallest coherent change.
9. Add or update tests for critical behavior.
10. Run the relevant tests and build/lint/type checks.
11. Inspect the resulting diff for accidental scope expansion.
12. Update documentation if architecture, contracts, configuration, or product behavior changed.
13. Report assumptions, validation results, and known limitations.

### Agent prohibitions

- Do not silently change product scope.
- Do not invent data or metadata to make a demo pass.
- Do not rewrite working modules merely for style.
- Do not bypass validation to make an analysis complete.
- Do not hide low confidence or missing evidence.
- Do not make unsupported scientific or psychological claims.
- Do not introduce undocumented object shapes or configuration keys.
- Do not claim tests passed unless they were actually run.

---

## 21. Hackathon Priority Order

When time or engineering capacity is limited, prioritize in this order:

1. **Temporal grounding.**
2. **Contrastive dataset quality and exact pairing.**
3. **Acoustic feature evidence.**
4. **Causal explanations tied to measurements.**
5. **Dashboard visualization and synchronization.**
6. **Reproducibility and code quality.**

Do not sacrifice the core contrastive comparison for decorative polish. Do not add unnecessary features outside the Track C problem statement merely to increase complexity.

---

## 22. Pull Request and Review Checklist

A change is ready for review only when the applicable items below are satisfied.

### Product correctness

- [ ] The change supports the same-text aligned-baseline principle.
- [ ] The change is consistent with `prd.md` and the official problem statement.
- [ ] No unrelated feature or scope expansion was introduced.

### Data and analysis

- [ ] Inputs and outputs are validated.
- [ ] Provenance is preserved.
- [ ] Missing or low-confidence evidence is explicit.
- [ ] Feature calculations remain in the analysis layer.
- [ ] Temporal findings have valid timestamps and affected units.
- [ ] Explanations are traceable to measured values.
- [ ] No unsupported psychological claims were introduced.

### Scoring and contracts

- [ ] Scoring is deterministic.
- [ ] Configuration is versioned and not hidden in UI code.
- [ ] API/request/response schemas are explicit.
- [ ] Units, ranges, and missing-data behavior are documented.

### Frontend and accessibility

- [ ] Loading, empty, error, and partial-data states are handled.
- [ ] Evidence is synchronized with timeline, transcript, and playback.
- [ ] Keyboard access, labels, contrast, focus, and non-color cues are present.
- [ ] Desktop and mobile layouts remain usable.

### Quality and reproducibility

- [ ] Critical tests were added or updated.
- [ ] Relevant tests/build/lint/type checks were run.
- [ ] Dependencies and configurations remain pinned/versioned.
- [ ] No secrets or hardcoded absolute paths were added.
- [ ] Documentation was updated for behavior or architecture changes.
- [ ] The diff is the smallest coherent change.

---

## 23. Final Rule

When forced to choose between a more elaborate feature and a more trustworthy result, choose the trustworthy result.

The project succeeds by making a speech-delivery difference **measurable, aligned, temporally grounded, explainable, reproducible, and inspectable**—not by producing the largest feature list.
