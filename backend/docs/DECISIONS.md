# Design Decisions Log

This document records architectural, mathematical, and algorithmic design choices made during development. Each decision is explained in plain terms for defense in technical reviews.

---

## M1: Alignment Backend and Dependency Strategy

### Decision: Python 3.11 with WhisperX and Parselmouth
- **Choice**: Standardized on Python 3.11 with `praat-parselmouth==0.4.7`, `whisperx==3.8.6`, and `torch==2.8.0` / `torchaudio==2.8.0`. Configured `ALIGNMENT_BACKEND = "whisperx"` with fallbacks to `torchaudio` and `proportional`.
- **Rationale**: Python 3.11 is the most stable runtime that provides pre-built Windows wheels for PyTorch, CTranslate2, and Parselmouth without C++ compilation errors. WhisperX provides phone-level forced alignment that handles continuous speech and conversational speaking rates significantly better than character-proportional heuristics. Keeping an explicit fallback hierarchy ensures the evaluation pipeline remains functional across varying compute constraints.

---

## M2: Forced Alignment and Unaligned Word Interpolation

### Decision: Caching and Hybrid Alignment Fallback
- **Choice**: Implemented `src/features/forced_align.py` with content-hashed JSON caching in `data/labels/alignments/`. Words dropped or missed by acoustic phone aligners are linearly interpolated between neighboring aligned anchors and flagged with `aligned=False`.
- **Rationale**: Real acoustic aligners (like wav2vec2 or CTC) occasionally drop rapid function words or words degraded by noise. Blindly skipping these words creates catastrophic timestamp drift for all downstream rubric metrics, whereas proportional interpolation guarantees that the transcript ordering and time continuity are strictly preserved. Caching alignment by SHA-256 hash avoids redundant multi-second model inference on repeated pipeline evaluations.


---

## M3: Spectral Features, Per-Word Features, and Filler Detector

### Decision: Percentile-Based Clarity (Not Traditional SNR)
- **Choice**: `compute_clarity_snr` estimates SNR from the 95th vs 10th percentile of per-frame RMS in dB, normalized by an ideal SNR of 20 dB. Result is clipped to [0, 1].
- **Rationale**: Traditional SNR requires a separate noise reference (or a Voice Activity Detector). The percentile spread is a reference-free proxy: loud speech frames cluster at the 95th percentile and background noise at the 10th. Dividing by 20 dB maps a typical clean recording to a score of ~1. This works without needing a silent reference segment, which we cannot guarantee in contestant audio.

### Decision: Filler Detection via Energy + Voiced Pitch Criteria
- **Choice**: A gap between two word timings is classified as a FILLER region when it is at least 200 ms, has low mean energy (z < 0.5), AND has voiced activity (at least one finite pitch frame) with flat pitch (std < 1.2 semitones).
- **Rationale**: Silence satisfies condition 1 and 2 but not 3 (no voiced frames). Real speech content satisfies condition 3 but not 2 (higher energy and pitch variation). Filler sounds (uh, um) are uniquely both voiced and acoustically flat. The three-way conjunction minimizes false positives from either silence or breaths being mistakenly flagged as fillers.

### Decision: Word Features Use Frame Rounding (Not Flooring)
- **Choice**: Converting word start/end times to frame indices uses `round(t * sr / hop_length)` rather than `floor(t * sr / hop_length)`.
- **Rationale**: Librosa's centered framing places frame `i` at time `i * hop_length / sr`. Using `round()` minimizes the average alignment error between word boundaries and the nearest frame center. Flooring would introduce a systematic negative bias (words always starting slightly earlier than they actually do in the feature array).

---

## M4: Dataset Generator and Severity Mapping

### Decision: In-Timeline Label Offsets for Duration-Shifting Flaws
- **Choice**: When slow/fast stretch or silence/filler insertion changes an audio segment's length, subsequent label start/end times are computed from the *new* audio timeline: `label_end = region_end + delta_s`.
- **Rationale**: Downstream detectors evaluate flawed files against their own actual elapsed time, not the original source recording's timeline. If timestamps were left at source coordinates, injected flaws occurring after a stretched segment would show artificial time displacement and fail IoU evaluation checks.

### Decision: Parselmouth-Based Monotone Injection (No Librosa Fallback)
- **Choice**: Used `praat-parselmouth` directly to flatten the pitch tier via PSOLA overlap-add synthesis.
- **Rationale**: Praat's pitch-tier manipulation modifies fundamental frequency without introducing robotic phase artifacts or altering formants, which simple pitch shifters do. Since Parselmouth 0.4.7 installed cleanly in M1, no fallback was required.

---

## M5: Word Matching and Contrastive Detection

### Decision: PAUSE_MISPLACED Omission vs. PAUSE_MISSING Inclusion
- **Choice**: Implemented `PAUSE_MISSING` (emitted when baseline had an inter-word gap > 300 ms that candidate omitted) but removed `PAUSE_MISPLACED` from active detection.
- **Rationale**: Distinguishing a "misplaced" pause from a legitimate rhetorical stylistic pause requires deep semantic understanding of clause boundaries (an oracle syntactic parser). Emitting `PAUSE_MISPLACED` on heuristic rules produces high false-positive rates that cannot be defended in technical evaluation. `PAUSE_MISSING` is concrete and directly verifiable from contrastive timing.

### Decision: NaN-Aware Pitch Range Ratio
- **Choice**: In `compare_words`, if either the baseline or candidate word has fewer than 2 voiced frames, `pitch_range_ratio` returns `float("nan")` and is explicitly ignored by pitch flaw detectors.
- **Rationale**: Short words (like "the", "a", "it") often consist entirely of unvoiced consonants or very brief vowels that pYIN classifies as unvoiced. Treating missing pitch as "0 Hz" or "zero range" would falsely flag every unvoiced function word as a monotone flaw.

---

## M6: Evaluation Metric Design

### Decision: 1-D Temporal IoU with Greedy Bipartite Matching
- **Choice**: Flaw detections are evaluated against ground truth intervals using 1-D temporal Intersection over Union (IoU), with an acceptance threshold of IoU ≥ 0.3. Matches are assigned greedily by highest IoU, with a strict 1-to-1 constraint.
- **Rationale**: Frame-level accuracy metrics are biased by duration (a 5-second silence flaw would dwarf ten 300 ms filler flaws). Event-level IoU treats each flaw region as a distinct semantic event. Setting the threshold at 0.3 accommodates small timing boundaries arising from frame-hop discretization (~32 ms) and acoustic co-articulation at word edges without allowing false positive matches.

---

## M7: Scoring Model Architecture

### Decision: Exponential Decay Scoring vs. Hard Linear Cutoffs
- **Choice**: Implemented `exponential_score(deviation, k) = exp(-k * deviation)` with decay rates `k` configured per rubric dimension in `src/config.py`. Values are normalized to [0, 1] to satisfy `schema.py` invariants (with 100× display multiplier).
- **Rationale**: Linear penalty functions create artificial "cliffs" where a 1 WPM deviation change suddenly zeros out a contestant's score. Exponential decay reflects human perceptual tolerance: small deviations incur negligible penalties, while severe deviations produce smooth asymptotic degradation without hard zero-clamping. Configuring dimension weights in `RUBRIC_WEIGHTS` allows fine-tuning overall composite score sensitivity without changing business logic.

---

## M8: FlawType Enum and Label Casing Consistency

### Decision: Universal UPPER_CASE for FlawType Values
- **Choice**: Standardized all `FlawType` enum values in `src/schema.py` to uppercase strings (e.g. `FlawType.PACE_TOO_FAST = "PACE_TOO_FAST"`), matching ground-truth labels in `data/labels/injections.csv`. Evaluators and tests reference enum members directly rather than hardcoded string literals.
- **Rationale**: A case mismatch between uppercase ground-truth injection labels and lowercase detector outputs caused 0 true positives during initial batch evaluation. Enforcing uppercase across all schema definitions, serialization layers, and evaluation scripts guarantees end-to-end consistency without fragile runtime `.upper()` or `.lower()` conversions.

---

## M9: Contrastive vs Standalone Flaw Detection Gating

### Decision: Baseline-Gated Contrastive Detection
- **Choice**: When a reference baseline recording is provided to `evaluate()`, only contrastive match-based detectors emit flaw regions; standalone absolute-threshold detectors run only in baseline-free mode.
- **Rationale**: An ideal reference speech naturally contains expressive stylistic features (e.g., steady pitch on a clause, deliberate rhetorical pauses) that absolute heuristic thresholds falsely flag as flaws. In contrastive evaluation, flaws represent meaningful deviations from the exemplar speaker rather than deviations from arbitrary static bounds. Comparing an ideal recording with itself now deterministically yields zero flaw regions.


---

## M10: Baseline-Relative Rubric Scoring

### Decision: Deviation-Based Scoring with Global + Localized Blend

- **Choice**: When a baseline recording is provided, all four rubric dimensions (pace, pause_pattern, pitch_variation, energy_consistency) now use baseline-relative deviation scores (`score = exp(-k * deviation)`) instead of absolute ideal-range penalties. The four new functions (`score_pace_relative`, `score_pause_pattern_relative`, `score_pitch_variation_relative`, `score_energy_consistency_relative`) live in `src/analysis/scoring.py`; the original absolute-range scorers remain for no-baseline mode only.
- **Formulas**:
  - **pace**: deviation = mean |log(duration_ratio)| per matched word
  - **pause_pattern**: deviation = blend of (extra_candidate_pause_time / cand_duration) and worst-window (extra_pause_delta / window_span)
  - **pitch_variation**: deviation = mean |log(pitch_range_ratio)| per voiced matched word; falls back to whole-file std log-ratio if no per-word data
  - **energy_consistency**: deviation = mean |energy_delta_z| per matched word
- **Global + Local Blend**: `final_deviation = (1 - SCORE_LOCAL_WEIGHT) * global_mean + SCORE_LOCAL_WEIGHT * worst_window_mean` where SCORE_WINDOW_WORDS=8 and SCORE_LOCAL_WEIGHT=0.5. This ensures a 3-6 word flaw isn't diluted by 100 ideal words: the worst 8-word sliding window captures the peak deviation even when the file-wide mean is low.
- **Verification**: speech1.wav vs itself gives composite 1.0000, all dimensions 1.0000. pause_excessive sev1/3/5 composites are 0.7413, 0.5401, 0.4405 (monotonically decreasing as required).
- **k values**: SCORE_K_PACE_REL=6.0, SCORE_K_PITCH_REL=3.0, SCORE_K_ENERGY_REL=2.0, SCORE_K_PAUSE_REL=8.0 (all in src/config.py). Pause k is highest because the pause local-window fraction can exceed 1.0 for large insertions.
- **Summary fix**: `_build_summary` now uses plain ASCII `-` instead of Unicode em-dash (eliminates UTF-8/Latin-1 garble), overrides grade to at most "fair" when any dimension < 0.70 or any flaw region exists, and cites the lowest-scoring dimension and detected flaw types.

