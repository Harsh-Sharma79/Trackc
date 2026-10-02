"""
src/config.py
=============
Central configuration for the Speech Evaluation System.
All constants live here; no other module may hardcode values.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16_000          # Hz, mono
CHANNELS: int = 1

# STFT / feature-frame parameters
FRAME_LENGTH: int = 2_048          # samples  (~128 ms at 16 kHz)
HOP_LENGTH: int = 512              # samples  (~32 ms at 16 kHz)
N_FFT: int = 2_048

# pYIN pitch-tracking parameters
FMIN_HZ: float = 60.0              # Hz  – lowest expected F0
FMAX_HZ: float = 600.0             # Hz  – highest expected F0
PITCH_FILL_NA: float = float("nan")  # semitones used when frame is unvoiced

# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------
PAUSE_ENERGY_THRESHOLD: float = -40.0   # dBFS below which a frame is silence
PAUSE_MIN_DURATION: float = 0.25         # seconds; shorter gaps are not pauses
PAUSE_IDEAL_MAX: float = 0.6             # seconds; pauses longer than this penalised

# ---------------------------------------------------------------------------
# Scoring / thresholds
# ---------------------------------------------------------------------------
PACE_IDEAL_WPM_MIN: float = 120.0
PACE_IDEAL_WPM_MAX: float = 160.0
PITCH_VAR_IDEAL_MIN: float = 2.0        # semitones std-dev
PITCH_VAR_IDEAL_MAX: float = 8.0
ENERGY_Z_CLIP: float = 3.0              # clip energy z-scores beyond ±3σ
SEVERITY_SCALE_FACTOR: float = 1.0      # multiplier before sigmoid/clip

# Region consolidation.  A quarter-second is long enough to avoid reporting
# transient frame/word-level noise as a listener-perceivable speech flaw.
MIN_FLAW_DURATION_SEC: float = 0.25
FLAW_MERGE_GAP_SEC: float = 0.20         # join same-type detections across brief gaps
MONOTONE_MIN_REGION_SEC: float = 0.75    # require a sustained voiced monotone run

# ---------------------------------------------------------------------------
# Spectral, Clarity & Filler parameters
# ---------------------------------------------------------------------------
N_MFCC: int = 13                        # number of MFCC coefficients
CLARITY_LOW_PERCENTILE: float = 10.0    # noise floor percentile
CLARITY_HIGH_PERCENTILE: float = 95.0   # speech peak percentile
CLARITY_IDEAL_SNR_DB: float = 20.0      # target SNR in dB
FILLER_MIN_DURATION: float = 0.20       # minimum duration for filler gap (seconds)
FILLER_PITCH_STD_MAX: float = 1.2       # semitones std-dev ceiling for flat filler sound
FILLER_ENERGY_Z_MAX: float = 0.5        # energy z-score ceiling for filler sound


# ---------------------------------------------------------------------------
# Paths  (relative to project root, resolved at import time)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
FLAWED_DIR: Path = DATA_DIR / "flawed"
LABELS_DIR: Path = DATA_DIR / "labels"
ALIGNMENT_CACHE_DIR: Path = LABELS_DIR / "alignments"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Forced Alignment Backend ("whisperx" | "torchaudio" | "proportional")
# ---------------------------------------------------------------------------
ALIGNMENT_BACKEND: str = "torchaudio"

# ---------------------------------------------------------------------------
# DTW alignment
# ---------------------------------------------------------------------------
DTW_RADIUS: int = 30               # Sakoe-Chiba band (frames)


# ---------------------------------------------------------------------------
# M4 Dataset Generation — Flaw Injection Parameters
# ---------------------------------------------------------------------------
# Region selection: number of words per injected region
INJECT_REGION_MIN_WORDS: int = 3
INJECT_REGION_MAX_WORDS: int = 6

# Fade duration (seconds) applied at region edges to avoid clicks
INJECT_FADE_S: float = 0.010  # 10 ms

# Number of combined-flaw files to generate
INJECT_COMBINED_COUNT: int = 10

# Severity level → stretch/compression ratio for PACE flaws
# Keys are severity levels 1–5; values are time-stretch rates (>1 = slower, <1 = faster)
INJECT_SLOW_RATES: dict[int, float] = {1: 1.15, 2: 1.30, 3: 1.50, 4: 1.75, 5: 2.00}
INJECT_FAST_RATES: dict[int, float] = {1: 0.88, 2: 0.78, 3: 0.67, 4: 0.57, 5: 0.50}

# Severity level → silence duration (seconds) inserted for PAUSE_EXCESSIVE
INJECT_PAUSE_DURATIONS: dict[int, float] = {
    1: 0.8, 2: 1.2, 3: 1.8, 4: 2.5, 5: 3.5
}

# Severity level → dB reduction for ENERGY_LOW
INJECT_ENERGY_DB_DROP: dict[int, float] = {
    1: 6.0, 2: 10.0, 3: 15.0, 4: 20.0, 5: 30.0
}

# Severity level → target SNR (dB) for adding Gaussian noise (UNCLEAR)
INJECT_NOISE_SNR: dict[int, float] = {
    1: 25.0, 2: 20.0, 3: 15.0, 4: 10.0, 5: 5.0
}

# Severity level → pitch flattening amount (fraction of pitch variation removed, 0–1)
# 1.0 = fully flattened to median; 0.0 = unchanged
INJECT_MONOTONE_FLATTEN: dict[int, float] = {
    1: 0.40, 2: 0.60, 3: 0.75, 4: 0.88, 5: 1.00
}

# Synthetic filler signal: duration in seconds per severity level
INJECT_FILLER_DURATIONS: dict[int, float] = {
    1: 0.20, 2: 0.35, 3: 0.50, 4: 0.70, 5: 1.00
}

# Synthetic filler signal parameters (low buzz at ~150 Hz)
INJECT_FILLER_FREQ_HZ: float = 150.0
INJECT_FILLER_AMPLITUDE: float = 0.05   # relative to full scale


# ---------------------------------------------------------------------------
# M5 Word Matching and Contrastive Detection Thresholds
# ---------------------------------------------------------------------------
# Duration ratio thresholds relative to baseline word duration
# ratio = candidate_duration / baseline_duration
MATCH_SLOW_RATIO_MIN: float = 1.30    # candidate > 1.30× baseline → too slow
MATCH_FAST_RATIO_MAX: float = 0.75    # candidate < 0.75× baseline → too fast

# Pause-before delta (candidate_pause - baseline_pause) in seconds
MATCH_PAUSE_EXCESS_DELTA: float = 0.40   # pause > baseline + 0.40 s → excessive
MATCH_PAUSE_MISSING_DELTA: float = -0.25  # pause < baseline - 0.25 s (and baseline > 0.3 s)
MATCH_PAUSE_BASELINE_MIN: float = 0.30   # baseline pause must exceed this for PAUSE_MISSING

# Minimum extra duration for an absorbed (intra-word) pause to count as PAUSE_EXCESSIVE.
# Candidate pause must exceed its nearest baseline counterpart by this many seconds.
# 0.80 s was chosen to sit above natural single-breath pause variation; 0.45 s is
# used because the synthetic benchmark injects 0.8 s silences at severity 1 and
# the fractional index mapping already guarantees zero extra on ideal-vs-ideal
# reads (every candidate pause maps to its own baseline counterpart).
MATCH_ABSORBED_PAUSE_EXCESS: float = 0.45

# ---------------------------------------------------------------------------
# Contrastive pitch (monotone) detection — paired per-word voiced dispersion
# ---------------------------------------------------------------------------
# Words are only compared when BOTH sides carry enough voiced evidence, and the
# baseline word must be expressive enough for the ratio to be meaningful.
# Comparing a word's pitch *std* (not max-min range) avoids the unbounded
# |log ratio| outliers that a near-zero baseline range produces.
MATCH_PITCH_STD_RATIO_MIN: float = 0.55      # cand_std / base_std below this → monotone
MATCH_PITCH_MIN_BASE_STD_ST: float = 0.30    # baseline word must vary this much (semitones)
MATCH_PITCH_MIN_VOICED_FRAMES: int = 4       # minimum voiced frames per side per word
MATCH_PITCH_MAX_ABS_LOG_DEV: float = 1.2     # cap per-word |log ratio| (outlier guard)

# ---------------------------------------------------------------------------
# Contrastive filler detection (acoustic, no ASR)
# ---------------------------------------------------------------------------
# A candidate inter-word gap is a probable filler when it is longer than its
# baseline counterpart AND is voiced throughout (unlike silence) with a flat,
# low-energy acoustic profile. The voiced-fraction gate is what separates a
# filler sound from an inserted pause: silence has no (or sporadic) voiced frames.
MATCH_FILLER_MIN_EXTRA_GAP_S: float = 0.15    # extra gap vs baseline counterpart
MATCH_FILLER_MIN_VOICED_FRACTION: float = 0.40  # fraction of gap frames with F0
MATCH_FILLER_PITCH_STD_MAX: float = 1.5       # semitones over voiced gap frames

# ---------------------------------------------------------------------------
# Contrastive clarity (UNCLEAR) detection — spectral evidence
# ---------------------------------------------------------------------------
# Low-pass filtering / added broadband noise changes the spectral centroid and
# the Wiener entropy (flatness) of the affected words relative to the baseline.
MATCH_CLARITY_CENTROID_RATIO_MIN: float = 1.25   # cand centroid / base centroid
MATCH_CLARITY_FLATNESS_DELTA_MIN: float = 0.03    # flatness(cand) - flatness(base)

# ---------------------------------------------------------------------------
# Region consolidation tolerance
# ---------------------------------------------------------------------------
# Guards against float noise (e.g. 26.878 - 26.678 == 0.2000000000000028)
# splitting one flaw into two adjacent regions.
REGION_MERGE_EPS_S: float = 1e-6


# Energy delta threshold (candidate energy z - baseline energy z)
MATCH_ENERGY_LOW_DELTA: float = -1.2     # candidate energy z < baseline - 1.2 → low

# Minimum number of adjacent flagged words to emit a FlawRegion
MATCH_MIN_REGION_WORDS: int = 1

# Minimum difflib SequenceMatcher ratio to accept a word match
MATCH_SIMILARITY_THRESHOLD: float = 0.6


# ---------------------------------------------------------------------------
# M7 Rubric Scoring — Exponential Decay Parameters
# Formula: score = exp(-k * mean_deviation), normalized to [0, 1]
# (Display score = 100 * normalized_score)
# ---------------------------------------------------------------------------
SCORE_K_PACE: float = 1.5
SCORE_K_PITCH: float = 0.8
SCORE_K_ENERGY: float = 1.0
SCORE_K_PAUSE: float = 1.2

RUBRIC_WEIGHTS: dict[str, float] = {
    "pace": 1.0,
    "pitch_variation": 1.0,
    "energy_consistency": 1.0,
    "pause_pattern": 1.0,
}

# ---------------------------------------------------------------------------
# Baseline-relative scoring — combined global+local deviation
# ---------------------------------------------------------------------------
# Sliding window size (number of words) for localized worst-case deviation
SCORE_WINDOW_WORDS: int = 8

# Blend weight: final_deviation = (1 - w) * global_dev + w * local_dev
# local_dev is the worst-window deviation; 0.0 = purely global, 1.0 = purely local.
# 0.5 weights them equally, ensuring a 3-6 word injected flaw has strong signal.
SCORE_LOCAL_WEIGHT: float = 0.5

# Baseline-relative k values (same as above but separated for clarity)
SCORE_K_PACE_REL: float = 6.0    # |log(dur_ratio)|: 0.18 nats → e^(-6*0.18)≈0.34
SCORE_K_PITCH_REL: float = 3.0   # |log(pitch_std_ratio)|: penalise large pitch changes
SCORE_K_ENERGY_REL: float = 2.0  # mean |energy_delta_z|
SCORE_K_PAUSE_REL: float = 8.0   # extra_pause_frac: 0.08 → e^(-8*0.08)≈0.53

