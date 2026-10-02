"""
src/pipeline.py
===============
End-to-end speech evaluation pipeline.

Given a candidate audio file + transcript and a baseline audio file,
produces an EvaluationResult JSON-serialisable object.

Usage:
    from src.pipeline import evaluate
    result = evaluate(
        candidate_path="data/flawed/speaker1.wav",
        baseline_path="data/raw/baseline.wav",
        transcript="Hello world, this is a test speech.",
        audio_id="speaker1_run1",
    )
    print(result.model_dump_json(indent=2))
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import ALIGNMENT_BACKEND, SEED
from src.analysis.detector import detect_all
from src.analysis.scoring import (
    compute_composite,
    score_energy_consistency,
    score_pace,
    score_pause_pattern,
    score_pitch_variation,
    score_energy_consistency_relative,
    score_pace_relative,
    score_pause_pattern_relative,
    score_pitch_variation_relative,
)
from src.explain.templates import explain
from src.features.alignment import align_features
from src.features.audio import AudioFeatures, extract_all
from src.features.forced_align import align_transcript, alignment_backend_availability
from src.features.spectral import compute_spectral_centroid, compute_spectral_flatness
from src.features.transcript import Word, estimate_word_times, tokenize, words_to_pace_wpm
from src.schema import EvaluationResult, FlawRegion, RubricScore
from src.analysis.match import compare_words, detect_from_matches, match_words

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineReport:
    """Non-scientific execution facts about one evaluation run.

    These values describe *how* the result was produced (which alignment
    backend, whether an explicit degradation happened, how long each stage
    took).  They are deliberately kept out of EvaluationResult, which remains
    the scientific output contract.
    """

    alignment_backend: str = ""
    alignment_degraded: bool = False
    has_baseline: bool = False
    words_total: int = 0
    words_unaligned: int = 0
    stage_seconds: dict[str, float] = field(default_factory=dict)


# Small in-process cache for per-frame spectral bundles.  Spectral arrays are a
# pure function of the waveform, so repeated evaluations against the same
# baseline file (a very common API pattern) do not need to re-run the STFT.
_SPECTRAL_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _spectral_bundle(
    waveform: np.ndarray, cache_key: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (spectral_centroid, spectral_flatness) for a waveform, cached."""
    if cache_key is not None:
        cached = _SPECTRAL_CACHE.get(cache_key)
        if cached is not None:
            return cached
    bundle = (
        compute_spectral_centroid(waveform),
        compute_spectral_flatness(waveform),
    )
    if cache_key is not None:
        _SPECTRAL_CACHE[cache_key] = bundle
    return bundle


def clear_caches() -> None:
    """Drop the in-process spectral cache (exposed for tests)."""
    _SPECTRAL_CACHE.clear()


def _seed_all() -> None:
    """Seed Python's random and NumPy for reproducibility."""
    random.seed(SEED)
    np.random.seed(SEED)


def _build_summary(
    composite: float,
    rubric_scores: list[RubricScore],
    flaw_regions: list[FlawRegion],
) -> str:
    """Build a one-paragraph human-readable summary of the evaluation.

    Rules:
    - Grade is derived from the composite score, but overridden to at most
      'fair' when any dimension score is below 0.70 or any flaw regions exist.
    - Cites the lowest-scoring dimension and its score.
    - Cites the flaw region count when non-zero.
    - Uses only plain ASCII punctuation to avoid encoding artefacts.

    Args:
        composite:     Overall composite score.
        rubric_scores: Per-dimension scores.
        flaw_regions:  Detected flaw regions.

    Returns:
        Summary string (plain ASCII, no Unicode special chars).
    """
    # Determine grade from composite
    grade = (
        "excellent" if composite >= 0.85
        else "good" if composite >= 0.70
        else "fair" if composite >= 0.55
        else "needs improvement"
    )

    # Override grade if any dimension is low or flaws detected
    n_flaws = len(flaw_regions)
    min_score = min((s.score for s in rubric_scores), default=1.0)
    if min_score < 0.70 or n_flaws > 0:
        if grade in ("excellent", "good"):
            grade = "fair" if composite >= 0.55 else "needs improvement"

    # Find worst dimension
    worst = min(rubric_scores, key=lambda s: s.score) if rubric_scores else None
    dims = ", ".join(
        f"{s.dimension.value} ({s.score:.2f})" for s in rubric_scores
    )

    # Flaw string
    if n_flaws == 0:
        flaw_str = "No flaw regions were detected."
    else:
        types = sorted({r.flaw_type.value for r in flaw_regions})
        flaw_str = (
            f"{n_flaws} flaw region(s) detected "
            f"({', '.join(types)})."
        )

    # Worst dimension citation
    worst_str = ""
    if worst and worst.score < 0.90:
        worst_str = (
            f" Lowest dimension: {worst.dimension.value} ({worst.score:.2f})."
        )

    return (
        f"Overall performance is {grade} with a composite score of {composite:.2f}. "
        f"Rubric breakdown - {dims}.{worst_str} {flaw_str}"
    )


def _resolve_transcript(transcript: str) -> str:
    """Resolve a transcript argument that may be a path to a .txt file."""
    try:
        tr_path = Path(transcript)
        if tr_path.is_file():
            return tr_path.read_text(encoding="utf-8-sig").strip()
    except (OSError, ValueError):
        pass
    return transcript


def _align_with_optional_fallback(
    audio_path: str | Path,
    transcript: str,
    duration: float,
    backend: str,
    allow_fallback: bool,
) -> tuple[list, bool]:
    """Align with the requested backend, optionally degrading to proportional.

    The default (allow_fallback=False) preserves the project's strict
    "no silent fallback" rule: a failing backend raises.  When a caller opts in,
    an unavailable backend degrades to the deterministic proportional timing
    estimator and the caller is told that this happened (degraded=True) so it
    can be reported to clients instead of being hidden.

    Returns:
        (aligned_words, degraded)
    """
    try:
        return align_transcript(audio_path, transcript, duration=duration, backend=backend), False
    except (ImportError, RuntimeError, OSError, ValueError) as exc:
        if not allow_fallback:
            raise
        availability = alignment_backend_availability()
        if not availability.get("proportional", True):
            raise
        logger.warning(
            "Alignment backend %r unavailable (%s); degrading to proportional timing. "
            "Word timings are estimates, not acoustic alignments.",
            backend,
            exc.__class__.__name__,
        )
        return (
            align_transcript(audio_path, transcript, duration=duration, backend="proportional"),
            True,
        )


def evaluate_with_report(
    candidate_path: str | Path,
    baseline_path: str | Path | None = None,
    transcript: str = "",
    audio_id: str = "",
    *,
    alignment_backend: str | None = None,
    allow_alignment_fallback: bool = False,
) -> tuple[EvaluationResult, PipelineReport]:
    """Run the pipeline and additionally return execution metadata.

    See :func:`evaluate` for the scientific contract.  This variant exists so
    the API layer can report which alignment backend actually ran, whether an
    explicit degradation happened, and where time was spent, without polluting
    EvaluationResult with non-scientific fields.

    Args:
        candidate_path:        Path to the candidate speaker's audio file.
        baseline_path:         Path to the baseline (ideal) audio file, or None.
        transcript:            The spoken text (or path to a transcript file).
        audio_id:              Unique identifier string for this evaluation.
        alignment_backend:     Override the configured alignment backend.
        allow_alignment_fallback: Degrade to proportional timing if the chosen
                                backend cannot run, and report it.

    Returns:
        (EvaluationResult, PipelineReport)
    """
    _seed_all()
    stage_seconds: dict[str, float] = {}
    request_backend = alignment_backend or ALIGNMENT_BACKEND
    transcript = _resolve_transcript(transcript)

    # ── 1. Feature extraction ─────────────────────────────────────────────
    t0 = time.perf_counter()
    cand: AudioFeatures = extract_all(candidate_path)
    has_baseline = baseline_path is not None and str(baseline_path).strip() != ""
    if has_baseline:
        base: AudioFeatures = extract_all(baseline_path)
    stage_seconds["features"] = time.perf_counter() - t0

    # ── 2. Align candidate to baseline frame grid ─────────────────────────
    t0 = time.perf_counter()
    if has_baseline:
        _b_pitch, _b_energy, w_pitch, w_energy = align_features(
            baseline_pitch=base.pitch_st,
            baseline_energy=base.energy_z,
            candidate_pitch=cand.pitch_st,
            candidate_energy=cand.energy_z,
        )
    else:
        w_pitch, w_energy = cand.pitch_st, cand.energy_z
    stage_seconds["dtw"] = time.perf_counter() - t0

    # ── 3. Forced Alignment → word timings → pace ────────────────────────
    t0 = time.perf_counter()
    candidate_key = str(Path(candidate_path).resolve())
    aligned_res, degraded = _align_with_optional_fallback(
        candidate_path,
        transcript,
        cand.duration,
        request_backend,
        allow_alignment_fallback,
    )
    words = [Word(text=aw["word"], start=aw["start"], end=aw["end"]) for aw in aligned_res]
    tokens = [w.text for w in words]
    wpm = words_to_pace_wpm(words)
    stage_seconds["alignment_candidate"] = time.perf_counter() - t0

    if has_baseline:
        # Baseline word timings for contrastive scoring and detection
        t0 = time.perf_counter()
        base_aligned, base_degraded = _align_with_optional_fallback(
            baseline_path,
            transcript,
            base.duration,
            request_backend,
            allow_alignment_fallback,
        )
        base_words = [Word(text=aw["word"], start=aw["start"], end=aw["end"]) for aw in base_aligned]
        degraded = degraded or base_degraded
        stage_seconds["alignment_baseline"] = time.perf_counter() - t0
        # Compute word comparisons once — shared by scoring and detection steps
        matched = match_words(base_words, words)
        comparisons = compare_words(
            matched,
            baseline_pitch=base.pitch_st,
            candidate_pitch=cand.pitch_st,
            baseline_energy=base.energy_z,
            candidate_energy=cand.energy_z,
        )

    # ── 4. Rubric scoring ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    if has_baseline:
        # Baseline-relative scoring: every dimension measures deviation from
        # the reference read. Identical inputs give deviation=0, score=1.0.
        # Standalone absolute-range scorers are NOT used in this mode.
        rubric: list[RubricScore] = [
            score_pace_relative(comparisons),
            score_pitch_variation_relative(comparisons, base.pitch_st, cand.pitch_st),
            score_energy_consistency_relative(comparisons),
            score_pause_pattern_relative(
                comparisons, cand.pauses, base.pauses, cand.duration
            ),
        ]
    else:
        # No-baseline mode: absolute heuristic ranges (original behaviour).
        rubric = [
            score_pace(wpm),
            score_pitch_variation(w_pitch),
            score_energy_consistency(w_energy),
            score_pause_pattern(cand.pauses, cand.duration, len(tokens)),
        ]
    composite = compute_composite(rubric)
    stage_seconds["scoring"] = time.perf_counter() - t0

    # ── 5. Flaw detection ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    if has_baseline:
        # When a baseline is provided, only baseline-relative (match-based)
        # detection may emit regions. Standalone absolute-threshold detectors run
        # only when no baseline is given, because an ideal reference speech can
        # itself contain natural stylistic variations (e.g., steady pitch on a clause,
        # natural rhetorical pauses) that absolute heuristic thresholds would falsely flag.
        cand_centroid, cand_flatness = _spectral_bundle(cand.waveform, candidate_key)
        base_centroid, base_flatness = _spectral_bundle(
            base.waveform, str(Path(baseline_path).resolve())
        )
        flaw_regions = detect_from_matches(
            comparisons,
            cand_pauses=cand.pauses,
            base_pauses=base.pauses,
            base_pitch=base.pitch_st,
            cand_pitch=cand.pitch_st,
            cand_energy=cand.energy_z,
            base_centroid=base_centroid,
            cand_centroid=cand_centroid,
            base_flatness=base_flatness,
            cand_flatness=cand_flatness,
        )
    else:
        # Standalone mode: when no reference baseline is given, evaluate against
        # absolute heuristic thresholds across single-signal feature dimensions.
        flaw_regions = detect_all(
            pitch_st=w_pitch,
            energy_z=w_energy,
            pauses=cand.pauses,
            words=words,
            duration=cand.duration,
        )
    stage_seconds["detection"] = time.perf_counter() - t0

    # ── 6. Summary ────────────────────────────────────────────────────────
    summary = _build_summary(composite, rubric, flaw_regions)

    result = EvaluationResult(
        audio_id=audio_id,
        transcript=transcript,
        duration=cand.duration,
        rubric_scores=rubric,
        composite_score=composite,
        flaw_regions=flaw_regions,
        summary=summary,
    )
    actual_backend = "proportional" if degraded else request_backend
    report = PipelineReport(
        alignment_backend=actual_backend,
        alignment_degraded=degraded,
        has_baseline=has_baseline,
        words_total=len(words),
        words_unaligned=sum(1 for aw in aligned_res if not aw["aligned"]),
        stage_seconds={k: round(v, 4) for k, v in stage_seconds.items()},
    )
    return result, report


def evaluate(
    candidate_path: str | Path,
    baseline_path: str | Path | None = None,
    transcript: str = "",
    audio_id: str = "",
    *,
    alignment_backend: str | None = None,
    allow_alignment_fallback: bool = False,
) -> EvaluationResult:
    """Run the full speech evaluation pipeline.

    Steps:
    1. Seed random state for reproducibility.
    2. Extract audio features from candidate and baseline (if provided).
    3. DTW-align candidate feature streams to baseline.
    4. Estimate word timings from transcript + candidate duration.
    5. Score each rubric dimension.
    6. Detect time-stamped flaw regions (contrastive if baseline given, else standalone).
    7. Build summary and assemble EvaluationResult.

    Args:
        candidate_path: Path to the candidate speaker's audio file.
        baseline_path:  Path to the baseline (ideal) audio file, or None.
        transcript:     The spoken text (or path to transcript file).
        audio_id:       Unique identifier string for this evaluation.
        alignment_backend: Optional override for the configured alignment backend.
        allow_alignment_fallback: Opt-in degradation to proportional timing when
                        the requested backend cannot run (reported by
                        :func:`evaluate_with_report`).

    Returns:
        EvaluationResult with rubric scores, flaw regions, and summary.
    """
    result, _report = evaluate_with_report(
        candidate_path,
        baseline_path,
        transcript,
        audio_id,
        alignment_backend=alignment_backend,
        allow_alignment_fallback=allow_alignment_fallback,
    )
    return result
