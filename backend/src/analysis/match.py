"""
src/analysis/match.py
=====================
Baseline-vs-participant word matching and contrastive flaw detection.

Workflow:
1. match_words()  – align baseline and candidate word lists using difflib,
                    returning MatchedWord pairs (one side may be None for
                    insertions / deletions).
2. compare_word() – compute per-word diagnostic scalars (duration ratio,
                    pause delta, energy delta, pitch range ratio).
3. detect_from_matches() – scan MatchedWord list for pace / pause / energy
                    flaws, merging adjacent same-type flags into FlawRegion.

Notes:
- PAUSE_MISSING: emitted when baseline word had a clear inter-word pause
  (> MATCH_PAUSE_BASELINE_MIN) and the candidate has none.
- PAUSE_MISPLACED: removed — without a sentence-boundary oracle, flagging
  misplaced pauses would cause too many false positives and cannot be reliably
  verified. See docs/DECISIONS.md for rationale.
- Pitch comparison uses NaN-aware ratio; if either side has all-NaN pitch in
  the word span the comparison returns None (not flagged as monotone).
"""
from __future__ import annotations

import difflib
import math
from typing import NamedTuple

import numpy as np

from src.config import (
    FILLER_ENERGY_Z_MAX,
    FILLER_MIN_DURATION,
    HOP_LENGTH,
    MATCH_ABSORBED_PAUSE_EXCESS,
    MATCH_CLARITY_CENTROID_RATIO_MIN,
    MATCH_CLARITY_FLATNESS_DELTA_MIN,
    MATCH_ENERGY_LOW_DELTA,
    MATCH_FAST_RATIO_MAX,
    MATCH_FILLER_MIN_EXTRA_GAP_S,
    MATCH_FILLER_MIN_VOICED_FRACTION,
    MATCH_FILLER_PITCH_STD_MAX,
    MATCH_MIN_REGION_WORDS,
    MATCH_PAUSE_BASELINE_MIN,
    MATCH_PAUSE_EXCESS_DELTA,
    MATCH_PAUSE_MISSING_DELTA,
    MATCH_PITCH_MAX_ABS_LOG_DEV,
    MATCH_PITCH_MIN_BASE_STD_ST,
    MATCH_PITCH_MIN_VOICED_FRAMES,
    MATCH_PITCH_STD_RATIO_MIN,
    MATCH_SIMILARITY_THRESHOLD,
    MATCH_SLOW_RATIO_MIN,
    FLAW_MERGE_GAP_SEC,
    MIN_FLAW_DURATION_SEC,
    MONOTONE_MIN_REGION_SEC,
    REGION_MERGE_EPS_S,
    SAMPLE_RATE,
)
from src.features.transcript import Word
from src.schema import FlawRegion, FlawType


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class MatchedWord(NamedTuple):
    """A pair of matched baseline and candidate words.

    Either side may be None:
    - baseline=None : word was inserted in the candidate.
    - candidate=None: word was deleted (skipped) by the candidate.
    """

    baseline: Word | None
    candidate: Word | None


class WordComparison(NamedTuple):
    """Per-matched-word scalar diagnostics.

    ``baseline`` is optional (defaults to None) so that synthetic comparisons
    built by callers/tests without a reference word remain valid.
    """

    candidate: Word
    duration_ratio: float       # candidate_dur / baseline_dur (nan if baseline missing)
    pause_delta_s: float        # candidate_pause_before - baseline_pause_before (nan if no ref)
    energy_delta: float         # candidate_energy_z - baseline_energy_z (nan if no ref)
    pitch_range_ratio: float    # candidate_range / baseline_range (nan if either is nan)
    baseline: Word | None = None  # matched baseline word (None for insertions)


# ---------------------------------------------------------------------------
# Step 1: Word matching
# ---------------------------------------------------------------------------

def match_words(
    baseline_words: list[Word],
    candidate_words: list[Word],
    threshold: float = MATCH_SIMILARITY_THRESHOLD,
) -> list[MatchedWord]:
    """Align baseline and candidate word lists using difflib opcodes.

    Uses difflib.SequenceMatcher on the lowercased word text. Operations:
      - equal / replace → paired MatchedWord(baseline, candidate)
      - insert          → MatchedWord(None, candidate) — extra candidate word
      - delete          → MatchedWord(baseline, None)  — missing candidate word

    For 'replace' blocks the words are zipped; any remainder from the longer
    side is emitted as insertions or deletions.

    Args:
        baseline_words:  Words from the ideal (reference) recording.
        candidate_words: Words from the participant recording.
        threshold:       Minimum SequenceMatcher ratio to treat a replace as a
                         valid pair (else treated as delete + insert).

    Returns:
        Ordered list of MatchedWord pairs.
    """
    b_texts = [w.text.lower() for w in baseline_words]
    c_texts = [w.text.lower() for w in candidate_words]

    matcher = difflib.SequenceMatcher(None, b_texts, c_texts, autojunk=False)
    result: list[MatchedWord] = []

    for tag, b1, b2, c1, c2 in matcher.get_opcodes():
        b_chunk = baseline_words[b1:b2]
        c_chunk = candidate_words[c1:c2]

        if tag == "equal":
            for bw, cw in zip(b_chunk, c_chunk):
                result.append(MatchedWord(bw, cw))

        elif tag == "replace":
            # Pair up as many as possible
            for bw, cw in zip(b_chunk, c_chunk):
                ratio = difflib.SequenceMatcher(
                    None, bw.text.lower(), cw.text.lower()
                ).ratio()
                if ratio >= threshold:
                    result.append(MatchedWord(bw, cw))
                else:
                    result.append(MatchedWord(bw, None))
                    result.append(MatchedWord(None, cw))
            # Leftovers
            for bw in b_chunk[len(c_chunk) :]:
                result.append(MatchedWord(bw, None))
            for cw in c_chunk[len(b_chunk) :]:
                result.append(MatchedWord(None, cw))

        elif tag == "delete":
            for bw in b_chunk:
                result.append(MatchedWord(bw, None))

        elif tag == "insert":
            for cw in c_chunk:
                result.append(MatchedWord(None, cw))

    return result


# ---------------------------------------------------------------------------
# Helpers: word-level feature extraction
# ---------------------------------------------------------------------------

def _word_energy_mean(word: Word, energy_z: np.ndarray) -> float:
    """Mean energy z-score over the word's frame span."""
    f_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    f_end = max(f_start + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
    f_start = max(0, min(f_start, len(energy_z)))
    f_end = max(f_start, min(f_end, len(energy_z)))
    if f_end <= f_start:
        return 0.0
    return float(np.mean(energy_z[f_start:f_end]))


def _word_pitch_range(word: Word, pitch_st: np.ndarray) -> float:
    """Voiced pitch range (max - min) in semitones, NaN if < 2 voiced frames."""
    f_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    f_end = max(f_start + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
    f_start = max(0, min(f_start, len(pitch_st)))
    f_end = max(f_start, min(f_end, len(pitch_st)))
    if f_end <= f_start:
        return float("nan")
    voiced = pitch_st[f_start:f_end]
    voiced = voiced[np.isfinite(voiced)]
    if voiced.size < 2:
        return float("nan")
    return float(np.max(voiced) - np.min(voiced))


def _pause_before(word_idx: int, words: list[Word]) -> float:
    """Gap between the previous word's end and this word's start (clamped to 0)."""
    if word_idx == 0:
        return 0.0
    return max(0.0, words[word_idx].start - words[word_idx - 1].end)


def _word_frames(word: Word, n_frames: int) -> tuple[int, int]:
    """Frame index range [start, end) covered by a word's time span."""
    f_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    f_end = max(f_start + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
    f_start = max(0, min(f_start, n_frames))
    f_end = max(f_start, min(f_end, n_frames))
    return f_start, f_end


def _word_pitch_std(word: Word, pitch_st: np.ndarray) -> tuple[float, int]:
    """Voiced pitch std-dev (semitones) and voiced-frame count for a word span.

    Uses std (not max-min range) because a range collapses to a meaningless
    near-zero denominator on short words, producing unbounded log-ratios.

    Returns:
        (std, n_voiced).  std is NaN when fewer than 2 voiced frames exist.
    """
    f_start, f_end = _word_frames(word, len(pitch_st))
    voiced = pitch_st[f_start:f_end]
    voiced = voiced[np.isfinite(voiced)]
    if voiced.size < 2:
        return float("nan"), int(voiced.size)
    return float(np.std(voiced)), int(voiced.size)


def _word_mean(values: np.ndarray, word: Word) -> float:
    """Mean of a frame-level array over a word's span (NaN if the span is empty)."""
    f_start, f_end = _word_frames(word, len(values))
    segment = values[f_start:f_end]
    segment = segment[np.isfinite(segment)]
    if segment.size == 0:
        return float("nan")
    return float(np.mean(segment))


# ---------------------------------------------------------------------------
# Step 2: Per-word comparison
# ---------------------------------------------------------------------------

def compare_words(
    matched: list[MatchedWord],
    baseline_pitch: np.ndarray,
    candidate_pitch: np.ndarray,
    baseline_energy: np.ndarray,
    candidate_energy: np.ndarray,
) -> list[WordComparison]:
    """Compute diagnostic scalars for each matched word pair.

    Args:
        matched:          Output of match_words().
        baseline_pitch:   Pitch array (semitones) for the baseline.
        candidate_pitch:  Pitch array for the candidate.
        baseline_energy:  Energy z-score array for the baseline.
        candidate_energy: Energy z-score array for the candidate.

    Returns:
        One WordComparison per MatchedWord that has a non-None candidate word.
    """
    results: list[WordComparison] = []

    # Build index-aware lists
    b_words = [m.baseline for m in matched if m.baseline is not None]
    c_words = [m.candidate for m in matched if m.candidate is not None]

    # We iterate matched pairs; for pause_delta we need position in each list
    b_idx = 0
    c_idx = 0

    for m in matched:
        if m.candidate is None:
            if m.baseline is not None:
                b_idx += 1
            continue

        cw = m.candidate
        bw = m.baseline

        # Duration ratio
        c_dur = cw.end - cw.start
        if bw is not None:
            if cw.start == bw.start and cw.end == bw.end:
                dur_ratio = 1.0
            else:
                b_dur = max(bw.end - bw.start, 1e-6)
                dur_ratio = c_dur / b_dur
        else:
            dur_ratio = float("nan")

        # Pause delta (candidate pause_before - baseline pause_before)
        c_pause = _pause_before(c_idx, c_words)
        if bw is not None:
            if c_idx == b_idx and cw.start == bw.start and (c_idx == 0 or c_words[c_idx - 1].end == b_words[b_idx - 1].end):
                pause_delta = 0.0
            else:
                b_pause = _pause_before(b_idx, b_words)
                pause_delta = c_pause - b_pause
        else:
            pause_delta = float("nan")

        # Energy delta
        c_energy = _word_energy_mean(cw, candidate_energy)
        if bw is not None:
            b_energy = _word_energy_mean(bw, baseline_energy)
            if cw.start == bw.start and cw.end == bw.end and np.isclose(c_energy, b_energy, atol=1e-5):
                energy_delta = 0.0
            else:
                energy_delta = c_energy - b_energy
        else:
            energy_delta = float("nan")

        # Pitch range ratio (NaN-aware)
        c_pitch_range = _word_pitch_range(cw, candidate_pitch)
        if bw is not None:
            b_pitch_range = _word_pitch_range(bw, baseline_pitch)
            if math.isnan(c_pitch_range) or math.isnan(b_pitch_range) or b_pitch_range < 1e-6:
                pitch_ratio = float("nan")
            elif cw.start == bw.start and cw.end == bw.end and np.isclose(c_pitch_range, b_pitch_range, atol=1e-5):
                pitch_ratio = 1.0
            else:
                pitch_ratio = c_pitch_range / b_pitch_range
        else:
            pitch_ratio = float("nan")

        results.append(
            WordComparison(
                candidate=cw,
                duration_ratio=dur_ratio,
                pause_delta_s=pause_delta,
                energy_delta=energy_delta,
                pitch_range_ratio=pitch_ratio,
                baseline=bw,
            )
        )

        c_idx += 1
        if bw is not None:
            b_idx += 1

    return results


# ---------------------------------------------------------------------------
# Step 3: Merge adjacent flags into FlawRegions
# ---------------------------------------------------------------------------

def _merge_runs(
    comparisons: list[WordComparison],
    flag_fn,
    flaw_type: FlawType,
    severity_fn,
    metadata_fn,
    min_duration: float = MIN_FLAW_DURATION_SEC,
) -> list[FlawRegion]:
    """Merge adjacent flagged words into FlawRegion objects.

    Args:
        comparisons:  List of WordComparison.
        flag_fn:      (WordComparison) → bool — True if word should be flagged.
        flaw_type:    The FlawType to assign.
        severity_fn:  (list[WordComparison]) → float — compute severity for a run.
        metadata_fn:  (list[WordComparison]) → dict — build metadata for a run.

    Returns:
        List of FlawRegion, one per contiguous run of flagged words.
    """
    from src.explain.templates import explain

    flaws: list[FlawRegion] = []
    run: list[WordComparison] = []

    def _emit(run: list[WordComparison]) -> None:
        if len(run) < MATCH_MIN_REGION_WORDS:
            return
        start_s = run[0].candidate.start
        end_s = run[-1].candidate.end
        if end_s <= start_s:
            return
        sev = severity_fn(run)
        meta = metadata_fn(run)
        flaws.append(
            FlawRegion(
                start=round(start_s, 3),
                end=round(end_s, 3),
                flaw_type=flaw_type,
                severity=round(sev, 4),
                explanation=explain(flaw_type, sev, meta),
                metadata=meta,
            )
        )

    for comp in comparisons:
        if flag_fn(comp):
            # Adjacent flagged words can have an unflagged word between them;
            # time, rather than list position, defines whether they are one run.
            # The epsilon absorbs float noise (26.878 - 26.678 == 0.2000000000000028)
            # that would otherwise split a single flaw into two regions.
            if (
                run
                and comp.candidate.start - run[-1].candidate.end
                > FLAW_MERGE_GAP_SEC + REGION_MERGE_EPS_S
            ):
                _emit(run)
                run = []
            run.append(comp)

    if run:
        _emit(run)

    return [
        flaw for flaw in flaws
        if flaw.end - flaw.start >= min_duration - REGION_MERGE_EPS_S
    ]


# ---------------------------------------------------------------------------
# Public: detect flaws from word comparisons
# ---------------------------------------------------------------------------

def _detect_absorbed_pauses(
    cand_pauses: list[tuple[float, float]],
    base_pauses: list[tuple[float, float]],
) -> list[FlawRegion]:
    """Detect pauses that forced alignment absorbed into word spans.

    When a long silence is inserted between words, the forced aligner stretches
    adjacent word boundaries to cover it.  The gap between words (pause_delta_s)
    then reads near-zero, hiding the flaw.  This function cross-references the
    raw candidate pause list (from dBFS energy) against the baseline pause list.

    Algorithm:
      For each candidate pause, find the nearest baseline pause by matching on
      fractional position (index / total count) so that the comparison is
      stable even after the time-shift caused by the insertion.  If no baseline
      pause maps within the same relative slot, the baseline duration is 0.
      Emit PAUSE_EXCESSIVE anchored at the candidate pause's own timestamps when
          candidate_pause_duration - matched_baseline_duration > MATCH_ABSORBED_PAUSE_EXCESS

    Ideal-vs-ideal safety: Every candidate pause maps to its own baseline
    counterpart with zero extra duration, so the threshold is never crossed.

    Args:
        cand_pauses: list of (start_s, end_s) silence intervals for the candidate.
        base_pauses: list of (start_s, end_s) silence intervals for the baseline.

    Returns:
        List of FlawRegion (PAUSE_EXCESSIVE), sorted by start time.
    """
    from src.analysis.detector import _severity
    from src.explain.templates import explain

    flaws: list[FlawRegion] = []
    n_cand = len(cand_pauses)
    n_base = len(base_pauses)

    for c_i, cp in enumerate(cand_pauses):
        c_dur = cp[1] - cp[0]
        # Map candidate index to closest baseline index by fractional position
        if n_base > 0:
            frac = c_i / max(n_cand - 1, 1)
            b_i = round(frac * (n_base - 1))
            b_i = max(0, min(b_i, n_base - 1))
            b_dur = base_pauses[b_i][1] - base_pauses[b_i][0]
        else:
            b_dur = 0.0

        extra = c_dur - b_dur
        if extra > MATCH_ABSORBED_PAUSE_EXCESS:
            sev = _severity(extra - MATCH_ABSORBED_PAUSE_EXCESS, scale=2.0)
            meta = {
                "pause_duration_s": round(c_dur, 3),
                "baseline_pause_duration_s": round(b_dur, 3),
                "extra_pause_s": round(extra, 3),
            }
            flaws.append(
                FlawRegion(
                    start=round(cp[0], 3),
                    end=round(cp[1], 3),
                    flaw_type=FlawType.PAUSE_EXCESSIVE,
                    severity=round(sev, 4),
                    explanation=explain(FlawType.PAUSE_EXCESSIVE, sev, meta),
                    metadata=meta,
                )
            )

    return sorted(flaws, key=lambda r: r.start)


def _detect_pitch_monotone(
    comparisons: list[WordComparison],
    base_pitch: np.ndarray,
    cand_pitch: np.ndarray,
) -> list[FlawRegion]:
    """Detect contrastively monotone passages (token-level, independent of pace).

    For every matched word pair the voiced pitch dispersion (std-dev in
    semitones) of the candidate is compared with the baseline's.  A word is
    flagged when the candidate is markedly flatter than the reference.  Guards:

    * both sides must carry >= MATCH_PITCH_MIN_VOICED_FRAMES voiced frames;
    * the baseline word must itself be expressive enough
      (>= MATCH_PITCH_MIN_BASE_STD_ST) for the ratio to be meaningful;
    * a matched baseline word must exist (insertions are skipped).

    Ideal-vs-ideal produces identical dispersions on both sides, so the ratio
    is exactly 1.0 and no region is emitted.

    Args:
        comparisons: Per-word diagnostics (must carry baseline word spans).
        base_pitch:  Baseline pitch array (semitones, NaN = unvoiced).
        cand_pitch:  Candidate pitch array.

    Returns:
        List of PITCH_MONOTONE FlawRegion (merged runs).
    """
    from src.analysis.detector import _severity

    def _ratio(comp: WordComparison) -> float:
        if comp.baseline is None:
            return float("nan")
        base_std, base_n = _word_pitch_std(comp.baseline, base_pitch)
        cand_std, cand_n = _word_pitch_std(comp.candidate, cand_pitch)
        if (
            base_n < MATCH_PITCH_MIN_VOICED_FRAMES
            or cand_n < MATCH_PITCH_MIN_VOICED_FRAMES
            or not np.isfinite(base_std)
            or base_std < MATCH_PITCH_MIN_BASE_STD_ST
        ):
            return float("nan")
        return cand_std / base_std

    def _flag(comp: WordComparison) -> bool:
        ratio = _ratio(comp)
        return np.isfinite(ratio) and ratio < MATCH_PITCH_STD_RATIO_MIN

    def _severity_for(run: list[WordComparison]) -> float:
        ratios = [r for r in (_ratio(c) for c in run) if np.isfinite(r)]
        if not ratios:
            return 0.0
        mean_ratio = float(np.mean(ratios))
        # deviation: how far below the ceiling the dispersion fell (relative)
        deviation = max(0.0, MATCH_PITCH_STD_RATIO_MIN / max(mean_ratio, 1e-6) - 1.0)
        return _severity(deviation, scale=1.2)

    def _metadata_for(run: list[WordComparison]) -> dict:
        ratios = [r for r in (_ratio(c) for c in run) if np.isfinite(r)]
        mean_ratio = float(np.mean(ratios)) if ratios else float("nan")
        cand_stds = [
            _word_pitch_std(c.candidate, cand_pitch)[0] for c in run
        ]
        cand_stds = [s for s in cand_stds if np.isfinite(s)]
        mean_cand_std = float(np.mean(cand_stds)) if cand_stds else float("nan")
        return {
            "mean_pitch_std_ratio": round(mean_ratio, 3) if np.isfinite(mean_ratio) else None,
            "pitch_std_st": round(mean_cand_std, 3) if np.isfinite(mean_cand_std) else None,
            "words": len(run),
        }

    return _merge_runs(
        comparisons,
        flag_fn=_flag,
        flaw_type=FlawType.PITCH_MONOTONE,
        severity_fn=_severity_for,
        metadata_fn=_metadata_for,
        min_duration=max(MIN_FLAW_DURATION_SEC, MONOTONE_MIN_REGION_SEC),
    )


def _detect_filler_gaps(
    comparisons: list[WordComparison],
    cand_pitch: np.ndarray,
    cand_energy: np.ndarray,
) -> list[FlawRegion]:
    """Detect probable filler sounds (uh/um) in candidate inter-word gaps.

    A gap between two consecutive matched candidate words is reported as FILLER
    when ALL of the following hold:

    1. the gap lasts >= FILLER_MIN_DURATION (short co-articulation is ignored);
    2. it is longer than its baseline counterpart by at least
       MATCH_FILLER_MIN_EXTRA_GAP_S (contrastive guard: the reference read has
       no equivalent break here);
    3. at least MATCH_FILLER_MIN_VOICED_FRACTION of its frames carry F0 -- this
       is what separates a filler (voiced, steady) from an inserted pause
       (unvoiced silence, where pYIN finds no or only sporadic F0);
    4. mean energy z-score stays below FILLER_ENERGY_Z_MAX (fillers are quiet);
    5. voiced pitch std stays below MATCH_FILLER_PITCH_STD_MAX (flat delivery).

    The detector is acoustic only: it does not transcribe the filler content.

    Args:
        comparisons: Per-word diagnostics in candidate order.
        cand_pitch:  Candidate pitch array (semitones, NaN = unvoiced).
        cand_energy: Candidate energy z-score array.

    Returns:
        List of FILLER FlawRegion, sorted by start time.
    """
    from src.explain.templates import explain

    flaws: list[FlawRegion] = []

    for i in range(1, len(comparisons)):
        prev_word = comparisons[i - 1].candidate
        word = comparisons[i].candidate
        gap_start = prev_word.end
        gap_end = word.start
        gap_dur = gap_end - gap_start
        if gap_dur < FILLER_MIN_DURATION:
            continue

        extra_gap = comparisons[i].pause_delta_s
        if not np.isfinite(extra_gap) or extra_gap < MATCH_FILLER_MIN_EXTRA_GAP_S:
            continue

        f_start, f_end = _word_frames(
            Word(text="", start=gap_start, end=gap_end), len(cand_pitch)
        )
        if f_end <= f_start:
            continue
        gap_pitch = cand_pitch[f_start:f_end]
        gap_energy = cand_energy[f_start:f_end]

        voiced = gap_pitch[np.isfinite(gap_pitch)]
        voiced_fraction = float(voiced.size) / max(int(gap_pitch.size), 1)
        if voiced_fraction < MATCH_FILLER_MIN_VOICED_FRACTION:
            continue

        mean_energy = float(np.mean(gap_energy)) if gap_energy.size else 0.0
        if mean_energy >= FILLER_ENERGY_Z_MAX:
            continue

        pitch_std = float(np.std(voiced)) if voiced.size >= 2 else float("nan")
        if not np.isfinite(pitch_std) or pitch_std >= MATCH_FILLER_PITCH_STD_MAX:
            continue

        sev = float(np.clip(np.tanh(gap_dur - FILLER_MIN_DURATION), 0.0, 1.0))
        meta = {
            "gap_duration_s": round(gap_dur, 3),
            "extra_gap_s": round(float(extra_gap), 3),
            "mean_energy_z": round(mean_energy, 3),
            "pitch_std_st": round(pitch_std, 3),
            "voiced_frame_fraction": round(voiced_fraction, 3),
        }
        flaws.append(
            FlawRegion(
                start=round(gap_start, 3),
                end=round(gap_end, 3),
                flaw_type=FlawType.FILLER,
                severity=round(sev, 4),
                explanation=explain(FlawType.FILLER, sev, meta),
                metadata=meta,
            )
        )

    return sorted(flaws, key=lambda r: r.start)


def _detect_unclear_words(
    comparisons: list[WordComparison],
    cand_centroid: np.ndarray,
    base_centroid: np.ndarray,
    cand_flatness: np.ndarray,
    base_flatness: np.ndarray,
) -> list[FlawRegion]:
    """Detect contrastively unclear (spectrally degraded) words.

    A word is flagged when BOTH of these hold relative to its baseline span:

    * the spectral centroid rises by >= MATCH_CLARITY_CENTROID_RATIO_MIN
      (broadband noise / articulation noise raises the spectral centre of mass);
    * spectral flatness increases by >= MATCH_CLARITY_FLATNESS_DELTA_MIN
      (noise-like frames have Wiener entropy closer to 1).

    Requiring both conditions rejects silence and level changes, which move the
    centroid but not flatness in the same direction.  This is a relative
    spectral-difference detector: it can flag a genuine clarity drop only when a
    reference read of the same words exists, and it is not a speech-recognition
    or intelligibility model.

    Args:
        comparisons:   Per-word diagnostics.
        cand_centroid: Candidate spectral centroid per frame (Hz).
        base_centroid: Baseline spectral centroid per frame (Hz).
        cand_flatness: Candidate spectral flatness per frame.
        base_flatness: Baseline spectral flatness per frame.

    Returns:
        List of UNCLEAR FlawRegion (merged runs).
    """
    from src.analysis.detector import _severity

    def _metrics(comp: WordComparison) -> tuple[float, float] | None:
        if comp.baseline is None:
            return None
        c_cent = _word_mean(cand_centroid, comp.candidate)
        b_cent = _word_mean(base_centroid, comp.baseline)
        if not np.isfinite(c_cent) or not np.isfinite(b_cent) or b_cent <= 1e-6:
            return None
        ratio = c_cent / b_cent
        c_flat = _word_mean(cand_flatness, comp.candidate)
        b_flat = _word_mean(base_flatness, comp.baseline)
        if not np.isfinite(c_flat) or not np.isfinite(b_flat):
            return None
        return ratio, c_flat - b_flat

    def _flag(comp: WordComparison) -> bool:
        metrics = _metrics(comp)
        if metrics is None:
            return False
        ratio, flat_delta = metrics
        return (
            ratio >= MATCH_CLARITY_CENTROID_RATIO_MIN
            and flat_delta >= MATCH_CLARITY_FLATNESS_DELTA_MIN
        )

    def _severity_for(run: list[WordComparison]) -> float:
        metrics = [m for m in (_metrics(c) for c in run) if m is not None]
        if not metrics:
            return 0.0
        mean_ratio = float(np.mean([m[0] for m in metrics]))
        mean_flat = float(np.mean([m[1] for m in metrics]))
        deviation = (mean_ratio - MATCH_CLARITY_CENTROID_RATIO_MIN) + (
            mean_flat - MATCH_CLARITY_FLATNESS_DELTA_MIN
        )
        return _severity(max(0.0, deviation), scale=1.0)

    def _metadata_for(run: list[WordComparison]) -> dict:
        metrics = [m for m in (_metrics(c) for c in run) if m is not None]
        mean_ratio = float(np.mean([m[0] for m in metrics])) if metrics else float("nan")
        mean_flat = float(np.mean([m[1] for m in metrics])) if metrics else float("nan")
        return {
            "mean_spectral_centroid_ratio": (
                round(mean_ratio, 3) if np.isfinite(mean_ratio) else None
            ),
            "mean_flatness_delta": (
                round(mean_flat, 4) if np.isfinite(mean_flat) else None
            ),
            "words": len(run),
        }

    return _merge_runs(
        comparisons,
        flag_fn=_flag,
        flaw_type=FlawType.UNCLEAR,
        severity_fn=_severity_for,
        metadata_fn=_metadata_for,
    )


def detect_from_matches(
    comparisons: list[WordComparison],
    cand_pauses: list[tuple[float, float]] | None = None,
    base_pauses: list[tuple[float, float]] | None = None,
    base_pitch: np.ndarray | None = None,
    cand_pitch: np.ndarray | None = None,
    cand_energy: np.ndarray | None = None,
    base_centroid: np.ndarray | None = None,
    cand_centroid: np.ndarray | None = None,
    base_flatness: np.ndarray | None = None,
    cand_flatness: np.ndarray | None = None,
) -> list[FlawRegion]:
    """Detect pace, pause, and energy flaws from word-level comparisons.

    Optional keyword arrays enable the contrastive detectors that need raw
    frame-level evidence (pitch dispersion, gap voicing, spectral clarity).
    When they are omitted those detectors stay silent, so existing callers keep
    their exact previous behaviour.

    Args:
        comparisons:  Per-word diagnostic scalars from compare_words().
        cand_pauses:  Raw candidate pause list (start_s, end_s) from AudioFeatures.
                      When provided together with base_pauses, absorbed-pause
                      detection runs to catch silences the aligner hid inside word spans.
        base_pauses:  Raw baseline pause list.  Must be provided with cand_pauses.
        base_pitch:   Baseline frame-level pitch array (semitones, NaN = unvoiced).
        cand_pitch:   Candidate frame-level pitch array.
        cand_energy:  Candidate frame-level energy z-score array.
        base_centroid: Baseline per-frame spectral centroid (Hz).
        cand_centroid: Candidate per-frame spectral centroid (Hz).
        base_flatness: Baseline per-frame spectral flatness.
        cand_flatness: Candidate per-frame spectral flatness.

    Returns:
        Time-sorted list of FlawRegion.
    """
    from src.analysis.detector import _severity

    flaws: list[FlawRegion] = []

    # --- PACE_TOO_SLOW ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.duration_ratio)
                              and c.duration_ratio > MATCH_SLOW_RATIO_MIN,
            flaw_type=FlawType.PACE_TOO_SLOW,
            severity_fn=lambda run: _severity(
                float(np.mean([c.duration_ratio for c in run])) - MATCH_SLOW_RATIO_MIN,
                scale=2.0,
            ),
            metadata_fn=lambda run: {
                "mean_duration_ratio": round(
                    float(np.mean([c.duration_ratio for c in run])), 3
                )
            },
        )
    )

    # --- PACE_TOO_FAST ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.duration_ratio)
                              and c.duration_ratio < MATCH_FAST_RATIO_MAX,
            flaw_type=FlawType.PACE_TOO_FAST,
            severity_fn=lambda run: _severity(
                MATCH_FAST_RATIO_MAX - float(np.mean([c.duration_ratio for c in run])),
                scale=2.0,
            ),
            metadata_fn=lambda run: {
                "mean_duration_ratio": round(
                    float(np.mean([c.duration_ratio for c in run])), 3
                )
            },
        )
    )

    # --- PAUSE_EXCESSIVE ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.pause_delta_s)
                              and c.pause_delta_s > MATCH_PAUSE_EXCESS_DELTA,
            flaw_type=FlawType.PAUSE_EXCESSIVE,
            severity_fn=lambda run: _severity(
                float(np.mean([c.pause_delta_s for c in run])) - MATCH_PAUSE_EXCESS_DELTA,
                scale=1.5,
            ),
            metadata_fn=lambda run: {
                "mean_pause_delta_s": round(
                    float(np.mean([c.pause_delta_s for c in run])), 3
                )
            },
        )
    )

    # --- PAUSE_MISSING ---
    # Only flag when baseline had a substantial pause that the candidate omitted.
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.pause_delta_s)
                              and c.pause_delta_s < MATCH_PAUSE_MISSING_DELTA,
            flaw_type=FlawType.PAUSE_MISSING,
            severity_fn=lambda run: _severity(
                abs(float(np.mean([c.pause_delta_s for c in run])))
                - abs(MATCH_PAUSE_MISSING_DELTA),
                scale=1.5,
            ),
            metadata_fn=lambda run: {
                "mean_pause_delta_s": round(
                    float(np.mean([c.pause_delta_s for c in run])), 3
                )
            },
        )
    )

    # --- ENERGY_LOW ---
    flaws.extend(
        _merge_runs(
            comparisons,
            flag_fn=lambda c: not math.isnan(c.energy_delta)
                              and c.energy_delta < MATCH_ENERGY_LOW_DELTA,
            flaw_type=FlawType.ENERGY_LOW,
            severity_fn=lambda run: _severity(
                abs(float(np.mean([c.energy_delta for c in run])))
                - abs(MATCH_ENERGY_LOW_DELTA),
                scale=0.8,
            ),
            metadata_fn=lambda run: {
                "mean_energy_delta": round(
                    float(np.mean([c.energy_delta for c in run])), 3
                )
            },
        )
    )

    # --- PITCH_MONOTONE (contrastive voiced dispersion) ---
    # PITCH_ERRATIC stays standalone-only: the contrastive benchmark contains no
    # erratic ground truth, and a symmetric |log ratio| rule fires on ordinary
    # expressive variation in the reference read.
    if base_pitch is not None and cand_pitch is not None:
        flaws.extend(_detect_pitch_monotone(comparisons, base_pitch, cand_pitch))

    # --- FILLER (acoustic gap signature) ---
    if cand_pitch is not None and cand_energy is not None:
        flaws.extend(_detect_filler_gaps(comparisons, cand_pitch, cand_energy))

    # --- UNCLEAR (contrastive spectral degradation) ---
    if (
        base_centroid is not None
        and cand_centroid is not None
        and base_flatness is not None
        and cand_flatness is not None
    ):
        flaws.extend(
            _detect_unclear_words(
                comparisons, cand_centroid, base_centroid, cand_flatness, base_flatness
            )
        )

    # --- PAUSE_EXCESSIVE (absorbed into word spans) ---
    # When the forced aligner stretches a word's timestamps to cover a long
    # inserted silence, pause_delta_s reads near-zero (no inter-word gap).
    # Cross-referencing the raw dBFS-detected pause lists catches these cases.
    if cand_pauses is not None and base_pauses is not None:
        absorbed = _detect_absorbed_pauses(cand_pauses, base_pauses)
        # Avoid double-counting: only add absorbed pauses whose time ranges are
        # not already covered by a word-boundary PAUSE_EXCESSIVE region.
        existing_ranges = [
            (f.start, f.end) for f in flaws if f.flaw_type == FlawType.PAUSE_EXCESSIVE
        ]
        for ap in absorbed:
            if not any(ap.start < e and ap.end > s for s, e in existing_ranges):
                flaws.append(ap)

    flaws.sort(key=lambda r: r.start)
    return flaws

