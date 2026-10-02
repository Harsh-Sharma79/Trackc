"""
tests/test_analysis_match.py
============================
Tests for src/analysis/match.py.

All tests use synthetic pitch/energy arrays and hand-crafted Word timings.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import (
    HOP_LENGTH,
    MATCH_SLOW_RATIO_MIN,
    SAMPLE_RATE,
    SEED,
)
from src.analysis.match import (
    MatchedWord,
    WordComparison,
    compare_words,
    detect_from_matches,
    match_words,
)
from src.features.transcript import Word
from src.schema import FlawType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_words(texts_and_times: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(t, s, e) for t, s, e in texts_and_times]


def _flat_arrays(n_frames: int, pitch_val: float = 0.0, energy_val: float = 0.0):
    pitch = np.full(n_frames, pitch_val, dtype=np.float64)
    energy = np.full(n_frames, energy_val, dtype=np.float64)
    return pitch, energy


# ---------------------------------------------------------------------------
# match_words
# ---------------------------------------------------------------------------

def test_match_words_identical():
    """Identical word lists → all MatchedWord have both sides non-None."""
    words = _make_words([("hello", 0.0, 0.3), ("world", 0.4, 0.7)])
    matched = match_words(words, words)
    assert all(m.baseline is not None and m.candidate is not None for m in matched)


def test_match_words_length_identical():
    """Same-text lists → same count of pairs as words."""
    words = _make_words([("a", 0.0, 0.1), ("b", 0.2, 0.3), ("c", 0.4, 0.5)])
    matched = match_words(words, words)
    assert len(matched) == 3


def test_match_words_insert_flagged():
    """Extra candidate word → at least one MatchedWord with baseline=None."""
    baseline = _make_words([("hello", 0.0, 0.3)])
    candidate = _make_words([("hello", 0.0, 0.3), ("extra", 0.4, 0.5)])
    matched = match_words(baseline, candidate)
    insertions = [m for m in matched if m.baseline is None]
    assert len(insertions) >= 1


def test_match_words_delete_flagged():
    """Missing candidate word → at least one MatchedWord with candidate=None."""
    baseline = _make_words([("hello", 0.0, 0.3), ("missing", 0.4, 0.6)])
    candidate = _make_words([("hello", 0.0, 0.3)])
    matched = match_words(baseline, candidate)
    deletions = [m for m in matched if m.candidate is None]
    assert len(deletions) >= 1


def test_match_words_order_preserved():
    """Matched pairs should appear in the candidate's chronological order."""
    words = _make_words([("a", 0.0, 0.1), ("b", 0.2, 0.3), ("c", 0.4, 0.5)])
    matched = match_words(words, words)
    candidate_starts = [m.candidate.start for m in matched if m.candidate]
    assert candidate_starts == sorted(candidate_starts)


# ---------------------------------------------------------------------------
# compare_words — duration ratio
# ---------------------------------------------------------------------------

def test_compare_duration_ratio_identical():
    """Same timing → duration_ratio ≈ 1.0 for all words."""
    words = _make_words([("a", 0.0, 0.5), ("b", 0.6, 1.0)])
    matched = match_words(words, words)
    N = int(SAMPLE_RATE * 1.5 / HOP_LENGTH) + 5
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    for c in comps:
        assert abs(c.duration_ratio - 1.0) < 1e-6


def test_compare_duration_ratio_slow():
    """Candidate word twice as long → duration_ratio ≈ 2.0."""
    baseline = _make_words([("hello", 0.0, 0.5)])
    candidate = _make_words([("hello", 0.0, 1.0)])  # 2× longer
    matched = match_words(baseline, candidate)
    N = int(SAMPLE_RATE * 2.0 / HOP_LENGTH) + 5
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    assert len(comps) == 1
    assert abs(comps[0].duration_ratio - 2.0) < 1e-4


# ---------------------------------------------------------------------------
# compare_words — NaN handling
# ---------------------------------------------------------------------------

def test_compare_pitch_ratio_nan_for_all_unvoiced():
    """If both sides have all-NaN pitch, pitch_range_ratio must be NaN."""
    words = _make_words([("a", 0.0, 0.5)])
    matched = match_words(words, words)
    N = 30
    p = np.full(N, float("nan"), dtype=np.float64)
    e = np.zeros(N, dtype=np.float64)
    comps = compare_words(matched, p, p, e, e)
    assert len(comps) == 1
    assert math.isnan(comps[0].pitch_range_ratio)


def test_compare_pitch_ratio_not_flagged_on_nan():
    """NaN pitch_range_ratio must NOT trigger pitch detection (avoids false positives)."""
    # If pitch is all NaN, detect_from_matches should not emit pitch flaws
    words = _make_words([("a", 0.0, 0.5), ("b", 0.6, 1.0)])
    matched = match_words(words, words)
    N = 40
    p = np.full(N, float("nan"), dtype=np.float64)
    e = np.zeros(N, dtype=np.float64)
    comps = compare_words(matched, p, p, e, e)
    flaws = detect_from_matches(comps)
    # No pace/pause/energy flaws expected for identical timing
    pitch_flaws = [f for f in flaws if f.flaw_type in (
        FlawType.PITCH_MONOTONE, FlawType.PITCH_ERRATIC
    )]
    assert len(pitch_flaws) == 0


# ---------------------------------------------------------------------------
# detect_from_matches — zero flaws for identical audio
# ---------------------------------------------------------------------------

def test_identical_words_gives_no_pace_flaws():
    """Identical baseline and candidate timings must produce zero pace flaws."""
    words = _make_words([
        ("the", 0.0, 0.2), ("quick", 0.2, 0.45), ("brown", 0.5, 0.75),
        ("fox", 0.8, 1.0), ("jumps", 1.1, 1.35),
    ])
    matched = match_words(words, words)
    N = int(SAMPLE_RATE * 2.0 / HOP_LENGTH) + 5
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    flaws = detect_from_matches(comps)
    pace_flaws = [f for f in flaws if f.flaw_type in (
        FlawType.PACE_TOO_SLOW, FlawType.PACE_TOO_FAST
    )]
    assert len(pace_flaws) == 0


def test_identical_words_gives_no_pause_flaws():
    """Identical timings → pause_delta == 0 → no pause flaws."""
    words = _make_words([("a", 0.0, 0.3), ("b", 0.4, 0.7), ("c", 0.8, 1.0)])
    matched = match_words(words, words)
    N = 50
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    flaws = detect_from_matches(comps)
    pause_flaws = [f for f in flaws if f.flaw_type in (
        FlawType.PAUSE_EXCESSIVE, FlawType.PAUSE_MISSING
    )]
    assert len(pause_flaws) == 0


# ---------------------------------------------------------------------------
# detect_from_matches — slowed segment gives pace flaw
# ---------------------------------------------------------------------------

def test_slowed_words_give_slow_flaw():
    """Words stretched 1.5× (> MATCH_SLOW_RATIO_MIN=1.30) must emit PACE_TOO_SLOW."""
    baseline = _make_words([("a", 0.0, 0.3), ("b", 0.3, 0.6)])
    # Candidate: same words but 1.5× longer
    candidate = _make_words([("a", 0.0, 0.45), ("b", 0.45, 0.9)])
    matched = match_words(baseline, candidate)
    N = int(SAMPLE_RATE * 1.5 / HOP_LENGTH) + 5
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    flaws = detect_from_matches(comps)
    slow_flaws = [f for f in flaws if f.flaw_type == FlawType.PACE_TOO_SLOW]
    assert len(slow_flaws) >= 1


def test_slowed_flaw_covers_region():
    """The PACE_TOO_SLOW region must cover the slow words' time span."""
    baseline = _make_words([("a", 0.0, 0.3), ("b", 0.3, 0.6)])
    candidate = _make_words([("a", 0.0, 0.45), ("b", 0.45, 0.9)])
    matched = match_words(baseline, candidate)
    N = int(SAMPLE_RATE * 1.5 / HOP_LENGTH) + 5
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    flaws = detect_from_matches(comps)
    slow_flaws = [f for f in flaws if f.flaw_type == FlawType.PACE_TOO_SLOW]
    assert len(slow_flaws) >= 1
    # Region must start at or before candidate start and end at or after candidate end
    assert slow_flaws[0].start <= 0.0 + 1e-3
    assert slow_flaws[0].end >= 0.9 - 1e-3


def test_flaw_severity_in_range():
    """All emitted FlawRegion severities must be in [0, 1]."""
    baseline = _make_words([("a", 0.0, 0.3), ("b", 0.3, 0.6)])
    candidate = _make_words([("a", 0.0, 0.6), ("b", 0.6, 1.2)])
    matched = match_words(baseline, candidate)
    N = int(SAMPLE_RATE * 2.0 / HOP_LENGTH) + 5
    p, e = _flat_arrays(N)
    comps = compare_words(matched, p, p, e, e)
    flaws = detect_from_matches(comps)
    for flaw in flaws:
        assert 0.0 <= flaw.severity <= 1.0
