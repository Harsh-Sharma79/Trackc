"""Regression coverage for flaw-region consolidation and DTW vectorization."""
from __future__ import annotations

import math

import numpy as np

from src.analysis.detector import detect_pitch_flaws
from src.analysis.match import WordComparison, detect_from_matches
from src.features.alignment import _sakoe_chiba_dtw
from src.features.transcript import Word
from src.schema import FlawType


def _comparison(start: float, end: float, ratio: float) -> WordComparison:
    return WordComparison(Word("word", start, end), ratio, 0.0, 0.0, math.nan)


def test_identical_comparisons_emit_no_regions():
    comparisons = [_comparison(0.0, 0.4, 1.0), _comparison(0.5, 0.9, 1.0)]
    assert detect_from_matches(comparisons) == []


def test_adjacent_pace_detections_merge_into_one_region():
    comparisons = [_comparison(0.0, 0.3, 0.5), _comparison(0.4, 0.8, 0.5)]
    flaws = detect_from_matches(comparisons)
    fast = [f for f in flaws if f.flaw_type == FlawType.PACE_TOO_FAST]
    assert len(fast) == 1
    assert (fast[0].start, fast[0].end) == (0.0, 0.8)


def test_short_word_level_region_is_dropped():
    flaws = detect_from_matches([_comparison(0.0, 0.1, 0.5)])
    assert flaws == []


def test_unvoiced_frames_do_not_form_monotone_region():
    pitch = np.full(200, np.nan)
    pitch[80:90] = 2.0  # voiced, but too short to make a sustained voiced run
    assert not any(f.flaw_type == FlawType.PITCH_MONOTONE for f in detect_pitch_flaws(pitch))


def _original_loop_dtw(cost: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference implementation retained here to compare the vectorized update."""
    n, m = cost.shape
    band = max(radius, abs(n - m) + 5)
    acc = np.full((n, m), np.inf, dtype=np.float64)
    acc[0, 0] = cost[0, 0]
    for i in range(1, n):
        for j in range(max(0, i - band), min(m - 1, i + band) + 1):
            acc[i, j] = cost[i, j] + min(
                acc[i - 1, j - 1] if j > 0 else np.inf,
                acc[i - 1, j],
                acc[i, j - 1] if j > 0 else np.inf,
            )
    ref, cand = [n - 1], [m - 1]
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            if best == acc[i - 1, j - 1]:
                i, j = i - 1, j - 1
            elif best == acc[i - 1, j]:
                i -= 1
            else:
                j -= 1
        ref.append(i)
        cand.append(j)
    return np.array(ref[::-1]), np.array(cand[::-1])


def test_vectorized_dtw_matches_original_loop_on_small_input():
    cost = np.array([[0.0, 3.0, 8.0], [2.0, 0.0, 3.0], [7.0, 2.0, 0.0], [9.0, 4.0, 1.0]])
    expected = _original_loop_dtw(cost, radius=1)
    actual = _sakoe_chiba_dtw(cost, radius=1)
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
