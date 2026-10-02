"""
tests/test_analysis_scoring.py
===============================
Tests for src/analysis/scoring.py.
"""
import numpy as np
import pytest

from src.analysis.scoring import (
    _linear_penalty,
    compute_composite,
    exponential_score,
    score_energy_consistency,
    score_pace,
    score_pause_pattern,
    score_pitch_variation,
)
from src.schema import RubricDimension, RubricScore


class TestExponentialScore:
    def test_zero_deviation_scores_one(self):
        assert exponential_score(0.0, k=1.5) == pytest.approx(1.0)

    def test_positive_deviation_decays(self):
        assert exponential_score(1.0, k=1.0) == pytest.approx(float(np.exp(-1.0)), abs=1e-4)

    def test_higher_k_decays_faster(self):
        score_low_k = exponential_score(1.0, k=0.5)
        score_high_k = exponential_score(1.0, k=2.0)
        assert score_high_k < score_low_k

    def test_negative_deviation_clamped_to_zero_deviation(self):
        # Negative deviation should be clamped to 0 → score 1.0
        assert exponential_score(-0.5, k=1.0) == pytest.approx(1.0)

    def test_score_always_in_zero_one(self):
        for dev in [0.0, 0.5, 1.0, 5.0, 100.0]:
            s = exponential_score(dev, k=1.5)
            assert 0.0 <= s <= 1.0



class TestLinearPenalty:
    def test_inside_range_returns_one(self):
        assert _linear_penalty(5.0, 3.0, 7.0) == pytest.approx(1.0)

    def test_at_boundary_returns_one(self):
        assert _linear_penalty(3.0, 3.0, 7.0) == pytest.approx(1.0)
        assert _linear_penalty(7.0, 3.0, 7.0) == pytest.approx(1.0)

    def test_far_above_approaches_zero(self):
        # value = hi + span = hi + (hi - lo) → score 0
        lo, hi = 120.0, 160.0
        span = hi - lo
        assert _linear_penalty(hi + span, lo, hi) == pytest.approx(0.0)

    def test_far_below_approaches_zero(self):
        lo, hi = 120.0, 160.0
        span = hi - lo
        assert _linear_penalty(lo - span, lo, hi) == pytest.approx(0.0)

    def test_output_clipped_to_zero_one(self):
        score = _linear_penalty(9999.0, 1.0, 2.0)
        assert 0.0 <= score <= 1.0


class TestScorePace:
    def test_ideal_pace_scores_one(self):
        from src.config import PACE_IDEAL_WPM_MIN, PACE_IDEAL_WPM_MAX
        wpm = (PACE_IDEAL_WPM_MIN + PACE_IDEAL_WPM_MAX) / 2
        rs = score_pace(wpm)
        assert rs.score == pytest.approx(1.0)
        assert rs.dimension == RubricDimension.PACE

    def test_too_fast_penalised(self):
        rs = score_pace(300.0)
        assert rs.score < 1.0

    def test_too_slow_penalised(self):
        rs = score_pace(50.0)
        assert rs.score < 1.0

    def test_score_in_bounds(self):
        for wpm in [0, 80, 140, 200, 300]:
            rs = score_pace(wpm)
            assert 0.0 <= rs.score <= 1.0


class TestScorePitchVariation:
    def test_ideal_std_scores_one(self):
        from src.config import PITCH_VAR_IDEAL_MIN, PITCH_VAR_IDEAL_MAX
        ideal_std = (PITCH_VAR_IDEAL_MIN + PITCH_VAR_IDEAL_MAX) / 2
        # Build array with desired std (excluding zeros = unvoiced frames)
        pitch = np.random.normal(0, ideal_std, 200)
        rs = score_pitch_variation(pitch)
        assert rs.score > 0.5   # should score well

    def test_monotone_penalised(self):
        # Constant pitch → zero std
        pitch = np.ones(100)
        rs = score_pitch_variation(pitch)
        assert rs.score < 1.0

    def test_all_unvoiced_handled(self):
        pitch = np.full(100, np.nan)   # all unvoiced (NaN)
        rs = score_pitch_variation(pitch)
        assert 0.0 <= rs.score <= 1.0

    def test_dimension_correct(self):
        rs = score_pitch_variation(np.zeros(10))
        assert rs.dimension == RubricDimension.PITCH_VARIATION


class TestScoreEnergyConsistency:
    def test_consistent_energy_scores_well(self):
        energy = np.random.normal(0, 1.0, 200)  # std ≈ 1.0, ideal
        rs = score_energy_consistency(energy)
        assert rs.score > 0.5

    def test_empty_array_handled(self):
        rs = score_energy_consistency(np.array([0.5]))
        assert 0.0 <= rs.score <= 1.0

    def test_dimension_correct(self):
        rs = score_energy_consistency(np.zeros(10))
        assert rs.dimension == RubricDimension.ENERGY_CONSISTENCY


class TestScorePausePattern:
    def test_no_pauses_short_speech_scores_one(self):
        rs = score_pause_pattern([], duration=30.0, word_count=80)
        assert rs.score == pytest.approx(1.0)

    def test_appropriate_pauses_score_one(self):
        # All pauses within ideal max
        from src.config import PAUSE_IDEAL_MAX
        pauses = [(5.0, 5.0 + PAUSE_IDEAL_MAX * 0.5)]
        rs = score_pause_pattern(pauses, duration=60.0, word_count=100)
        assert rs.score == pytest.approx(1.0)

    def test_long_pause_penalised(self):
        from src.config import PAUSE_IDEAL_MAX
        pauses = [(5.0, 5.0 + PAUSE_IDEAL_MAX * 3)]   # clearly excessive
        rs = score_pause_pattern(pauses, duration=60.0, word_count=100)
        assert rs.score < 1.0

    def test_missing_pause_in_long_speech_penalised(self):
        rs = score_pause_pattern([], duration=90.0, word_count=200)
        assert rs.score < 1.0

    def test_score_in_bounds(self):
        pauses = [(i * 10.0, i * 10.0 + 2.0) for i in range(5)]
        rs = score_pause_pattern(pauses, duration=120.0, word_count=300)
        assert 0.0 <= rs.score <= 1.0

    def test_dimension_correct(self):
        rs = score_pause_pattern([], duration=10.0, word_count=20)
        assert rs.dimension == RubricDimension.PAUSE_PATTERN


class TestComputeComposite:
    def test_equal_weights(self):
        scores = [
            RubricScore(dimension=RubricDimension.PACE, score=0.8),
            RubricScore(dimension=RubricDimension.PITCH_VARIATION, score=0.6),
        ]
        comp = compute_composite(scores)
        assert comp == pytest.approx(0.7, abs=1e-4)

    def test_custom_weights(self):
        scores = [
            RubricScore(dimension=RubricDimension.PACE, score=1.0, weight=3.0),
            RubricScore(dimension=RubricDimension.PITCH_VARIATION, score=0.0, weight=1.0),
        ]
        comp = compute_composite(scores)
        assert comp == pytest.approx(0.75, abs=1e-4)

    def test_empty_list_returns_zero(self):
        assert compute_composite([]) == pytest.approx(0.0)

    def test_single_score(self):
        scores = [RubricScore(dimension=RubricDimension.PACE, score=0.55)]
        assert compute_composite(scores) == pytest.approx(0.55, abs=1e-4)
