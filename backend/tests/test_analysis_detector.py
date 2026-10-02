"""
tests/test_analysis_detector.py
================================
Tests for src/analysis/detector.py.
"""
import numpy as np
import pytest

from src.analysis.detector import (
    _severity,
    detect_all,
    detect_energy_flaws,
    detect_pace_flaws,
    detect_pause_flaws,
    detect_pitch_flaws,
)
from src.features.transcript import Word, estimate_word_times
from src.schema import FlawType


class TestSeverity:
    def test_zero_deviation_returns_zero(self):
        assert _severity(0.0) == pytest.approx(0.0, abs=1e-6)

    def test_large_deviation_approaches_one(self):
        assert _severity(10.0) > 0.99

    def test_output_clipped_to_zero_one(self):
        assert 0.0 <= _severity(0.5) <= 1.0


class TestDetectPaceFlaws:
    def _fast_words(self, n: int = 30, duration: float = 5.0):
        """Words at ~360 WPM (very fast)."""
        tokens = [f"w{i}" for i in range(n)]
        return estimate_word_times(tokens, duration=duration)

    def _slow_words(self, n: int = 10, duration: float = 10.0):
        """Words at ~60 WPM (very slow)."""
        tokens = [f"w{i}" for i in range(n)]
        return estimate_word_times(tokens, duration=duration)

    def test_fast_pace_detected(self):
        words = self._fast_words()
        flaws = detect_pace_flaws(words)
        assert any(f.flaw_type == FlawType.PACE_TOO_FAST for f in flaws)

    def test_slow_pace_detected(self):
        words = self._slow_words()
        flaws = detect_pace_flaws(words)
        assert any(f.flaw_type == FlawType.PACE_TOO_SLOW for f in flaws)

    def test_ideal_pace_no_flaws(self):
        # Realistic varied word timings: word durations naturally fluctuate
        # between 0.32s and 0.49s (reflecting short vs long words), maintaining
        # local pacing safely within the 120 - 160 WPM ideal band across all 10-word windows.
        words: list[Word] = []
        cur_time = 0.0
        durations = [0.38, 0.45, 0.32, 0.49, 0.40, 0.44, 0.36, 0.47, 0.41, 0.43] * 15
        for i, dur in enumerate(durations):
            words.append(Word(text=f"word_{i}", start=round(cur_time, 4), end=round(cur_time + dur, 4)))
            cur_time += dur

        flaws = detect_pace_flaws(words)
        pace_flaws = [
            f for f in flaws
            if f.flaw_type in (FlawType.PACE_TOO_FAST, FlawType.PACE_TOO_SLOW)
        ]
        assert len(pace_flaws) == 0, (
            f"Expected no pace flaws with realistic varied timings, got: {[(f.flaw_type, f.metadata) for f in pace_flaws]}"
        )


    def test_flaw_regions_have_valid_times(self):
        words = self._fast_words()
        for flaw in detect_pace_flaws(words):
            assert flaw.end > flaw.start

    def test_flaw_severity_in_bounds(self):
        words = self._fast_words()
        for flaw in detect_pace_flaws(words):
            assert 0.0 <= flaw.severity <= 1.0


class TestDetectPitchFlaws:
    def test_monotone_detected(self):
        # A constant non-zero voiced pitch has std = 0, which should trigger PITCH_MONOTONE.
        pitch = np.full(300, 2.0)   # constant 2.0 semitones
        flaws = detect_pitch_flaws(pitch)
        assert any(f.flaw_type == FlawType.PITCH_MONOTONE for f in flaws)

    def test_all_nan_unvoiced_produces_no_flaws(self):
        # An entirely unvoiced audio segment (all NaN pitch frames) should
        # be skipped and produce zero pitch flaws.
        pitch = np.full(300, np.nan)
        flaws = detect_pitch_flaws(pitch)
        assert len(flaws) == 0

    def test_ideal_pitch_no_flaws(self):
        from src.config import PITCH_VAR_IDEAL_MIN, PITCH_VAR_IDEAL_MAX
        ideal_std = (PITCH_VAR_IDEAL_MIN + PITCH_VAR_IDEAL_MAX) / 2
        rng = np.random.default_rng(42)
        pitch = rng.normal(0, ideal_std, 300)
        flaws = detect_pitch_flaws(pitch)
        types = {f.flaw_type for f in flaws}
        assert FlawType.PITCH_MONOTONE not in types

    def test_flaw_times_non_negative(self):
        pitch = np.full(200, 2.0)
        flaws = detect_pitch_flaws(pitch)
        assert len(flaws) > 0
        for flaw in flaws:
            assert flaw.start >= 0.0
            assert flaw.end > flaw.start

    def test_too_few_voiced_frames_skipped(self):
        # Only 3 voiced frames in a sea of unvoiced NaN frames -> window skipped
        pitch = np.full(200, np.nan)
        pitch[50:53] = 1.0
        flaws = detect_pitch_flaws(pitch)
        assert isinstance(flaws, list)
        assert len(flaws) == 0



class TestDetectEnergyFlaws:
    def test_low_energy_detected(self):
        # Mean energy deeply negative
        energy = np.full(200, -3.0)
        flaws = detect_energy_flaws(energy)
        assert any(f.flaw_type == FlawType.ENERGY_LOW for f in flaws)

    def test_inconsistent_energy_detected(self):
        rng = np.random.default_rng(0)
        energy = rng.normal(0, 4.0, 200)   # very high std
        flaws = detect_energy_flaws(energy)
        assert any(f.flaw_type == FlawType.ENERGY_INCONSISTENT for f in flaws)

    def test_normal_energy_no_flaws(self):
        rng = np.random.default_rng(42)
        energy = rng.normal(0, 1.0, 200)
        flaws = detect_energy_flaws(energy)
        assert len(flaws) == 0

    def test_severity_in_bounds(self):
        energy = np.full(200, -3.0)
        for flaw in detect_energy_flaws(energy):
            assert 0.0 <= flaw.severity <= 1.0


class TestDetectPauseFlaws:
    def test_excessive_pause_detected(self):
        from src.config import PAUSE_IDEAL_MAX
        pauses = [(1.0, 1.0 + PAUSE_IDEAL_MAX * 3)]
        flaws = detect_pause_flaws(pauses, duration=10.0)
        assert any(f.flaw_type == FlawType.PAUSE_EXCESSIVE for f in flaws)

    def test_ideal_pause_no_flaw(self):
        from src.config import PAUSE_IDEAL_MAX
        pauses = [(1.0, 1.0 + PAUSE_IDEAL_MAX * 0.5)]
        flaws = detect_pause_flaws(pauses, duration=10.0)
        assert len(flaws) == 0

    def test_no_pauses_returns_empty(self):
        flaws = detect_pause_flaws([], duration=30.0)
        assert flaws == []

    def test_flaw_region_bounds(self):
        from src.config import PAUSE_IDEAL_MAX
        pauses = [(2.0, 2.0 + PAUSE_IDEAL_MAX * 2)]
        for flaw in detect_pause_flaws(pauses, duration=15.0):
            assert flaw.end > flaw.start


class TestDetectAll:
    def test_returns_sorted_list(self):
        pitch = np.zeros(300)
        energy = np.full(300, -3.0)
        pauses = [(5.0, 7.0)]
        tokens = [f"w{i}" for i in range(30)]
        words = estimate_word_times(tokens, duration=5.0)
        flaws = detect_all(pitch, energy, pauses, words, duration=60.0)
        assert isinstance(flaws, list)
        starts = [f.start for f in flaws]
        assert starts == sorted(starts)

    def test_all_flaw_types_have_explanation(self):
        pitch = np.zeros(300)
        energy = np.full(300, -3.0)
        pauses = [(5.0, 7.0)]
        tokens = [f"w{i}" for i in range(30)]
        words = estimate_word_times(tokens, duration=5.0)
        flaws = detect_all(pitch, energy, pauses, words, duration=60.0)
        for flaw in flaws:
            assert len(flaw.explanation) > 0
