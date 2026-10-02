"""
tests/test_features_word_features.py
=====================================
Tests for src/features/word_features.py.

All tests use synthetic pitch/energy arrays so no audio files are required.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.config import HOP_LENGTH, SAMPLE_RATE
from src.features.transcript import Word
from src.features.word_features import (
    WordFeatures,
    extract_all_word_features,
    extract_word_features,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arrays(n_frames: int, pitch_val: float = 2.0, energy_val: float = 0.5):
    """Build constant pitch and energy arrays."""
    pitch = np.full(n_frames, pitch_val, dtype=np.float64)
    energy = np.full(n_frames, energy_val, dtype=np.float64)
    return pitch, energy


def _frames_for(start_s: float, end_s: float) -> tuple[int, int]:
    """Same formula as _word_frame_slice in word_features.py."""
    return (
        int(round(start_s * SAMPLE_RATE / HOP_LENGTH)),
        int(round(end_s * SAMPLE_RATE / HOP_LENGTH)),
    )


# ---------------------------------------------------------------------------
# Basic field correctness
# ---------------------------------------------------------------------------

def test_duration_correct():
    """duration should equal end - start."""
    word = Word("hello", 0.5, 0.9)
    pitch, energy = _make_arrays(50)
    wf = extract_word_features(word, pitch, energy)
    assert abs(wf.duration - 0.4) < 1e-6


def test_pause_before_correct():
    """pause_before = word.start - prev_word.end."""
    word = Word("world", 1.2, 1.6)
    pitch, energy = _make_arrays(100)
    wf = extract_word_features(word, pitch, energy, prev_word_end=1.0)
    assert abs(wf.pause_before - 0.2) < 1e-6


def test_pause_after_correct():
    """pause_after = next_word.start - word.end."""
    word = Word("world", 1.2, 1.6)
    pitch, energy = _make_arrays(100)
    wf = extract_word_features(word, pitch, energy, next_word_start=1.9)
    assert abs(wf.pause_after - 0.3) < 1e-6


def test_pause_clamped_to_zero():
    """Negative pause (overlapping words) should be clamped to 0."""
    word = Word("overlap", 1.0, 1.4)
    pitch, energy = _make_arrays(100)
    # prev word ends AFTER this word starts (overlap)
    wf = extract_word_features(word, pitch, energy, prev_word_end=1.1)
    assert wf.pause_before == 0.0


# ---------------------------------------------------------------------------
# Pitch NaN semantics
# ---------------------------------------------------------------------------

def test_all_nan_pitch_gives_nan_stats():
    """If all frames in the word's span are unvoiced (NaN), pitch stats are NaN."""
    word = Word("silent", 0.0, 0.5)
    n_frames = 50
    pitch = np.full(n_frames, float("nan"), dtype=np.float64)
    energy = np.zeros(n_frames, dtype=np.float64)
    wf = extract_word_features(word, pitch, energy)
    assert math.isnan(wf.pitch_mean_st)
    assert math.isnan(wf.pitch_range_st)


def test_single_voiced_frame_range_is_nan():
    """With only 1 voiced frame, pitch_mean is finite but pitch_range is NaN."""
    word = Word("one", 0.0, 0.1)
    n_frames = 50
    pitch = np.full(n_frames, float("nan"), dtype=np.float64)
    # Force exactly 1 voiced frame inside the word's span
    f_start = int(round(0.0 * SAMPLE_RATE / HOP_LENGTH))
    pitch[f_start] = 3.0
    energy = np.zeros(n_frames, dtype=np.float64)
    wf = extract_word_features(word, pitch, energy)
    assert math.isfinite(wf.pitch_mean_st)
    assert math.isnan(wf.pitch_range_st)


def test_voiced_pitch_mean_and_range():
    """With constant voiced pitch, mean == value and range == 0."""
    word = Word("test", 0.0, 0.5)
    n_frames = 50
    pitch = np.full(n_frames, 4.0, dtype=np.float64)  # all voiced at +4 st
    energy = np.zeros(n_frames, dtype=np.float64)
    wf = extract_word_features(word, pitch, energy)
    assert abs(wf.pitch_mean_st - 4.0) < 1e-6
    assert abs(wf.pitch_range_st - 0.0) < 1e-6


def test_zero_semitone_voiced_not_dropped():
    """A voiced frame at exactly 0 semitones must NOT be treated as unvoiced."""
    word = Word("zero", 0.0, 0.5)
    n_frames = 50
    pitch = np.full(n_frames, 0.0, dtype=np.float64)  # 0.0 is voiced (finite)
    energy = np.zeros(n_frames, dtype=np.float64)
    wf = extract_word_features(word, pitch, energy)
    assert math.isfinite(wf.pitch_mean_st)
    assert abs(wf.pitch_mean_st) < 1e-9


# ---------------------------------------------------------------------------
# Energy
# ---------------------------------------------------------------------------

def test_energy_z_mean_correct():
    """energy_z_mean should be the mean of the word's frame-level z-scores."""
    word = Word("test", 0.0, 0.5)
    n_frames = 50
    pitch, energy = _make_arrays(n_frames, energy_val=1.5)
    wf = extract_word_features(word, pitch, energy)
    assert abs(wf.energy_z_mean - 1.5) < 1e-4


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

def test_batch_length_matches_words():
    """extract_all_word_features should return one entry per word."""
    words = [Word("a", 0.0, 0.3), Word("b", 0.4, 0.7), Word("c", 0.8, 1.0)]
    pitch, energy = _make_arrays(100)
    results = extract_all_word_features(words, pitch, energy)
    assert len(results) == 3


def test_batch_first_no_pause_before():
    """First word has no preceding word; pause_before should be 0."""
    words = [Word("first", 0.0, 0.3), Word("second", 0.4, 0.7)]
    pitch, energy = _make_arrays(50)
    results = extract_all_word_features(words, pitch, energy)
    assert results[0].pause_before == 0.0


def test_batch_last_no_pause_after():
    """Last word has no following word; pause_after should be 0."""
    words = [Word("first", 0.0, 0.3), Word("last", 0.4, 0.7)]
    pitch, energy = _make_arrays(50)
    results = extract_all_word_features(words, pitch, energy)
    assert results[-1].pause_after == 0.0


def test_batch_inter_word_pause():
    """Internal gap between words should appear as pause_after / pause_before."""
    words = [Word("a", 0.0, 0.3), Word("b", 0.5, 0.8)]
    pitch, energy = _make_arrays(100)
    results = extract_all_word_features(words, pitch, energy)
    assert abs(results[0].pause_after - 0.2) < 1e-6
    assert abs(results[1].pause_before - 0.2) < 1e-6


def test_return_type_is_wordfeatures():
    """Each element should be a WordFeatures namedtuple."""
    words = [Word("hello", 0.0, 0.5)]
    pitch, energy = _make_arrays(30)
    result = extract_all_word_features(words, pitch, energy)
    assert isinstance(result[0], WordFeatures)
