"""
src/features/word_features.py
==============================
Per-word acoustic features derived from frame-level pitch and energy arrays.

Provides:
- WordFeatures     – NamedTuple holding one word's features
- extract_word_features  – compute features for a single Word
- extract_all_word_features – batch over a list of Words
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from src.config import HOP_LENGTH, SAMPLE_RATE
from src.features.transcript import Word


class WordFeatures(NamedTuple):
    """Acoustic features for a single word span."""

    word: str
    start: float                # seconds
    end: float                  # seconds
    duration: float             # seconds  (end - start)
    pause_before: float         # seconds of silence before this word
    pause_after: float          # seconds of silence after this word
    pitch_mean_st: float        # mean semitone (NaN if all frames unvoiced)
    pitch_range_st: float       # max-min semitones of voiced frames (NaN if < 2 voiced)
    energy_z_mean: float        # mean z-score energy over word span


def _word_frame_slice(word: Word) -> tuple[int, int]:
    """Convert word start/end seconds to frame index range.

    Uses the same centered-framing formula as librosa (center=True default):
        frame_i = round(t * sr / hop_length)

    Args:
        word: Word with start and end in seconds.

    Returns:
        (frame_start, frame_end) integer indices (frame_end is exclusive).
    """
    frame_start = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    frame_end = int(round(word.end * SAMPLE_RATE / HOP_LENGTH))
    # Ensure at least one frame
    if frame_end <= frame_start:
        frame_end = frame_start + 1
    return frame_start, frame_end


def extract_word_features(
    word: Word,
    pitch_st: np.ndarray,
    energy_z: np.ndarray,
    prev_word_end: float | None = None,
    next_word_start: float | None = None,
) -> WordFeatures:
    """Compute per-word acoustic features from frame-level arrays.

    Pitch statistics are NaN-aware: only voiced (finite) frames are used.
    If no voiced frames exist, pitch_mean_st and pitch_range_st are NaN.

    Args:
        word:            Word with text, start, and end fields.
        pitch_st:        Full-utterance pitch array (semitones, NaN = unvoiced).
        energy_z:        Full-utterance energy z-score array.
        prev_word_end:   End time of the preceding word (None if first word).
        next_word_start: Start time of the following word (None if last word).

    Returns:
        WordFeatures for this word.
    """
    n_frames = len(pitch_st)
    f_start, f_end = _word_frame_slice(word)
    f_start = max(0, min(f_start, n_frames))
    f_end = max(f_start + 1, min(f_end, n_frames))

    pitch_slice = pitch_st[f_start:f_end]
    energy_slice = energy_z[f_start:f_end]

    # Pitch: NaN-aware stats over voiced frames only
    voiced = pitch_slice[np.isfinite(pitch_slice)]
    if voiced.size == 0:
        pitch_mean = float("nan")
        pitch_range = float("nan")
    elif voiced.size == 1:
        pitch_mean = float(voiced[0])
        pitch_range = float("nan")
    else:
        pitch_mean = float(np.nanmean(voiced))
        pitch_range = float(np.nanmax(voiced) - np.nanmin(voiced))

    energy_mean = float(np.mean(energy_slice)) if energy_slice.size > 0 else 0.0

    # Pause gaps
    pause_before = (word.start - prev_word_end) if prev_word_end is not None else 0.0
    pause_after = (next_word_start - word.end) if next_word_start is not None else 0.0
    # Clamp to zero (negative can occur from overlap in proportional timing)
    pause_before = max(0.0, pause_before)
    pause_after = max(0.0, pause_after)

    return WordFeatures(
        word=word.text,
        start=round(word.start, 3),
        end=round(word.end, 3),
        duration=round(word.end - word.start, 3),
        pause_before=round(pause_before, 3),
        pause_after=round(pause_after, 3),
        pitch_mean_st=pitch_mean,
        pitch_range_st=pitch_range,
        energy_z_mean=round(energy_mean, 4),
    )


def extract_all_word_features(
    words: list[Word],
    pitch_st: np.ndarray,
    energy_z: np.ndarray,
) -> list[WordFeatures]:
    """Batch-compute WordFeatures for every word in a list.

    Args:
        words:    Ordered list of Word objects with timing.
        pitch_st: Full-utterance pitch array.
        energy_z: Full-utterance energy z-score array.

    Returns:
        List of WordFeatures in the same order as ``words``.
    """
    results: list[WordFeatures] = []
    for i, word in enumerate(words):
        prev_end = words[i - 1].end if i > 0 else None
        next_start = words[i + 1].start if i < len(words) - 1 else None
        results.append(
            extract_word_features(word, pitch_st, energy_z, prev_end, next_start)
        )
    return results
