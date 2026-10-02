"""
src/features/transcript.py
==========================
Transcript processing utilities.

Provides simple, dependency-light word-level timing estimation.
When forced-alignment tooling (e.g. aeneas, WhisperX) is unavailable,
we estimate per-word timings by distributing the audio duration
proportionally to character count.

Public API:
- Word               – NamedTuple for a word with start/end times
- tokenize           – split raw transcript string into Word tokens
- estimate_word_times – assign start/end times proportional to char length
- words_to_pace_wpm  – compute speaking rate in words-per-minute
"""
from __future__ import annotations

import re
from typing import NamedTuple


class Word(NamedTuple):
    """A single word with its estimated time boundaries."""

    text: str
    start: float   # seconds
    end: float     # seconds


def tokenize(transcript: str) -> list[str]:
    """Split a raw transcript into a list of word tokens.

    Strips punctuation and normalises whitespace.  Returns only non-empty
    tokens so that repeated spaces / punctuation-only segments are ignored.

    Args:
        transcript: Raw transcript string.

    Returns:
        List of lowercase word strings (no punctuation).
    """
    # Remove all characters except letters, digits, apostrophes, hyphens, spaces
    cleaned = re.sub(r"[^a-zA-Z0-9'\- ]+", " ", transcript)
    tokens = [t.strip("-'").lower() for t in cleaned.split() if t.strip("-'")]
    return [t for t in tokens if t]


def estimate_word_times(
    words: list[str],
    duration: float,
    speech_start: float = 0.0,
    speech_end: float | None = None,
) -> list[Word]:
    """Assign per-word start/end times proportional to character length.

    This is a lightweight fallback when forced alignment is unavailable.
    The total speech interval [speech_start, speech_end] is divided among
    words proportionally to their character lengths.

    Args:
        words:        List of word strings (from tokenize).
        duration:     Total audio duration in seconds (used to set speech_end
                      when speech_end is None).
        speech_start: Start of the speaking region in seconds.
        speech_end:   End of the speaking region in seconds.  Defaults to
                      ``duration``.

    Returns:
        List of Word named-tuples with estimated start/end times.
    """
    if not words:
        return []

    if speech_end is None:
        speech_end = duration

    if speech_end <= speech_start:
        raise ValueError(
            f"speech_end ({speech_end}) must be greater than speech_start ({speech_start})."
        )

    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        # All words are empty strings – distribute equally
        total_chars = len(words)
        char_counts = [1] * len(words)
    else:
        char_counts = [max(len(w), 1) for w in words]

    speech_span = speech_end - speech_start
    result: list[Word] = []
    cursor = speech_start

    for w, chars in zip(words, char_counts):
        proportion = chars / sum(char_counts)
        word_dur = proportion * speech_span
        result.append(Word(text=w, start=round(cursor, 6), end=round(cursor + word_dur, 6)))
        cursor += word_dur

    return result


def words_to_pace_wpm(words: list[Word]) -> float:
    """Compute speaking rate in words-per-minute from a list of Word timings.

    Args:
        words: List of Word objects with start/end times.

    Returns:
        Speaking rate in words-per-minute, or 0.0 if fewer than 2 words.
    """
    if len(words) < 2:
        return 0.0

    total_duration_minutes = (words[-1].end - words[0].start) / 60.0
    if total_duration_minutes <= 0.0:
        return 0.0

    return len(words) / total_duration_minutes
