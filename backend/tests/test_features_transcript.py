"""
tests/test_features_transcript.py
==================================
Tests for src/features/transcript.py.
"""
import pytest
from src.features.transcript import (
    Word,
    estimate_word_times,
    tokenize,
    words_to_pace_wpm,
)


class TestTokenize:
    def test_basic_split(self):
        tokens = tokenize("Hello, World!")
        assert tokens == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = tokenize("It's a test—OK?")
        assert "it's" in tokens or "its" in tokens  # apostrophe behaviour
        assert "ok" in tokens

    def test_numbers_kept(self):
        tokens = tokenize("Step 1 and step 2")
        assert "1" in tokens
        assert "2" in tokens

    def test_repeated_spaces(self):
        tokens = tokenize("  one   two   three  ")
        assert tokens == ["one", "two", "three"]

    def test_all_punctuation_returns_empty(self):
        assert tokenize("!!! ??? ...") == []


class TestEstimateWordTimes:
    def test_returns_correct_count(self):
        words = ["hello", "world", "test"]
        result = estimate_word_times(words, duration=3.0)
        assert len(result) == 3

    def test_times_span_duration(self):
        words = ["a", "bb", "ccc"]
        result = estimate_word_times(words, duration=6.0)
        assert result[0].start == pytest.approx(0.0, abs=1e-5)
        assert result[-1].end == pytest.approx(6.0, abs=1e-5)

    def test_end_after_start_for_each_word(self):
        words = ["alpha", "bravo", "charlie"]
        result = estimate_word_times(words, duration=5.0)
        for w in result:
            assert w.end > w.start

    def test_monotone_times(self):
        words = ["one", "two", "three", "four"]
        result = estimate_word_times(words, duration=4.0)
        for i in range(len(result) - 1):
            assert result[i].end <= result[i + 1].start + 1e-9

    def test_empty_words_returns_empty(self):
        result = estimate_word_times([], duration=5.0)
        assert result == []

    def test_speech_start_offset(self):
        words = ["a", "b"]
        result = estimate_word_times(words, duration=4.0, speech_start=1.0, speech_end=3.0)
        assert result[0].start >= 1.0
        assert result[-1].end <= 3.0 + 1e-9

    def test_invalid_speech_end_raises(self):
        with pytest.raises(ValueError):
            estimate_word_times(["a", "b"], duration=5.0, speech_start=3.0, speech_end=1.0)


class TestWordsToPaceWpm:
    def test_exact_wpm(self):
        # 60 words over 60 s → 60 WPM
        words = estimate_word_times([f"w{i}" for i in range(60)], duration=60.0)
        wpm = words_to_pace_wpm(words)
        assert wpm == pytest.approx(60.0, rel=0.01)

    def test_single_word_returns_zero(self):
        words = estimate_word_times(["hello"], duration=5.0)
        assert words_to_pace_wpm(words) == 0.0

    def test_empty_returns_zero(self):
        assert words_to_pace_wpm([]) == 0.0

    def test_150_wpm(self):
        # 150 words over 60 s → 150 WPM
        words = estimate_word_times([f"w{i}" for i in range(150)], duration=60.0)
        wpm = words_to_pace_wpm(words)
        assert wpm == pytest.approx(150.0, rel=0.01)
