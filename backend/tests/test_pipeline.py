"""
tests/test_pipeline.py
=======================
Integration test for src/pipeline.py.

Synthesises two WAV files (baseline + candidate) in memory and exercises
the full pipeline end-to-end.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE
from src.pipeline import evaluate
from src.schema import EvaluationResult


TRANSCRIPT = (
    "The quick brown fox jumps over the lazy dog. "
    "Speech evaluation is an important task for public speaking coaches. "
    "Consistent pace, clear pronunciation, and appropriate pauses matter."
)


def _sine(freq: float, duration: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _make_wav(waveform: np.ndarray) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, waveform, SAMPLE_RATE)
    return Path(tmp.name)


@pytest.fixture(scope="module")
def audio_paths():
    baseline = np.concatenate([_sine(200, 3.0), np.zeros(int(0.4 * SAMPLE_RATE)), _sine(200, 3.0)])
    candidate = np.concatenate([_sine(180, 2.5), np.zeros(int(0.6 * SAMPLE_RATE)), _sine(220, 3.5)])
    b_path = _make_wav(baseline)
    c_path = _make_wav(candidate)
    yield b_path, c_path


class TestEvaluatePipeline:
    def test_returns_evaluation_result(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="test_run")
        assert isinstance(result, EvaluationResult)

    def test_audio_id_preserved(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="my_id_42")
        assert result.audio_id == "my_id_42"

    def test_transcript_preserved(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        assert result.transcript == TRANSCRIPT

    def test_duration_positive(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        assert result.duration > 0.0

    def test_four_rubric_dimensions(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        assert len(result.rubric_scores) == 4

    def test_composite_score_in_range(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        assert 0.0 <= result.composite_score <= 1.0

    def test_flaw_regions_valid(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        for flaw in result.flaw_regions:
            assert flaw.end > flaw.start
            assert 0.0 <= flaw.severity <= 1.0
            assert len(flaw.explanation) > 0

    def test_flaw_regions_sorted(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        starts = [f.start for f in result.flaw_regions]
        assert starts == sorted(starts)

    def test_json_serialisable(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        restored = EvaluationResult.model_validate_json(json_str)
        assert restored.audio_id == result.audio_id

    def test_deterministic(self, audio_paths):
        """Two identical calls must produce identical composite scores."""
        b_path, c_path = audio_paths
        r1 = evaluate(c_path, b_path, TRANSCRIPT, audio_id="det")
        r2 = evaluate(c_path, b_path, TRANSCRIPT, audio_id="det")
        assert r1.composite_score == r2.composite_score

    def test_summary_non_empty(self, audio_paths):
        b_path, c_path = audio_paths
        result = evaluate(c_path, b_path, TRANSCRIPT, audio_id="t")
        assert len(result.summary) > 0
