"""
tests/test_pitch_nan.py
=======================
Tests verifying NaN representation for unvoiced frames, ensuring that:
1. Signals with voiced and silent segments yield NaN in silent parts and finite pitch in voiced parts.
2. A voiced frame exactly at 0.0 semitones (at the speaker's median F0) is NOT dropped.
3. Pitch variation score does not change when silence is appended or prepended.
4. Schema models handle NaN values in diagnostic metadata or pitch time-series by serializing them as JSON null.
"""
from __future__ import annotations

import json
import numpy as np
import pytest

from src.analysis.scoring import score_pitch_variation
from src.config import SAMPLE_RATE
from src.features.audio import extract_pitch
from src.schema import FlawRegion, FlawType, RubricDimension, RubricScore


def _sine(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


def test_voiced_and_silent_segments_yield_nan_and_finite():
    """Voiced segment yields finite pitch semitones; silent segment yields NaN."""
    # 1.5s sine wave (voiced at 220 Hz) followed by 1.0s silence
    voiced = _sine(220.0, 1.5)
    silent = _silence(1.0)
    audio = np.concatenate([voiced, silent])

    pitch_st, median_f0 = extract_pitch(audio)

    assert pitch_st.ndim == 1
    assert pitch_st.size > 0

    # The voiced portion should contain finite pitch estimates
    # (Checking frames in the middle of the voiced portion)
    assert np.any(np.isfinite(pitch_st[: len(pitch_st) // 2]))

    # The silent portion (near the end) should be NaN
    assert np.any(np.isnan(pitch_st[int(len(pitch_st) * 0.6) :]))

    # Median F0 should be close to 220 Hz
    assert median_f0 == pytest.approx(220.0, rel=0.05)


def test_voiced_frame_at_zero_semitones_not_dropped():
    """A voiced frame exactly at 0.0 semitones (speaker median) must NOT be dropped.

    Previously, `pitch_st != 0.0` incorrectly dropped 0.0 semitone voiced frames.
    Now, `np.isfinite(pitch_st)` correctly retains them.
    """
    # Create an array with two voiced frames at exactly 0.0 semitones and some NaNs
    pitch_st = np.array([np.nan, 0.0, 0.0, np.nan, 1.0, -1.0])

    # With np.isfinite, the four finite frames [0.0, 0.0, 1.0, -1.0] are kept.
    # The mean is 0.0, variance is (0^2 + 0^2 + 1^2 + (-1)^2)/4 = 0.5, std = sqrt(0.5) ≈ 0.7071
    score = score_pitch_variation(pitch_st)

    assert score.dimension == RubricDimension.PITCH_VARIATION
    # If 0.0 were dropped, std of [1.0, -1.0] would be 1.0. With 0.0 kept, std is ~0.7071.
    expected_std = float(np.std([0.0, 0.0, 1.0, -1.0]))
    assert f"{expected_std:.2f}" in score.details


def test_pitch_variation_score_unchanged_when_silence_added():
    """Appending silence (which produces NaN frames) does not alter pitch variation score."""
    rng = np.random.default_rng(42)
    # Voiced pitch trajectory (varying around median)
    voiced_pitch = rng.normal(0.0, 3.5, 150)

    score_without_silence = score_pitch_variation(voiced_pitch)

    # Add unvoiced frames (NaN) representing silence
    pitch_with_silence = np.concatenate([
        np.full(50, np.nan),
        voiced_pitch,
        np.full(50, np.nan),
    ])

    score_with_silence = score_pitch_variation(pitch_with_silence)

    assert score_without_silence.score == score_with_silence.score
    assert score_without_silence.details == score_with_silence.details


def test_schema_nan_serialization_to_null():
    """Pydantic serialization should convert NaN values in metadata/time-series to JSON null."""
    flaw = FlawRegion(
        start=1.0,
        end=2.5,
        flaw_type=FlawType.PITCH_MONOTONE,
        severity=0.6,
        explanation="Monotone region detected.",
        metadata={
            "f0_semitones": [np.nan, 0.2, 0.3, float("nan")],
            "avg_pitch": float("nan"),
        },
    )

    json_str = flaw.model_dump_json()

    # In JSON specification, NaN is not valid; Pydantic serializes NaN as null
    assert "null" in json_str
    parsed = json.loads(json_str)
    assert parsed["metadata"]["f0_semitones"] == [None, 0.2, 0.3, None]
    assert parsed["metadata"]["avg_pitch"] is None
