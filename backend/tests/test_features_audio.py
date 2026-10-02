"""
tests/test_features_audio.py
============================
Tests for src/features/audio.py.

Audio files are synthesised in memory using numpy so no real
recordings are needed.
"""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf
import tempfile
from pathlib import Path

from src.config import FRAME_LENGTH, HOP_LENGTH, PAUSE_MIN_DURATION, SAMPLE_RATE
from src.features.audio import (
    _rms_db,
    detect_pauses,
    extract_all,
    extract_energy,
    extract_pitch,
    frames_to_time,
    load_audio,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_wav(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write a numpy array to a temp WAV file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, waveform.astype(np.float32), sr)
    return Path(tmp.name)


def _sine(freq_hz: float, duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a normalised sine wave."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _silence(duration_s: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(sr * duration_s), dtype=np.float32)


# ── load_audio ───────────────────────────────────────────────────────────────

class TestLoadAudio:
    def test_returns_float32(self):
        wav = _make_wav(_sine(440, 1.0))
        audio = load_audio(wav)
        assert audio.dtype == np.float32

    def test_length_matches_duration(self):
        dur = 2.0
        wav = _make_wav(_sine(220, dur))
        audio = load_audio(wav)
        assert abs(len(audio) / SAMPLE_RATE - dur) < 0.02

    def test_mono_output(self):
        wav = _make_wav(_sine(440, 0.5))
        audio = load_audio(wav)
        assert audio.ndim == 1


# ── frames_to_time ───────────────────────────────────────────────────────────

class TestFramesToTime:
    def test_first_frame_is_zero(self):
        times = frames_to_time(np.array([0]))
        assert times[0] == pytest.approx(0.0, abs=1e-6)

    def test_monotone_increasing(self):
        times = frames_to_time(np.arange(10))
        assert np.all(np.diff(times) > 0)


# ── extract_energy ───────────────────────────────────────────────────────────

class TestExtractEnergy:
    def test_output_shape(self):
        # librosa.feature.rms uses centered framing by default (center=True),
        # which pads the waveform on both sides by FRAME_LENGTH // 2.
        # The padded signal length is len(sig) + 2 * (FRAME_LENGTH // 2).
        # With window length FRAME_LENGTH and step HOP_LENGTH, the number of frames is:
        #   n_frames = 1 + (len(sig) + 2 * (FRAME_LENGTH // 2) - FRAME_LENGTH) // HOP_LENGTH
        #            = 1 + len(sig) // HOP_LENGTH
        # For a 1.0 s signal at 16000 Hz, 1 + 16000 // 512 = 1 + 31 = 32 frames.
        sig = _sine(440, 1.0)
        e = extract_energy(sig)
        expected_frames = 1 + len(sig) // HOP_LENGTH
        assert e.ndim == 1
        assert e.shape[0] == expected_frames
        assert e.shape[0] == 32

    def test_dtype_float64(self):
        e = extract_energy(_sine(440, 0.5))
        assert e.dtype == np.float64

    def test_constant_signal_energy_behavior(self):
        # For a constant DC signal (amplitude 0.5), all samples within the unpadded
        # region have identical RMS = 0.5. However, librosa's centered framing pads
        # the boundaries with zeros (pad_mode='constant', 0).
        # Consequently:
        # 1. Boundary frames overlap with zeros, resulting in RMS < 0.5.
        # 2. Interior frames have constant RMS = 0.5, which is strictly above the mean.
        # 3. Z-scoring yields negative z-scores at the edges and identical positive
        #    z-scores across all interior frames, with mean ~0 and std ~1.
        sig = np.full(SAMPLE_RATE, 0.5, dtype=np.float32)
        e = extract_energy(sig)
        assert e.ndim == 1
        assert np.all(np.isfinite(e))
        # Boundary frames have lower RMS due to zero padding -> negative z-score
        assert e[0] < 0.0
        assert e[-1] < 0.0
        # Interior frames (unaffected by padding) have identical positive z-scores
        pad_frames = (FRAME_LENGTH // 2) // HOP_LENGTH + 1
        interior = e[pad_frames:-pad_frames]
        assert len(interior) > 0
        assert np.allclose(interior, interior[0])
        assert interior[0] > 0.0
        # Overall z-score distribution properties
        assert float(np.mean(e)) == pytest.approx(0.0, abs=1e-6)
        assert float(np.std(e)) == pytest.approx(1.0, abs=1e-6)

    def test_z_score_mean_near_zero(self):
        sig = _sine(110, 2.0)
        e = extract_energy(sig)
        assert abs(float(np.mean(e))) < 0.15


# ── extract_pitch ─────────────────────────────────────────────────────────────

class TestExtractPitch:
    def test_output_shape(self):
        sig = _sine(200, 1.0)
        pitch_st, _ = extract_pitch(sig)
        # pYIN may produce slightly different frame counts due to its internals
        assert pitch_st.ndim == 1
        assert pitch_st.size > 0

    def test_dtype_float64(self):
        pitch_st, _ = extract_pitch(_sine(200, 0.5))
        assert pitch_st.dtype == np.float64

    def test_silence_returns_fill_na(self):
        from src.config import PITCH_FILL_NA
        assert np.isnan(PITCH_FILL_NA)
        silence = _silence(1.0)
        pitch_st, median_hz = extract_pitch(silence)
        # All frames should be NaN (PITCH_FILL_NA) when no voiced content
        assert np.all(np.isnan(pitch_st))

    def test_median_f0_positive(self):
        sig = _sine(180, 1.5)
        _, median_hz = extract_pitch(sig)
        assert median_hz > 0


# ── _rms_db ───────────────────────────────────────────────────────────────────

class TestRmsDb:
    def test_silence_below_threshold(self):
        from src.config import PAUSE_ENERGY_THRESHOLD
        silence = _silence(0.5)
        db = _rms_db(silence, 0, len(silence))
        assert db < PAUSE_ENERGY_THRESHOLD

    def test_loud_sine_above_threshold(self):
        from src.config import PAUSE_ENERGY_THRESHOLD
        sig = _sine(440, 0.1)
        db = _rms_db(sig, 0, len(sig))
        assert db > PAUSE_ENERGY_THRESHOLD

    def test_empty_segment_returns_neg_inf(self):
        sig = _sine(440, 0.1)
        db = _rms_db(sig, 5, 5)   # zero-length slice
        assert db == -np.inf


# ── detect_pauses ─────────────────────────────────────────────────────────────

class TestDetectPauses:
    def test_pure_silence_long_enough_detected(self):
        silence = _silence(1.0)
        pauses = detect_pauses(silence)
        assert len(pauses) > 0, "1 s of silence should produce at least one pause"
        start, end = pauses[0]
        assert end > start

    def test_pure_sine_no_pauses(self):
        sig = _sine(440, 1.0)
        pauses = detect_pauses(sig)
        assert len(pauses) == 0

    def test_short_silence_not_detected(self):
        # Silence shorter than PAUSE_MIN_DURATION should be ignored
        short_silence = _silence(PAUSE_MIN_DURATION * 0.5)
        pauses = detect_pauses(short_silence)
        assert len(pauses) == 0

    def test_pause_boundaries_within_audio(self):
        audio = np.concatenate([_sine(440, 0.5), _silence(0.8), _sine(440, 0.5)])
        duration = len(audio) / SAMPLE_RATE
        pauses = detect_pauses(audio)
        for start, end in pauses:
            assert start >= 0.0
            assert end <= duration + 0.1   # small tolerance for frame rounding


# ── extract_all integration ───────────────────────────────────────────────────

class TestExtractAll:
    def test_returns_audio_features(self):
        from src.features.audio import AudioFeatures
        wav = _make_wav(np.concatenate([_sine(200, 1.5), _silence(0.5), _sine(200, 1.5)]))
        af = extract_all(wav)
        assert isinstance(af, AudioFeatures)
        assert af.sr == SAMPLE_RATE
        assert af.duration > 0
        assert af.pitch_st.ndim == 1
        assert af.energy_z.ndim == 1
        assert isinstance(af.pauses, list)
