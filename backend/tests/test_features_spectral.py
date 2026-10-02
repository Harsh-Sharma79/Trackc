"""
tests/test_features_spectral.py
================================
Synthetic-audio tests for src/features/spectral.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.config import HOP_LENGTH, N_FFT, N_MFCC, SAMPLE_RATE
from src.features.spectral import (
    compute_clarity_snr,
    compute_mfcc,
    compute_spectral_centroid,
    compute_spectral_flatness,
    compute_spectrogram,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float = 440.0, duration_s: float = 1.0) -> np.ndarray:
    """Generate a float32 sine wave at SAMPLE_RATE."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _silence(duration_s: float = 1.0) -> np.ndarray:
    """Generate a float32 all-zero waveform."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _white_noise(duration_s: float = 1.0, seed: int = 42) -> np.ndarray:
    """Generate float32 white noise."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(SAMPLE_RATE * duration_s)).astype(np.float32)


# ---------------------------------------------------------------------------
# compute_spectrogram
# ---------------------------------------------------------------------------

def test_spectrogram_shape():
    """Magnitude STFT should have shape (1 + n_fft//2, n_frames)."""
    sig = _sine()
    spec = compute_spectrogram(sig)
    expected_freq_bins = 1 + N_FFT // 2
    assert spec.shape[0] == expected_freq_bins
    assert spec.ndim == 2
    assert spec.dtype == np.float64


def test_spectrogram_nonnegative():
    """Magnitude spectrogram values must all be >= 0."""
    spec = compute_spectrogram(_sine())
    assert np.all(spec >= 0.0)


# ---------------------------------------------------------------------------
# compute_mfcc
# ---------------------------------------------------------------------------

def test_mfcc_shape():
    """13 MFCCs for a 1-second sine wave."""
    sig = _sine(440.0, 1.0)
    mfcc = compute_mfcc(sig)
    assert mfcc.shape[0] == N_MFCC
    assert mfcc.ndim == 2
    assert mfcc.dtype == np.float64


def test_mfcc_finite():
    """All MFCC values should be finite for a normal sine signal."""
    mfcc = compute_mfcc(_sine())
    assert np.all(np.isfinite(mfcc))


def test_mfcc_different_signals_differ():
    """MFCCs of different signals should not be identical."""
    mfcc_sine = compute_mfcc(_sine(440.0))
    mfcc_noise = compute_mfcc(_white_noise())
    assert not np.allclose(mfcc_sine, mfcc_noise)


# ---------------------------------------------------------------------------
# compute_spectral_centroid
# ---------------------------------------------------------------------------

def test_spectral_centroid_positive():
    """Spectral centroid must be > 0 for any non-silent signal."""
    cent = compute_spectral_centroid(_sine())
    assert np.all(cent > 0.0)


def test_spectral_centroid_shape():
    """Centroid shape should match frame count."""
    sig = _sine(1.0, 1.0)
    cent = compute_spectral_centroid(sig)
    spec = compute_spectrogram(sig)
    assert cent.shape == (spec.shape[1],)


def test_spectral_centroid_high_freq_greater():
    """A 2 kHz sine should have a higher centroid than a 200 Hz sine."""
    cent_low = float(np.mean(compute_spectral_centroid(_sine(200.0))))
    cent_high = float(np.mean(compute_spectral_centroid(_sine(2000.0))))
    assert cent_high > cent_low


# ---------------------------------------------------------------------------
# compute_spectral_flatness
# ---------------------------------------------------------------------------

def test_flatness_range():
    """Spectral flatness must be in [0, 1]."""
    flatness = compute_spectral_flatness(_sine())
    assert np.all(flatness >= 0.0)
    assert np.all(flatness <= 1.0 + 1e-6)


def test_flatness_white_noise_near_one():
    """White noise is spectrally flat — mean flatness should be > 0.5."""
    flatness = compute_spectral_flatness(_white_noise())
    assert float(np.mean(flatness)) > 0.5, (
        "White noise should have high spectral flatness (near 1)"
    )


def test_flatness_sine_near_zero():
    """A pure sine is maximally tonal — mean flatness should be < 0.1."""
    flatness = compute_spectral_flatness(_sine())
    assert float(np.mean(flatness)) < 0.1, (
        "Pure sine should have low spectral flatness (near 0)"
    )


# ---------------------------------------------------------------------------
# compute_clarity_snr
# ---------------------------------------------------------------------------

def test_clarity_snr_sine_high():
    """A signal with large amplitude variation has high percentile-spread SNR.

    The clarity metric measures the dB spread between the 95th and 10th
    percentiles of per-frame RMS.  A constant-amplitude sine has nearly
    identical RMS across all frames → spread ≈ 0.  To test a 'high-clarity'
    case we use a signal that alternates between loud and quiet segments so
    the percentile spread is large.
    """
    # loud half followed by quiet (but non-zero) half
    loud = _sine(440.0, 0.5)
    quiet = _sine(440.0, 0.5) * 0.01
    sig = np.concatenate([loud, quiet])
    snr_db, clarity = compute_clarity_snr(sig)
    assert snr_db > 10.0, f"Expected SNR > 10 dB for amplitude-varied signal, got {snr_db}"
    assert clarity > 0.3, f"Expected clarity > 0.3, got {clarity}"


def test_clarity_snr_silence_low():
    """Silence has negligible SNR and clarity ≈ 0."""
    snr_db, clarity = compute_clarity_snr(_silence())
    # All frames are near -180 dBFS; percentile spread is tiny → SNR ≈ 0
    assert snr_db < 5.0, f"Expected near-zero SNR for silence, got {snr_db}"
    assert clarity < 0.3, f"Expected low clarity for silence, got {clarity}"


def test_clarity_output_types():
    """Output should be (float, float) tuple with values in valid ranges."""
    snr_db, clarity = compute_clarity_snr(_sine())
    assert isinstance(snr_db, float)
    assert isinstance(clarity, float)
    assert 0.0 <= clarity <= 1.0
    assert snr_db >= 0.0
