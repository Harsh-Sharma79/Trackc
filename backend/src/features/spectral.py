"""
src/features/spectral.py
========================
Spectral analysis utilities: 13 MFCCs, spectral centroid, spectral flatness,
STFT magnitude spectrogram helper, and percentile-based clarity (SNR) measure.
"""
from __future__ import annotations

import librosa
import numpy as np

from src.config import (
    CLARITY_HIGH_PERCENTILE,
    CLARITY_IDEAL_SNR_DB,
    CLARITY_LOW_PERCENTILE,
    HOP_LENGTH,
    N_FFT,
    N_MFCC,
    SAMPLE_RATE,
)


def compute_spectrogram(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute magnitude STFT spectrogram.

    Args:
        waveform: 1-D audio samples.
        n_fft: FFT window size.
        hop_length: Hop length between successive frames.

    Returns:
        2-D array of shape (1 + n_fft // 2, n_frames).
    """
    stft = librosa.stft(
        y=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )
    return np.abs(stft).astype(np.float64)


def compute_mfcc(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute 13 Mel-Frequency Cepstral Coefficients (MFCC).

    Args:
        waveform: 1-D audio waveform.
        sr: Sampling rate.
        n_mfcc: Number of MFCCs to return.
        n_fft: FFT window size.
        hop_length: Hop length between frames.

    Returns:
        2-D float64 array of shape (n_mfcc, n_frames).
    """
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )
    return mfcc.astype(np.float64)


def compute_spectral_centroid(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute spectral centroid frequency for each frame.

    Args:
        waveform: 1-D audio waveform.
        sr: Sampling rate.
        n_fft: FFT size.
        hop_length: Hop length.

    Returns:
        1-D float64 array of shape (n_frames,).
    """
    cent = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )[0]
    return cent.astype(np.float64)


def compute_spectral_flatness(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute spectral flatness (Wiener entropy) for each frame.

    Values near 1 indicate noise-like signal, values near 0 indicate tonal signal.

    Args:
        waveform: 1-D audio waveform.
        n_fft: FFT size.
        hop_length: Hop length.

    Returns:
        1-D float64 array of shape (n_frames,).
    """
    flatness = librosa.feature.spectral_flatness(
        y=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )[0]
    return flatness.astype(np.float64)


def compute_clarity_snr(
    waveform: np.ndarray,
    frame_length: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    low_percentile: float = CLARITY_LOW_PERCENTILE,
    high_percentile: float = CLARITY_HIGH_PERCENTILE,
    ideal_snr_db: float = CLARITY_IDEAL_SNR_DB,
) -> tuple[float, float]:
    """Compute SNR-style clarity metric from frame energy percentiles.

    Formula:
        RMS_dB = 20 * log10(max(RMS, 1e-9))
        E_speech = percentile(RMS_dB, high_percentile)
        E_noise  = percentile(RMS_dB, low_percentile)
        Estimated_SNR_dB = E_speech - E_noise
        Clarity_Score = clip(Estimated_SNR_dB / ideal_snr_db, 0.0, 1.0)

    Returns:
        Tuple of (estimated_snr_db, clarity_score [0, 1]).
    """
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    with np.errstate(divide="ignore"):
        rms_db = 20.0 * np.log10(np.maximum(rms, 1e-9))

    e_speech = float(np.percentile(rms_db, high_percentile))
    e_noise = float(np.percentile(rms_db, low_percentile))
    snr_db = max(0.0, e_speech - e_noise)

    clarity_score = float(np.clip(snr_db / ideal_snr_db, 0.0, 1.0))
    return round(snr_db, 2), round(clarity_score, 4)
