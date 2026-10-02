"""
src/features/audio.py
=====================
Low-level audio feature extraction.

Provides:
- load_audio          – load & resample to mono 16 kHz
- extract_pitch       – per-frame F0 in semitones (relative to speaker median)
- extract_energy      – per-frame RMS energy as z-score
- detect_pauses       – list of (start_s, end_s) silence intervals
- frames_to_time      – convert frame indices to seconds
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple

import librosa
import numpy as np

from src.config import (
    CHANNELS,
    FMAX_HZ,
    FMIN_HZ,
    FRAME_LENGTH,
    HOP_LENGTH,
    N_FFT,
    PAUSE_ENERGY_THRESHOLD,
    PAUSE_MIN_DURATION,
    PITCH_FILL_NA,
    SAMPLE_RATE,
)


class AudioFeatures(NamedTuple):
    """Container for all per-frame features extracted from one audio file."""

    waveform: np.ndarray          # shape (n_samples,), float32
    pitch_st: np.ndarray          # shape (n_frames,), semitones rel. to median F0
    energy_z: np.ndarray          # shape (n_frames,), z-score of RMS
    pauses: list[tuple[float, float]]  # list of (start_s, end_s)
    duration: float               # seconds
    sr: int                       # sample rate (always SAMPLE_RATE)
    median_f0_hz: float           # speaker's median voiced F0 in Hz


def load_audio(path: str | Path) -> np.ndarray:
    """Load an audio file, resample to SAMPLE_RATE, and convert to mono float32.

    Args:
        path: Filesystem path to any audio format supported by librosa/soundfile.

    Returns:
        1-D float32 numpy array of audio samples at SAMPLE_RATE.
    """
    waveform, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    return waveform


def frames_to_time(frame_indices: np.ndarray) -> np.ndarray:
    """Convert frame indices to time in seconds using global hop/sample-rate settings.

    Args:
        frame_indices: Integer array of frame indices.

    Returns:
        Float array of corresponding start times in seconds.
    """
    return librosa.frames_to_time(
        frame_indices, sr=SAMPLE_RATE, hop_length=HOP_LENGTH
    )


def extract_pitch(waveform: np.ndarray) -> tuple[np.ndarray, float]:
    """Estimate per-frame F0 using pYIN, expressed as semitones relative to median F0.

    Unvoiced frames are assigned NaN (from PITCH_FILL_NA).

    Args:
        waveform: 1-D float32 waveform at SAMPLE_RATE.

    Returns:
        Tuple of:
          - pitch_st: float64 array of shape (n_frames,) in semitones (NaN for unvoiced).
          - median_f0_hz: median voiced F0 in Hz (for downstream reference).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f0, voiced_flag, _ = librosa.pyin(
            waveform,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            sr=SAMPLE_RATE,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH,
            fill_na=None,   # keep NaN so we can compute median on voiced only
        )

    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if voiced_f0.size == 0:
        # No voiced frames; fall back to midpoint of expected range
        median_hz = float(np.sqrt(FMIN_HZ * FMAX_HZ))
    else:
        median_hz = float(np.median(voiced_f0))

    # Convert Hz → semitones relative to speaker median
    # semitone(f) = 12 * log2(f / median_hz)
    with np.errstate(divide="ignore", invalid="ignore"):
        pitch_st = np.where(
            voiced_flag & ~np.isnan(f0),
            12.0 * np.log2(np.where(f0 > 0, f0, 1.0) / median_hz),
            PITCH_FILL_NA,
        )

    return pitch_st.astype(np.float64), median_hz


def extract_energy(waveform: np.ndarray) -> np.ndarray:
    """Compute per-frame RMS energy and return as z-score.

    Args:
        waveform: 1-D float32 waveform at SAMPLE_RATE.

    Returns:
        float64 array of shape (n_frames,) containing z-scored RMS energy.
        If all frames have identical energy (std == 0), returns zeros.
    """
    rms = librosa.feature.rms(
        y=waveform, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]  # shape (n_frames,)

    mean = float(np.mean(rms))
    std = float(np.std(rms))
    if std < 1e-9:
        return np.zeros_like(rms, dtype=np.float64)

    energy_z = (rms - mean) / std
    return energy_z.astype(np.float64)


def _rms_db(waveform: np.ndarray, frame_start: int, frame_end: int) -> float:
    """Compute dBFS RMS for a sample-index slice of the waveform."""
    segment = waveform[frame_start:frame_end]
    if segment.size == 0:
        return -np.inf
    rms_val = float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))
    if rms_val < 1e-10:
        return -96.0
    return 20.0 * np.log10(rms_val)


def detect_pauses(waveform: np.ndarray) -> list[tuple[float, float]]:
    """Detect silence intervals in the waveform.

    A frame is considered silence when its dBFS RMS is below
    PAUSE_ENERGY_THRESHOLD.  Consecutive silent frames are merged into a
    single interval; only intervals of at least PAUSE_MIN_DURATION are kept.

    Args:
        waveform: 1-D float32 waveform at SAMPLE_RATE.

    Returns:
        Sorted list of (start_s, end_s) tuples for each detected pause.
    """
    n_frames = 1 + (len(waveform) - FRAME_LENGTH) // HOP_LENGTH
    if n_frames <= 0:
        return []

    silent_flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start_sample = i * HOP_LENGTH
        end_sample = start_sample + FRAME_LENGTH
        silent_flags[i] = _rms_db(waveform, start_sample, end_sample) < PAUSE_ENERGY_THRESHOLD

    # Group consecutive silent frames into intervals
    pauses: list[tuple[float, float]] = []
    in_pause = False
    pause_start_frame = 0

    for i, is_silent in enumerate(silent_flags):
        if is_silent and not in_pause:
            in_pause = True
            pause_start_frame = i
        elif not is_silent and in_pause:
            in_pause = False
            start_s = float(pause_start_frame * HOP_LENGTH / SAMPLE_RATE)
            end_s = float(i * HOP_LENGTH / SAMPLE_RATE)
            if (end_s - start_s) >= PAUSE_MIN_DURATION:
                pauses.append((start_s, end_s))

    # Handle a pause that reaches the end of the waveform
    if in_pause:
        start_s = float(pause_start_frame * HOP_LENGTH / SAMPLE_RATE)
        end_s = float(len(waveform) / SAMPLE_RATE)
        if (end_s - start_s) >= PAUSE_MIN_DURATION:
            pauses.append((start_s, end_s))

    return pauses


_AUDIO_FEATURES_CACHE: dict[str, AudioFeatures] = {}


def extract_all(path: str | Path, use_cache: bool = True) -> AudioFeatures:
    """Full feature-extraction pipeline for a single audio file.

    Caches extracted features in memory by resolved absolute path so repeated
    evaluations against the same baseline audio file do not re-run expensive pYIN.

    Args:
        path: Path to the audio file.
        use_cache: Whether to use/populate the in-memory features cache.

    Returns:
        AudioFeatures named tuple with waveform, pitch, energy, pauses,
        duration, sample rate, and median F0.
    """
    resolved_path = str(Path(path).resolve())
    if use_cache and resolved_path in _AUDIO_FEATURES_CACHE:
        return _AUDIO_FEATURES_CACHE[resolved_path]

    waveform = load_audio(path)
    pitch_st, median_f0_hz = extract_pitch(waveform)
    energy_z = extract_energy(waveform)
    pauses = detect_pauses(waveform)
    duration = float(len(waveform) / SAMPLE_RATE)

    features = AudioFeatures(
        waveform=waveform,
        pitch_st=pitch_st,
        energy_z=energy_z,
        pauses=pauses,
        duration=duration,
        sr=SAMPLE_RATE,
        median_f0_hz=median_f0_hz,
    )
    if use_cache:
        _AUDIO_FEATURES_CACHE[resolved_path] = features
    return features
