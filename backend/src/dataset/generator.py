"""
src/dataset/generator.py
========================
Region-based flaw injection for building the labeled dataset.

Injectors:
- inject_slow          – librosa time_stretch (rate < 1 → slower)
- inject_fast          – librosa time_stretch (rate > 1 → faster)
- inject_pause         – insert silence block
- inject_monotone      – Parselmouth pitch-tier flattening (fallback: skip voiced frames)
- inject_low_energy    – scale amplitude by dB drop
- inject_filler        – insert synthetic buzz clip
- inject_unclear       – add Gaussian noise at target SNR

Each injector takes (waveform, region_start_s, region_end_s, severity_level, rng)
and returns (new_waveform, label_start_s, label_end_s).

The label timestamps reflect the NEW audio timeline (after duration shifts from
stretch or insertion), so downstream metrics see correct ground-truth boundaries.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from src.config import (
    INJECT_ENERGY_DB_DROP,
    INJECT_FADE_S,
    INJECT_FAST_RATES,
    INJECT_FILLER_AMPLITUDE,
    INJECT_FILLER_DURATIONS,
    INJECT_FILLER_FREQ_HZ,
    INJECT_MONOTONE_FLATTEN,
    INJECT_NOISE_SNR,
    INJECT_PAUSE_DURATIONS,
    INJECT_REGION_MAX_WORDS,
    INJECT_REGION_MIN_WORDS,
    INJECT_SLOW_RATES,
    SAMPLE_RATE,
)
from src.features.transcript import Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _samples(seconds: float) -> int:
    """Convert seconds to integer sample count."""
    return int(round(seconds * SAMPLE_RATE))


def _fade(region: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply linear fade-in and fade-out to region to avoid clicks.

    Args:
        region:       1-D float32 array.
        fade_samples: Number of samples for each fade ramp.

    Returns:
        Region with fades applied in-place (copy returned).
    """
    region = region.copy()
    n = len(region)
    fade_n = min(fade_samples, n // 2)
    if fade_n > 0:
        ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        region[:fade_n] *= ramp
        region[-fade_n:] *= ramp[::-1]
    return region


def pick_region(
    words: list[Word],
    rng: np.random.Generator,
    min_words: int = INJECT_REGION_MIN_WORDS,
    max_words: int = INJECT_REGION_MAX_WORDS,
) -> tuple[int, int]:
    """Randomly pick a contiguous word-range for flaw injection.

    Args:
        words:     Ordered list of Word objects.
        rng:       Seeded numpy Generator.
        min_words: Minimum number of words in the region.
        max_words: Maximum number of words in the region.

    Returns:
        (word_start_idx, word_end_idx) — inclusive indices into words.

    Raises:
        ValueError: If the word list is too short.
    """
    if len(words) < min_words:
        raise ValueError(
            f"Need at least {min_words} words; got {len(words)}."
        )
    region_len = int(rng.integers(min_words, min(max_words, len(words)) + 1))
    start_idx = int(rng.integers(0, len(words) - region_len + 1))
    return start_idx, start_idx + region_len - 1


# ---------------------------------------------------------------------------
# Core injection function
# ---------------------------------------------------------------------------

def inject(
    waveform: np.ndarray,
    region_start_s: float,
    region_end_s: float,
    injector: Callable[
        [np.ndarray, np.ndarray, int, np.random.Generator],
        tuple[np.ndarray, float],
    ],
    severity: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    """Replace [region_start_s, region_end_s] using the given injector function.

    The injector receives (pre_region, region_samples, severity, rng) and
    returns (replacement_region, duration_delta_s).  The pre and post segments
    are spliced around the replacement.

    Args:
        waveform:       Full 1-D float32 waveform.
        region_start_s: Start of the region to replace (seconds).
        region_end_s:   End of the region to replace (seconds).
        injector:       Callable that modifies the region.
        severity:       Integer severity level 1–5.
        rng:            Seeded numpy Generator.

    Returns:
        Tuple of (new_waveform, label_start_s, label_end_s).
    """
    fade_n = _samples(INJECT_FADE_S)
    s_start = _samples(region_start_s)
    s_end = _samples(region_end_s)
    s_end = min(s_end, len(waveform))

    pre = waveform[:s_start]
    region = waveform[s_start:s_end].astype(np.float32)
    post = waveform[s_end:]

    new_region, delta_s = injector(pre, region, severity, rng)
    new_region = _fade(new_region.astype(np.float32), fade_n)

    new_waveform = np.concatenate([pre, new_region, post])
    label_start_s = region_start_s
    label_end_s = region_end_s + delta_s
    return new_waveform, label_start_s, label_end_s


# ---------------------------------------------------------------------------
# Injector functions
# ---------------------------------------------------------------------------

def _injector_slow(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Time-stretch region to be slower.

    INJECT_SLOW_RATES stores a slowdown multiplier (> 1 means that many times slower).
    librosa.time_stretch uses rate as playback speed (> 1 = faster, < 1 = slower),
    so we pass 1 / multiplier to librosa.
    """
    multiplier = INJECT_SLOW_RATES[severity]
    # librosa rate = 1/multiplier → output is multiplier times longer
    stretched = librosa.effects.time_stretch(
        region.astype(np.float32), rate=1.0 / multiplier
    )
    delta_s = (len(stretched) - len(region)) / SAMPLE_RATE
    return stretched, delta_s


def _injector_fast(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Time-stretch region to be faster.

    INJECT_FAST_RATES stores the target playback-speed fraction (< 1 means faster).
    We pass this directly to librosa as the rate (< 1 = longer output, > 1 = shorter).
    Wait — INJECT_FAST_RATES values are < 1 (e.g. 0.88), meaning the audio is
    compressed to 88% of original duration, so rate = 1/fraction > 1 → shorter.
    """
    fraction = INJECT_FAST_RATES[severity]  # fraction of original duration kept
    # librosa rate > 1 → shorter output; rate = 1/fraction achieves the target compression
    stretched = librosa.effects.time_stretch(
        region.astype(np.float32), rate=1.0 / fraction
    )
    delta_s = (len(stretched) - len(region)) / SAMPLE_RATE
    return stretched, delta_s


def _injector_pause(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Insert a silence block at the beginning of the region."""
    pause_dur = INJECT_PAUSE_DURATIONS[severity]
    silence = np.zeros(_samples(pause_dur), dtype=np.float32)
    new_region = np.concatenate([silence, region])
    return new_region, pause_dur


def _injector_monotone(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Flatten pitch variation using Parselmouth (if available) or silence swap.

    Parselmouth IS installed (praat-parselmouth==0.4.7), so we use it.
    The pitch tier is read, its variation around the mean is reduced by
    INJECT_MONOTONE_FLATTEN[severity] (1.0 = fully flattened to mean).
    """
    try:
        import parselmouth
        from parselmouth.praat import call

        flatten = INJECT_MONOTONE_FLATTEN[severity]
        snd = parselmouth.Sound(
            region.astype(np.float64), sampling_frequency=float(SAMPLE_RATE)
        )
        manipulation = call(snd, "To Manipulation", 0.01, 75, 600)
        pitch_tier = call(manipulation, "Extract pitch tier")

        n_points = call(pitch_tier, "Get number of points")
        if n_points > 0:
            times = [
                call(pitch_tier, "Get time from index", i + 1)
                for i in range(n_points)
            ]
            values = [
                call(pitch_tier, "Get value at index", i + 1)
                for i in range(n_points)
            ]
            mean_val = float(np.mean(values))
            new_values = [
                mean_val + (v - mean_val) * (1.0 - flatten) for v in values
            ]
            call(pitch_tier, "Remove points between", times[0] - 0.001, times[-1] + 0.001)
            for t, v in zip(times, new_values):
                if v > 0:
                    call(pitch_tier, "Add point", t, v)

        call([manipulation, pitch_tier], "Replace pitch tier")
        result_snd = call(manipulation, "Get resynthesis (overlap-add)")
        result = result_snd.values[0].astype(np.float32)
        # Trim or pad to match original length
        if len(result) > len(region):
            result = result[: len(region)]
        elif len(result) < len(region):
            result = np.pad(result, (0, len(region) - len(result)))
        return result, 0.0

    except Exception as exc:
        raise RuntimeError(
            f"Parselmouth pitch-tier flattening failed for monotone injection: {exc}"
        ) from exc


def _injector_low_energy(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Reduce amplitude by a fixed dB amount."""
    db_drop = INJECT_ENERGY_DB_DROP[severity]
    scale = 10.0 ** (-db_drop / 20.0)
    return (region * scale).astype(np.float32), 0.0


def _injector_filler(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Insert a synthetic filler buzz before the region.

    [SYNTHETIC PLACEHOLDER] — data/raw/fillers/ not yet populated.
    Generates a soft 150 Hz sine (marked as synthetic filler).
    """
    dur = INJECT_FILLER_DURATIONS[severity]
    n_samp = _samples(dur)
    t = np.linspace(0.0, dur, n_samp, endpoint=False, dtype=np.float32)
    buzz = (INJECT_FILLER_AMPLITUDE * np.sin(2 * np.pi * INJECT_FILLER_FREQ_HZ * t)).astype(
        np.float32
    )
    new_region = np.concatenate([buzz, region])
    return new_region, dur


def _injector_unclear(
    pre: np.ndarray, region: np.ndarray, severity: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Add Gaussian noise to the region at a target SNR."""
    target_snr = INJECT_NOISE_SNR[severity]
    signal_rms = float(np.sqrt(np.mean(region.astype(np.float64) ** 2)))
    if signal_rms < 1e-9:
        return region.copy(), 0.0
    noise_rms = signal_rms / (10.0 ** (target_snr / 20.0))
    noise = rng.standard_normal(len(region)).astype(np.float32) * float(noise_rms)
    return np.clip(region + noise, -1.0, 1.0).astype(np.float32), 0.0


# ---------------------------------------------------------------------------
# Injector registry
# ---------------------------------------------------------------------------

INJECTOR_MAP: dict[str, Callable] = {
    "PACE_TOO_SLOW": _injector_slow,
    "PACE_TOO_FAST": _injector_fast,
    "PAUSE_EXCESSIVE": _injector_pause,
    "PITCH_MONOTONE": _injector_monotone,
    "ENERGY_LOW": _injector_low_energy,
    "FILLER": _injector_filler,
    "UNCLEAR": _injector_unclear,
}


# ---------------------------------------------------------------------------
# High-level: generate one flawed file
# ---------------------------------------------------------------------------

def generate_flawed_file(
    src_path: Path,
    words: list[Word],
    flaw_type: str,
    severity: int,
    rng: np.random.Generator,
    out_path: Path,
) -> tuple[float, float]:
    """Inject a single flaw into a source audio file and write the result.

    Args:
        src_path:   Path to the source WAV file.
        words:      Word timing list from forced alignment / proportional.
        flaw_type:  Key in INJECTOR_MAP (e.g. \"PACE_TOO_SLOW\").
        severity:   Integer 1–5.
        rng:        Seeded numpy Generator (caller owns the seed).
        out_path:   Where to write the flawed WAV file.

    Returns:
        (label_start_s, label_end_s) in the new audio timeline.
    """
    waveform, _ = librosa.load(str(src_path), sr=SAMPLE_RATE, mono=True)
    waveform = waveform.astype(np.float32)

    w_start_idx, w_end_idx = pick_region(words, rng)
    region_start_s = words[w_start_idx].start
    region_end_s = words[w_end_idx].end

    injector_fn = INJECTOR_MAP[flaw_type]
    new_waveform, label_start_s, label_end_s = inject(
        waveform, region_start_s, region_end_s, injector_fn, severity, rng
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), new_waveform, SAMPLE_RATE, subtype="PCM_16")
    return label_start_s, label_end_s
