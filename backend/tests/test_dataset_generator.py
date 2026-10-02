"""
tests/test_dataset_generator.py
================================
Tests for src/dataset/generator.py.

Uses a synthetic 2-second WAV written to a temp directory — no real data needed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import (
    INJECT_FILLER_DURATIONS,
    INJECT_PAUSE_DURATIONS,
    INJECT_SLOW_RATES,
    SAMPLE_RATE,
    SEED,
)
from src.dataset.generator import (
    INJECTOR_MAP,
    generate_flawed_file,
    inject,
    pick_region,
    _injector_fast,
    _injector_filler,
    _injector_low_energy,
    _injector_monotone,
    _injector_pause,
    _injector_slow,
    _injector_unclear,
)
from src.features.transcript import Word


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sine_wav(tmp_dir: Path, duration_s: float = 2.0, freq: float = 220.0) -> Path:
    """Write a synthetic sine WAV and return its path."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    path = tmp_dir / "test.wav"
    sf.write(str(path), wav, SAMPLE_RATE, subtype="PCM_16")
    return path


def _words() -> list[Word]:
    """Eight evenly-spaced fake words across 2 seconds."""
    return [Word(f"w{i}", i * 0.25, (i + 1) * 0.25) for i in range(8)]


def _region() -> np.ndarray:
    """0.5 s of 220 Hz sine as a fake region."""
    n = int(0.5 * SAMPLE_RATE)
    t = np.linspace(0, 0.5, n, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence_region() -> np.ndarray:
    return np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)


# ---------------------------------------------------------------------------
# pick_region
# ---------------------------------------------------------------------------

def test_pick_region_bounds():
    """Returned indices must be within the words list."""
    words = _words()
    rng = _rng()
    for _ in range(20):
        s, e = pick_region(words, rng)
        assert 0 <= s <= e < len(words)


def test_pick_region_min_length():
    """Region must span at least INJECT_REGION_MIN_WORDS words."""
    from src.config import INJECT_REGION_MIN_WORDS
    words = _words()
    rng = _rng()
    for _ in range(20):
        s, e = pick_region(words, rng)
        assert (e - s + 1) >= INJECT_REGION_MIN_WORDS


def test_pick_region_deterministic():
    """Same seed produces same region twice."""
    words = _words()
    r1 = pick_region(words, _rng(42))
    r2 = pick_region(words, _rng(42))
    assert r1 == r2


def test_pick_region_too_few_words():
    """Fewer words than min raises ValueError."""
    with pytest.raises(ValueError):
        pick_region([Word("only", 0.0, 0.5)], _rng())


# ---------------------------------------------------------------------------
# Individual injectors — duration shifts
# ---------------------------------------------------------------------------

def test_slow_increases_duration():
    """_injector_slow must return a longer region for severity 1."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    new_region, delta_s = _injector_slow(pre, region, severity=1, rng=_rng())
    assert len(new_region) > len(region), "Slow injection must lengthen the region"
    assert delta_s > 0.0


def test_fast_decreases_duration():
    """_injector_fast must return a shorter region for severity 1."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    new_region, delta_s = _injector_fast(pre, region, severity=1, rng=_rng())
    assert len(new_region) < len(region), "Fast injection must shorten the region"
    assert delta_s < 0.0


def test_pause_increases_duration():
    """_injector_pause must prepend silence; delta == INJECT_PAUSE_DURATIONS[sev]."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    sev = 2
    new_region, delta_s = _injector_pause(pre, region, severity=sev, rng=_rng())
    assert len(new_region) > len(region)
    expected_delta = INJECT_PAUSE_DURATIONS[sev]
    assert abs(delta_s - expected_delta) < 1e-3


def test_filler_increases_duration():
    """_injector_filler must prepend buzz; delta == INJECT_FILLER_DURATIONS[sev]."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    sev = 1
    new_region, delta_s = _injector_filler(pre, region, severity=sev, rng=_rng())
    assert len(new_region) > len(region)
    expected_delta = INJECT_FILLER_DURATIONS[sev]
    assert abs(delta_s - expected_delta) < 1e-3


def test_low_energy_reduces_amplitude():
    """_injector_low_energy must reduce RMS; delta == 0."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    new_region, delta_s = _injector_low_energy(pre, region, severity=3, rng=_rng())
    assert float(np.sqrt(np.mean(new_region ** 2))) < float(
        np.sqrt(np.mean(region ** 2))
    )
    assert delta_s == 0.0


def test_unclear_adds_noise():
    """_injector_unclear should change the signal (not identical to source)."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    new_region, delta_s = _injector_unclear(pre, region, severity=3, rng=_rng())
    assert not np.allclose(new_region, region)
    assert delta_s == 0.0


def test_monotone_no_duration_change():
    """_injector_monotone must return delta_s == 0."""
    region = _region()
    pre = np.zeros(100, dtype=np.float32)
    _, delta_s = _injector_monotone(pre, region, severity=2, rng=_rng())
    assert delta_s == 0.0


# ---------------------------------------------------------------------------
# inject() wrapper
# ---------------------------------------------------------------------------

def test_inject_label_start_equals_region_start():
    """label_start_s must equal the region start passed in."""
    n = int(2.0 * SAMPLE_RATE)
    waveform = np.zeros(n, dtype=np.float32)
    region_start = 0.5
    region_end = 1.0
    new_waveform, label_start, label_end = inject(
        waveform, region_start, region_end, _injector_pause, severity=1, rng=_rng()
    )
    assert abs(label_start - region_start) < 1e-6


def test_inject_label_end_shifted():
    """After pause injection label_end must be region_end + pause_duration."""
    n = int(2.0 * SAMPLE_RATE)
    waveform = np.zeros(n, dtype=np.float32)
    sev = 1
    _, _, label_end = inject(
        waveform, 0.5, 1.0, _injector_pause, severity=sev, rng=_rng()
    )
    expected = 1.0 + INJECT_PAUSE_DURATIONS[sev]
    assert abs(label_end - expected) < 1e-3


def test_inject_total_duration_consistent():
    """Total output duration = original duration + delta from injector."""
    n = int(2.0 * SAMPLE_RATE)
    orig_dur = 2.0
    waveform = np.zeros(n, dtype=np.float32)
    sev = 2
    new_waveform, label_start, label_end = inject(
        waveform, 0.5, 1.0, _injector_pause, severity=sev, rng=_rng()
    )
    expected_n = n + int(round(INJECT_PAUSE_DURATIONS[sev] * SAMPLE_RATE))
    # Allow ±1 sample rounding
    assert abs(len(new_waveform) - expected_n) <= 2


# ---------------------------------------------------------------------------
# generate_flawed_file — end-to-end
# ---------------------------------------------------------------------------

def test_generate_flawed_file_writes_wav(tmp_path):
    """generate_flawed_file should write a WAV and return finite timestamps."""
    wav_path = _sine_wav(tmp_path)
    words = _words()
    out_path = tmp_path / "out.wav"
    start_s, end_s = generate_flawed_file(
        wav_path, words, "PAUSE_EXCESSIVE", severity=1, rng=_rng(), out_path=out_path
    )
    assert out_path.exists()
    assert end_s > start_s
    assert start_s >= 0.0


def test_generate_flawed_file_label_inside_file(tmp_path):
    """Label end_sec must not exceed the new file's duration."""
    wav_path = _sine_wav(tmp_path, duration_s=3.0)
    words = [Word(f"w{i}", i * 0.375, (i + 1) * 0.375) for i in range(8)]
    out_path = tmp_path / "out.wav"
    start_s, end_s = generate_flawed_file(
        wav_path, words, "PACE_TOO_SLOW", severity=1, rng=_rng(), out_path=out_path
    )
    # Read the output and check duration
    data, sr = sf.read(str(out_path))
    file_dur = len(data) / sr
    assert end_s <= file_dur + 0.1  # small tolerance for rounding


def test_generate_flawed_file_deterministic(tmp_path):
    """Same seed must produce identical output bytes."""
    wav_path = _sine_wav(tmp_path)
    words = _words()
    out1 = tmp_path / "out1.wav"
    out2 = tmp_path / "out2.wav"
    generate_flawed_file(wav_path, words, "ENERGY_LOW", severity=2, rng=_rng(42), out_path=out1)
    generate_flawed_file(wav_path, words, "ENERGY_LOW", severity=2, rng=_rng(42), out_path=out2)
    assert out1.read_bytes() == out2.read_bytes()
