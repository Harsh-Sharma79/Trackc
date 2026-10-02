"""
tests/test_features_forced_align.py
===================================
Tests for forced alignment (WhisperX, Torchaudio MMS_FA, proportional fallback).
Includes cache verification, interpolation for unaligned words, and skipped-if-no-data
tests for real audio speeches.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import ALIGNMENT_CACHE_DIR, RAW_DIR, SAMPLE_RATE
from src.features.forced_align import (
    align_transcript,
    compute_alignment_hash,
    interpolate_unaligned_words,
)


def _make_dummy_wav(duration_s: float = 3.0) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    sf.write(tmp.name, sig, SAMPLE_RATE)
    return Path(tmp.name)


def test_compute_alignment_hash_deterministic():
    wav = _make_dummy_wav(1.0)
    text = "hello world"
    h1 = compute_alignment_hash(wav, text)
    h2 = compute_alignment_hash(wav, text)
    assert h1 == h2
    assert len(h1) == 24


def test_interpolate_unaligned_words():
    words = ["the", "quick", "brown", "fox", "jumps"]
    # Only "the" (idx 0) and "fox" (idx 3) were aligned by the model
    raw = [
        {"word": "the", "start": 0.1, "end": 0.4},
        {"word": "fox", "start": 1.5, "end": 1.8},
    ]
    interpolated = interpolate_unaligned_words(words, raw, duration=2.5)

    assert len(interpolated) == len(words)
    # Check aligned flags
    assert interpolated[0]["aligned"] is True
    assert interpolated[0]["word"] == "the"
    assert interpolated[1]["aligned"] is False
    assert interpolated[2]["aligned"] is False
    assert interpolated[3]["aligned"] is True
    assert interpolated[3]["word"] == "fox"
    assert interpolated[4]["aligned"] is False

    # Check monotonicity
    for i in range(len(interpolated) - 1):
        assert interpolated[i]["end"] <= interpolated[i + 1]["start"] + 1e-4
        assert interpolated[i]["end"] > interpolated[i]["start"]


def test_proportional_fallback_and_caching():
    wav = _make_dummy_wav(2.0)
    transcript = "testing forced alignment caching and fallback"
    # Force proportional backend for unit testing without GPU/model download
    res1 = align_transcript(wav, transcript, backend="proportional", use_cache=True)
    assert len(res1) == len(transcript.split())
    for item in res1:
        assert item["end"] > item["start"]
        assert item["aligned"] is False

    # Verify cache file exists
    f_hash = compute_alignment_hash(wav, transcript)
    cache_file = ALIGNMENT_CACHE_DIR / f"{f_hash}.json"
    assert cache_file.exists()

    # Second call reads from cache
    res2 = align_transcript(wav, transcript, backend="proportional", use_cache=True)
    assert res1 == res2


def test_real_speech_alignment_skipped_if_no_data():
    """Real forced alignment on true human speech can only be verified when real speech files exist.

    If no real speech WAVs exist in data/raw/, this test is cleanly skipped.
    """
    real_wavs = list(RAW_DIR.glob("*.wav"))
    if not real_wavs:
        pytest.skip(
            "Unverified: Real speech forced alignment is unverified because no real speech files exist in data/raw/. "
            "Verification will run automatically once speeches are added."
        )

    # If real speech is present, run alignment
    sample_wav = real_wavs[0]
    txt_file = sample_wav.with_suffix(".txt")
    if not txt_file.exists():
        pytest.skip(f"No corresponding transcript file found for {sample_wav}")

    transcript = txt_file.read_text(encoding="utf-8")
    aligned = align_transcript(sample_wav, transcript, backend="whisperx")
    assert len(aligned) > 0
    assert any(w["aligned"] is True for w in aligned)


def test_torchaudio_failure_raises(monkeypatch):
    """Bug 3: Explicitly selected backend that fails must RAISE, not silently fall back."""
    import src.features.forced_align as fa

    wav = _make_dummy_wav(1.5)
    text = "hello world test failure"

    def mock_fail(*args, **kwargs):
        raise RuntimeError("Torchaudio MMS_FA forced alignment internal crash")

    monkeypatch.setattr(fa, "_align_with_torchaudio", mock_fail)

    with pytest.raises(RuntimeError, match="Torchaudio MMS_FA forced alignment internal crash"):
        fa.align_transcript(wav, text, backend="torchaudio", use_cache=False)

    # Verify no cache was written for the failed aligner
    f_hash = fa.compute_alignment_hash(wav, text)
    cache_file = ALIGNMENT_CACHE_DIR / f"{f_hash}.json"
    assert not cache_file.exists()


def test_proportional_only_when_selected():
    """Bug 3: Proportional timing is used only when ALIGNMENT_BACKEND == 'proportional' or backend='proportional'."""
    from src.features.transcript import tokenize
    wav = _make_dummy_wav(1.0)
    text = "proportional only when selected"

    # Explicit backend="proportional" succeeds and marks all words aligned=False
    res = align_transcript(wav, text, backend="proportional", use_cache=False)
    assert len(res) == len(tokenize(text))
    assert all(w["aligned"] is False for w in res)


def test_no_silent_fallback_whisperx(monkeypatch):
    """Bug 3: Selecting 'whisperx' when not installed raises ImportError, no silent fallback."""
    import builtins
    wav = _make_dummy_wav(1.0)
    text = "testing whisperx import error"

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "whisperx":
            raise ImportError("No module named 'whisperx'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="whisperx is not installed"):
        align_transcript(wav, text, backend="whisperx", use_cache=False)


def test_real_speech_alignment_uses_configured_backend():
    """Bug 4: Real speech alignment using the configured backend (torchaudio)."""
    from src.config import ALIGNMENT_BACKEND
    real_wavs = list(RAW_DIR.glob("*.wav"))
    if not real_wavs:
        pytest.skip(
            "Unverified: No real speech files in data/raw/. "
            "Verification will run automatically once speeches are added."
        )

    sample_wav = real_wavs[0]
    txt_file = sample_wav.with_suffix(".txt")
    if not txt_file.exists():
        pytest.skip(f"No corresponding transcript file found for {sample_wav}")

    transcript = txt_file.read_text(encoding="utf-8-sig")
    aligned = align_transcript(sample_wav, transcript, backend=ALIGNMENT_BACKEND)
    assert len(aligned) > 0
    assert any(w["aligned"] is True for w in aligned)


def test_real_speech_alignment_sanity_checks(monkeypatch):
    """ALIGNMENT SANITY CHECKS:
    For each data/raw/*.wav with a .txt:
    - word count out of aligner equals normalized transcript word count;
    - starts are non-decreasing;
    - each end >= start;
    - all times lie within audio duration;
    - at least 80% of words have aligned=True (otherwise fail and print unaligned words);
    - cache JSON is reused on a second call (assert no model call).
    - Reports number of unaligned words.
    """
    import soundfile as sf
    from src.features.transcript import tokenize
    import src.features.forced_align as fa

    real_wavs = list(RAW_DIR.glob("*.wav"))
    if not real_wavs:
        pytest.skip("No real speech files in data/raw/ to run alignment sanity checks.")

    for wav_path in real_wavs:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue

        transcript = txt_path.read_text(encoding="utf-8-sig")
        expected_tokens = tokenize(transcript)
        data, sr = sf.read(str(wav_path))
        duration = len(data) / sr

        # Call 1: Run alignment
        aligned = fa.align_transcript(wav_path, transcript, duration=duration, use_cache=True)

        # 1. Word count equals normalized transcript word count
        assert len(aligned) == len(expected_tokens), (
            f"Word count mismatch: aligner got {len(aligned)}, expected {len(expected_tokens)}"
        )

        # 2 & 3 & 4. Timestamps sanity
        for i, w in enumerate(aligned):
            assert w["end"] >= w["start"], f"Word {w['word']}: end {w['end']} < start {w['start']}"
            assert 0.0 <= w["start"] <= duration + 0.1, f"Word {w['word']}: start {w['start']} outside [0, {duration}]"
            assert 0.0 <= w["end"] <= duration + 0.5, f"Word {w['word']}: end {w['end']} outside [0, {duration}]"
            if i > 0:
                assert w["start"] >= aligned[i - 1]["start"] - 1e-4, (
                    f"Start non-decreasing violation at index {i}: {w['start']} < {aligned[i - 1]['start']}"
                )

        # 5. At least 80% aligned
        unaligned_words = [w["word"] for w in aligned if not w["aligned"]]
        pct_aligned = (len(aligned) - len(unaligned_words)) / max(len(aligned), 1)
        assert pct_aligned >= 0.80, (
            f"Only {pct_aligned:.1%} of words aligned (< 80%). Unaligned words: {unaligned_words}"
        )
        print(f"\n[Sanity Check] {wav_path.name}: {len(aligned)} words total, {len(unaligned_words)} unaligned ({pct_aligned:.1%} aligned).")

        # 6. Cache reuse: monkeypatch model to raise if called on second invocation
        def boom(*args, **kwargs):
            raise AssertionError("Model should NOT be called when cache exists!")

        monkeypatch.setattr(fa, "_align_with_torchaudio", boom)
        cached_result = fa.align_transcript(wav_path, transcript, duration=duration, use_cache=True)
        assert cached_result == aligned

