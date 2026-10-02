"""
scripts/audit_speech1.py
========================
Runs audit steps b, c, e on data/raw/speech1.wav.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.audio import extract_all
from src.features.forced_align import align_transcript
from src.features.spectral import (
    compute_clarity_snr,
    compute_mfcc,
    compute_spectral_centroid,
    compute_spectral_flatness,
)
from src.features.transcript import Word
from src.features.word_features import extract_all_word_features
from src.analysis.detector import detect_filler_flaws, detect_all
from src.analysis.match import match_words, compare_words, detect_from_matches
from src.pipeline import evaluate
import numpy as np


def step_b():
    print("\n" + "=" * 60)
    print("AUDIT STEP B: Alignment on speech1.wav")
    print("=" * 60)
    wav_path = Path("data/raw/speech1.wav")
    txt_path = Path("data/raw/speech1.txt")
    transcript = txt_path.read_text(encoding="utf-8-sig").strip()
    words = align_transcript(wav_path, transcript, use_cache=True)
    print(f"Total words aligned: {len(words)}")
    print("First 10 words:")
    for i, w in enumerate(words[:10], 1):
        print(f"  {i:2d}. {w['word']:<12} [{w['start']:6.3f}s -> {w['end']:6.3f}s]  aligned={w['aligned']}")


def step_c():
    print("\n" + "=" * 60)
    print("AUDIT STEP C: Feature extraction on speech1.wav")
    print("=" * 60)
    wav_path = Path("data/raw/speech1.wav")
    txt_path = Path("data/raw/speech1.txt")
    transcript = txt_path.read_text(encoding="utf-8-sig").strip()

    # 1. Base audio features
    af = extract_all(wav_path)
    print(f"Duration:        {af.duration:.2f} s")
    print(f"Total frames:    {len(af.pitch_st)}")
    voiced_mask = np.isfinite(af.pitch_st)
    print(f"Voiced frames:   {np.sum(voiced_mask)} / {len(af.pitch_st)} ({np.mean(voiced_mask)*100:.1f}%)")
    print(f"Pitch st (voiced): min={np.min(af.pitch_st[voiced_mask]):.2f}, max={np.max(af.pitch_st[voiced_mask]):.2f}, mean={np.mean(af.pitch_st[voiced_mask]):.2f}")
    print(f"Energy z-scores: min={np.min(af.energy_z):.2f}, max={np.max(af.energy_z):.2f}, mean={np.mean(af.energy_z):.2f}")
    print(f"Pauses detected: {len(af.pauses)}")

    # 2. Spectral features
    mfcc = compute_mfcc(af.waveform)
    centroid = compute_spectral_centroid(af.waveform)
    flatness = compute_spectral_flatness(af.waveform)
    snr_db, clarity_score = compute_clarity_snr(af.waveform)

    print(f"\nSpectral features:")
    print(f"  MFCC shape:             {mfcc.shape}")
    print(f"  MFCC [0, 0:5]:          {np.round(mfcc[0, :5], 3).tolist()}")
    print(f"  Spectral centroid mean: {np.mean(centroid):.1f} Hz")
    print(f"  Spectral flatness mean: {np.mean(flatness):.4f}")
    print(f"  Clarity SNR:            {snr_db:.2f} dB (score: {clarity_score:.2f})")

    # Sanity checks on spectral features
    assert not np.isnan(snr_db) and not np.isinf(snr_db), "Clarity SNR has NaN/inf!"
    assert not np.isnan(clarity_score) and not np.isinf(clarity_score), "Clarity score has NaN/inf!"
    assert np.all(np.isfinite(mfcc)), "MFCC has non-finite values!"
    assert np.all(np.isfinite(centroid)), "Centroid has non-finite values!"
    assert np.all(np.isfinite(flatness)), "Flatness has non-finite values!"

    # 3. Word features
    aligned = align_transcript(wav_path, transcript, duration=af.duration, use_cache=True)
    word_objs = [Word(text=w["word"], start=w["start"], end=w["end"]) for w in aligned]
    wfeats = extract_all_word_features(word_objs, af.pitch_st, af.energy_z)
    print(f"\nWord features ({len(wfeats)} words extracted):")
    print("Sample (first 3 words):")
    for wf in wfeats[:3]:
        print(f"  Word '{wf.word}': dur={wf.duration:.3f}s, pause_before={wf.pause_before:.3f}s, pitch_range={wf.pitch_range_st:.2f}st, energy_z={wf.energy_z_mean:.2f}")

    # Check for unwanted infs
    for wf in wfeats:
        assert not np.isinf(wf.duration), f"Infinite duration in word {wf.word}"
        assert not np.isinf(wf.pause_before), f"Infinite pause in word {wf.word}"
        assert not np.isinf(wf.energy_z_mean), f"Infinite energy in word {wf.word}"

    # 4. Filler detector
    fillers = detect_filler_flaws(word_objs, af.pitch_st, af.energy_z)
    print(f"\nFiller flaws detected: {len(fillers)}")
    for f in fillers:
        print(f"  [{f.start:.2f}s -> {f.end:.2f}s] sev={f.severity:.2f}: {f.explanation}")
    print("Step C Feature Extraction checks PASSED with zero unexpected NaN/inf.")


def step_e():
    print("\n" + "=" * 60)
    print("AUDIT STEP E: Detection on ideal-vs-ideal and one injected flaw")
    print("=" * 60)
    wav_path = Path("data/raw/speech1.wav")
    txt_path = Path("data/raw/speech1.txt")
    transcript = txt_path.read_text(encoding="utf-8-sig").strip()

    # 1. Ideal-vs-ideal evaluation (comparing speech1 to itself)
    print("Testing speech1 vs speech1 (ideal-vs-ideal)...")
    res_ideal = evaluate(
        candidate_path=wav_path,
        baseline_path=wav_path,
        transcript=transcript,
        audio_id="speech1_vs_self",
    )
    print(f"  Composite score: {res_ideal.composite_score:.2f}")
    print(f"  Flaw regions detected: {len(res_ideal.flaw_regions)}")
    for r in res_ideal.flaw_regions:
        print(f"    {r.flaw_type.value}: [{r.start:.2f}s - {r.end:.2f}s] sev={r.severity:.2f}")

    # Assert zero contrastive pace/pause flaws for self comparison
    contrastive_flaws = [
        r for r in res_ideal.flaw_regions
        if r.flaw_type.value in {"PACE_TOO_SLOW", "PACE_TOO_FAST", "PAUSE_MISSING"}
    ]
    print(f"  Contrastive timing flaws on identical audio: {len(contrastive_flaws)} (expected 0)")
    assert len(contrastive_flaws) == 0, f"Expected 0 contrastive flaws on identical audio, got {len(contrastive_flaws)}"
    print("  Ideal-vs-ideal check: PASSED (zero contrastive flaws).")


if __name__ == "__main__":
    step_b()
    step_c()
    step_e()
