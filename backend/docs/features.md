# Feature Documentation

## Pitch (`src/features/audio.py`)

- **Algorithm:** pYIN (`librosa.pyin`), `fmin=60 Hz`, `fmax=600 Hz`.
- **Representation:** Semitones relative to the speaker's voiced-frame median F0.
  Formula: `12 * log2(f0 / median_f0_hz)`.
- **Unvoiced frames:** `NaN` (not zero). All downstream pitch stats use
  `np.isfinite` / `np.nanmean` / `np.nanstd` to ignore them.

---

## Energy (`src/features/audio.py`)

- **Algorithm:** RMS via `librosa.feature.rms` (centered framing, same hop as pitch).
- **Representation:** z-score.  Formula: `(rms - mean_rms) / std_rms`.
  If `std_rms < 1e-9` (constant signal), returns all zeros.
- **Frame count formula (centered framing):**
  `n_frames = 1 + floor(n_samples / HOP_LENGTH)`

---

## Spectral Features (`src/features/spectral.py`)

### MFCC
- 13 coefficients, `librosa.feature.mfcc`, centered framing.
- Shape: `(13, n_frames)`.

### Spectral Centroid
- Centre of mass of each STFT frame's magnitude spectrum (Hz).
- Shape: `(n_frames,)`.

### Spectral Flatness
- Wiener entropy: geometric mean / arithmetic mean of spectrum magnitudes.
- Values near **1** → noise-like; near **0** → tonal.
- Shape: `(n_frames,)`.

### Clarity (SNR-style, `compute_clarity_snr`)

```
RMS_dB[i] = 20 * log10(max(rms[i], 1e-9))
E_speech   = percentile(RMS_dB, CLARITY_HIGH_PERCENTILE=95)
E_noise    = percentile(RMS_dB, CLARITY_LOW_PERCENTILE=10)
SNR_dB     = max(0, E_speech - E_noise)
clarity    = clip(SNR_dB / CLARITY_IDEAL_SNR_DB, 0, 1)
```

**Interpretation:** A clean spoken sentence has SNR ≈ 20 dB → clarity ≈ 1.0.
Background noise or very quiet speech reduces clarity towards 0.

**Limits:**
- Assumes the recording is predominantly speech; fails on music or ambient noise.
- Sensitive to encoding artifacts (MP3 compression may inflate flatness-based noise estimates).

---

## Per-Word Features (`src/features/word_features.py`)

For each word (derived from forced-alignment or proportional timing):

| Field | Unit | Notes |
|---|---|---|
| `duration` | seconds | `end - start` |
| `pause_before` | seconds | gap since previous word end |
| `pause_after` | seconds | gap to next word start |
| `pitch_mean_st` | semitones | NaN if all frames unvoiced |
| `pitch_range_st` | semitones | `max - min` over voiced frames; NaN if < 2 voiced frames |
| `energy_z_mean` | z-score | mean over word's frame span |

Frame mapping: `frame = round(t * sr / hop_length)` (matches librosa centered framing).

---

## Filler Detector (`src/analysis/detector.py :: detect_filler_flaws`)

**Criterion:** A gap between consecutive words that satisfies ALL of:
1. `gap_duration >= FILLER_MIN_DURATION` (default 0.20 s)
2. Mean frame energy z-score in the gap < `FILLER_ENERGY_Z_MAX` (default 0.5)
3. Voiced pitch std in the gap < `FILLER_PITCH_STD_MAX` (default 1.2 semitones)

**Rationale:** Filler sounds (uh, um) are low-energy, voiced, and pitch-flat
relative to normal speech; conditions 2 and 3 jointly filter out both silence
(unvoiced) and expressive content.

**Limits:**
- Requires word-level timing from forced alignment; with proportional timing,
  gap boundaries are approximate.
- Does not transcribe the filler content; only detects probable filler-like
  acoustic activity in inter-word gaps.
- Short fillers (< 200 ms) are not detected.

---

## Alignment (`src/features/forced_align.py`)

- **Backend:** `whisperx` → `torchaudio MMS_FA` → proportional (fallback).
- **Cache:** JSON files keyed by SHA-256 of `(audio_bytes + transcript)` in
  `data/labels/alignments/`.
- **Unaligned words:** interpolated and flagged `aligned=False`.
