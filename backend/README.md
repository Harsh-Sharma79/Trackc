# Speech Evaluation Backend — Track C · Multimodal AI Hackathon 2026 (IIT Mandi)

> **Track C — Contrastive Speech Analytics with Temporal Flaw Grounding**
> Backend only. Frontend / dashboard built by teammate.

Given a participant's audio file and an "ideal" baseline recording of the same text, this system:
- force-aligns words to timestamps (torchaudio MMS_FA),
- extracts acoustic features (F0, RMS energy, MFCCs, spectral centroid/flatness, speech rate, pauses),
- scores four rubric dimensions baseline-relatively (`score = exp(-k × deviation)`, ideal-vs-ideal = 1.0),
- detects time-stamped flaw regions with severity scores and causal text explanations,
- outputs a self-contained JSON object per evaluation.

---

## Team

| Role | Name |
|------|------|
| Backend | **[YOUR FULL NAME]** |
| Frontend / Dashboard | **[TEAMMATE FULL NAME]** |

> ⚠️ Both names must appear on the Devpost submission exactly as they appear on your college ID.

---

## Environment Setup

The venv lives on the D drive (C was nearly full). **Always call Python directly — do not use `python` or `py` from PATH.**

```powershell
# Open a new PowerShell window and run this block every time:
cd C:\Users\HP\.gemini\antigravity\scratch\speech_eval
$py = "D:\speech\venv\Scripts\python.exe"
$env:TEMP        = "D:\speech\temp"
$env:TMP         = "D:\speech\temp"
$env:PIP_CACHE_DIR = "D:\speech\pip_cache"
$env:HF_HOME     = "D:\speech\models\hf"
$env:TORCH_HOME  = "D:\speech\models\torch"
$env:PYTHONPATH  = $PWD.Path
```

**Python 3.11.9** · torch + torchaudio **2.8.0+cpu** (do **not** upgrade — `forced_align` is removed in 2.9)

Key packages: `numpy 2.4.6`, `praat-parselmouth 0.4.7`, `librosa`, `soundfile`, `scipy`, `pydantic`, `pytest`

> The first alignment run downloads ~1.5 GB of MMS_FA models into `$env:HF_HOME`. Subsequent runs use the cache.

---

## Quickstart

```powershell
# 1. Test (237 pass, 1 expected fail — whisperx/ffmpeg test, see Known Issues)
& $py -m pytest tests/ -q

# 2. Evaluate a single file
& $py -c "
from src.pipeline import evaluate
import json
res = evaluate(
    candidate_path='data/flawed/speech1_pause_excessive_sev5.wav',
    baseline_path='data/raw/speech1.wav',
    transcript='data/raw/speech1.txt',
    audio_id='demo_run',
)
print(json.dumps(res.model_dump(), indent=2, ensure_ascii=False))
"

# 3. Generate the full synthetic dataset (then validate labels)
& $py -m src.dataset.generate
& $py scripts/validate_labels.py

# 4. Evaluate all flawed files and print precision / recall
& $py scripts/evaluate_detections.py

# 5. End-to-end pipeline (skips the slow test suite)
& $py scripts/run_all.py --skip-tests
```

---

## API — Entry Point

```python
from src.pipeline import evaluate

result = evaluate(
    candidate_path="data/flawed/speaker1.wav",   # participant audio
    baseline_path="data/raw/baseline.wav",        # ideal reference (same text)
    transcript="data/raw/speech1.txt",            # path or raw string
    audio_id="speaker1_run1",                     # arbitrary unique ID
)
# result is an EvaluationResult (Pydantic v2 model)
print(result.model_dump_json(indent=2))
```

### `EvaluationResult` schema (see [`src/schema.py`](src/schema.py))

| Field | Type | Description |
|-------|------|-------------|
| `audio_id` | `str` | Identifier passed in |
| `transcript` | `str` | Text used for alignment |
| `duration` | `float` | Candidate audio duration (s) |
| `composite_score` | `float` | Weighted mean of rubric scores, 0–1 |
| `rubric_scores` | `list[RubricScore]` | One entry per dimension |
| `flaw_regions` | `list[FlawRegion]` | Time-stamped detected flaws |
| `summary` | `str` | Plain-English one-paragraph summary |

### `FlawRegion`

```json
{
  "start": 14.5,
  "end": 17.89,
  "flaw_type": "PAUSE_EXCESSIVE",
  "severity": 0.997,
  "explanation": "A 3.39 s pause (2.43 s longer than the reference) was detected at 14.50–17.89 s.",
  "metadata": {"pause_duration_s": 3.392, "baseline_pause_duration_s": 0.96, "extra_pause_s": 2.432}
}
```

> **Important for teammate:** `flaw_type` strings are **UPPER_CASE** (e.g. `"PAUSE_EXCESSIVE"`, `"PITCH_MONOTONE"`). NaN pitch is serialised as `null`. All JSON is written UTF-8; detail strings use plain ASCII `"-"` not `"—"`.

### Flaw taxonomy

| `flaw_type` value | Meaning |
|---|---|
| `PACE_TOO_SLOW` | Words stretched >1.30× baseline duration |
| `PACE_TOO_FAST` | Words compressed <0.75× baseline duration |
| `PAUSE_EXCESSIVE` | Inter-word or absorbed silence >0.80 s above baseline |
| `PAUSE_MISSING` | Expected pause omitted |
| `PITCH_MONOTONE` | Flat pitch over a voiced run |
| `ENERGY_LOW` | Sustained drop in RMS energy vs baseline |
| `FILLER` | Um/uh detected (classifier placeholder) |
| `UNCLEAR` | Low clarity SNR region |

---

## Scoring Model

When a baseline is provided, **all four dimensions are baseline-relative** — the score of an ideal read compared with itself is exactly **1.0**.

```
deviation = 0.5 × global_mean + 0.5 × worst_8_word_window_mean
score     = exp(-k × deviation)
```

Blending the file-wide mean with the worst 8-word sliding window ensures a 3–6 word injected flaw is not diluted by 100 ideal words around it.

| Dimension | What it measures | k |
|-----------|-----------------|---|
| `pace` | `mean |log(word_duration_ratio)|` | 6.0 |
| `pause_pattern` | Extra pause time as fraction of duration | 8.0 |
| `pitch_variation` | `|log(pitch_range_ratio)|` per voiced word | 3.0 |
| `energy_consistency` | `mean |energy_z_delta|` per word | 2.0 |

Without a baseline, the original absolute ideal-range scorers are used (no change to that code path).

---

## Dataset

| Path | Contents |
|------|----------|
| `data/raw/` | Source recordings (`*.wav` + `*.txt`) |
| `data/flawed/` | 45 synthetic flaw files |
| `data/labels/injections.csv` | 60 ground-truth rows |
| `data/labels/alignments/` | Cached forced-alignment JSONs |
| `outputs/` | `*_eval.json` per evaluated file |

**injections.csv columns:** `file, source, flaw_type, severity, start_sec, end_sec`

> 📦 Dataset download (wav + labels): **[INSERT GOOGLE DRIVE LINK]**  
> Source speech: **[INSERT RECORDING TITLE AND LICENSE]**

### Flaw injection — how it works

The generator (`src/dataset/generator.py`) injects flaws into a region of the aligned source audio:

| Flaw | Mechanism | Severity range |
|------|-----------|---------------|
| Slow / Fast pace | Time-stretch via `librosa.effects.time_stretch` | ×1.1 – ×2.0 / ×0.5 – ×0.9 |
| Pause excessive | Insert silence samples | 0.5 s – 2.5 s |
| Pitch monotone | Flatten pitch with PSOLA (parselmouth) | 40% – 100% flattened |
| Energy low | Scale amplitude | 25% – 5% of original |
| Filler | Inject 150 Hz buzz | 0.20 s – 1.00 s |
| Unclear | Low-pass filter | 5 kHz – 1 kHz cutoff |

All injection positions are deterministic (seed = 42). The label CSV is written from the same timeline, so start/end times are exact by construction.

---

## Detection Results

Evaluated on 45 synthetic files (one source speaker), IoU threshold ≥ 0.3.

| Flaw type | TP | FP | FN | Precision | Recall |
|-----------|----|----|-----|-----------|--------|
| ENERGY_LOW | 8 | 5 | 3 | **0.62** | **0.73** |
| PACE_TOO_FAST | 3 | 0 | 3 | **1.00** | 0.50 |
| PACE_TOO_SLOW | 3 | 2 | 5 | 0.60 | 0.38 |
| PAUSE_EXCESSIVE | 4 | 5 | 6 | 0.44 | 0.40 |
| FILLER | 0 | 0 | 9 | — | **0.00** |
| PITCH_MONOTONE | 0 | 0 | 8 | — | **0.00** |
| UNCLEAR | 0 | 0 | 8 | — | **0.00** |
| **Overall** | **18** | **12** | **42** | **0.60** | **0.30** |

> **F1 = 0.40** overall. Pace and energy detection work. Filler, monotone and unclear are not yet detected as regions — documented as limitations, not silently ignored.

### Sanity checks that pass

- Ideal-vs-ideal: **0 regions, composite 1.0**
- `speech1_pace_too_slow`: detected region 3.04–4.20 s vs label 3.042–4.243 s ✓
- `speech1_pause_excessive_sev5`: detected 14.50–17.89 s vs label 14.49–19.59 s ✓
- Monotone injection verified: pitch std drops from 1.59 → 0.05 semitones
- Scores fall monotonically with severity for pause and slow pace ✓

---

## Known Issues & Limitations

See [`docs/limitations.md`](docs/limitations.md) for the full table. Short version:

| Issue | Status |
|-------|--------|
| FILLER / PITCH_MONOTONE / UNCLEAR: 0 recall | Known; classifiers not implemented — documented |
| Short pauses (<0.8 s absorbed silence) missed | Threshold deliberately above natural variation |
| Pitch score contaminated by pause/pace flaws | DTW warping spreads the inserted silence across pitch frames |
| Energy region sometimes split into two | Merge gap parameter needs tuning |
| One test fails: `test_real_speech_alignment_skipped_if_no_data` | Hardcodes `backend="whisperx"`, ffmpeg not installed — **do not edit** |
| Single source speaker, synthetic flaws | Results may not generalise to other speakers or real delivery errors |

---

## Repository Map

```
speech_eval/
├── src/
│   ├── config.py               All constants, thresholds, seeds (seed=42)
│   ├── schema.py               Pydantic v2 output schemas
│   ├── pipeline.py             Evaluation orchestrator (evaluate() entry point)
│   ├── features/
│   │   ├── audio.py            pYIN F0 (NaN=unvoiced), RMS energy z-score, pause detection
│   │   ├── forced_align.py     torchaudio MMS_FA forced alignment, cache-first
│   │   ├── transcript.py       Tokenisation, proportional timing fallback
│   │   ├── spectral.py         13 MFCCs, spectral centroid, flatness, clarity SNR
│   │   ├── word_features.py    Per-word pitch/energy/duration/pause metrics
│   │   └── alignment.py        Vectorised Sakoe-Chiba DTW feature alignment
│   ├── analysis/
│   │   ├── scoring.py          Baseline-relative exp-decay rubric scoring
│   │   ├── detector.py         Single-stream flaw detectors (standalone/no-baseline mode)
│   │   └── match.py            difflib word matching + contrastive detection
│   ├── dataset/
│   │   ├── generator.py        Seeded flaw injection + label CSV writer
│   │   └── generate.py         CLI: python -m src.dataset.generate
│   └── explain/
│       └── templates.py        Deterministic causal explanation strings
├── scripts/
│   ├── run_all.py              End-to-end runner (--skip-tests flag available)
│   ├── validate_labels.py      Integrity checks on injections.csv
│   ├── evaluate_detections.py  Temporal IoU benchmark (train split)
│   └── tune_thresholds.py      Grid search over detection thresholds
├── tests/                      237 passing tests (pytest)
├── data/
│   ├── raw/                    Source audio + transcripts
│   ├── flawed/                 Injected flaw files (45 files)
│   └── labels/
│       ├── injections.csv      Ground-truth flaw annotations
│       └── alignments/         Cached MMS_FA alignment JSONs
├── outputs/                    Per-file evaluation JSON results
├── docs/
│   ├── DECISIONS.md            Full technical decision log (M1–M10)
│   ├── limitations.md          Honest limitations table
│   ├── features.md             Mathematical feature definitions
│   └── example_output.json     Clean example JSON for teammates and judges
└── AGENTS.md                   Rules for AI coding agents (committed)
```

---

## Reproducibility

```powershell
# Two runs must produce identical output hashes:
& $py scripts/run_all.py --skip-tests
Get-FileHash outputs\*.json | Sort-Object Hash | Format-Table -AutoSize
# Run again — hashes must be identical
```

All randomness is seeded at 42 (`src/config.py: SEED = 42`). Alignment results are cached in `data/labels/alignments/` so re-runs do not re-download or re-align.

---

## Devpost Checklist

- [ ] Project description written
- [ ] GitHub repo set to **public** (verify in a private browser window)
- [ ] Dataset uploaded to Google Drive, link-sharing enabled — add link above
- [ ] 3–10 minute YouTube video (unlisted is fine, **private is not**); English audio or subtitles
- [ ] All team members added to the submission with **real full names**
- [ ] 6-page technical report submitted
- [ ] `docs/example_output.json` committed (use `speech1_energy_low_sev5_eval.json`, not the pause file whose pitch score is contaminated)
- [ ] Source recording license verified and noted in README
- [ ] `docs/DECISIONS.md` M1/M2 corrected to say *torchaudio MMS_FA* (not whisperx)

---

## Teammate Handoff Notes

1. **Entry point:** `from src.pipeline import evaluate` — see API section above.
2. **`flaw_type` is always UPPER_CASE** — e.g. `"PAUSE_EXCESSIVE"`, `"PITCH_MONOTONE"`.
3. **Unvoiced pitch** is serialised as `null` in JSON (not `0.0`).
4. **Example output:** `docs/example_output.json` — copy from `outputs/speech1_energy_low_sev5_eval.json`.
5. **No baseline mode:** pass `baseline_path=None`; absolute ideal-range scorers activate instead.
6. The system writes all JSON with explicit UTF-8; detail strings use plain `-` not `—`.
