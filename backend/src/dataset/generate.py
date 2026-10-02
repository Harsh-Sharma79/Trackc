"""
src/dataset/generate.py
========================
CLI entry point for dataset generation.

Usage:
    python -m src.dataset.generate [--seed SEED] [--raw-dir PATH] [--out-dir PATH]

Reads every *.wav in RAW_DIR (with a matching .txt transcript).
For each source file + each flaw type + each severity level (1–5), produces one
flawed WAV in FLAWED_DIR and records a row in data/labels/injections.csv.
Also generates INJECT_COMBINED_COUNT combined-flaw files (multiple flaws per file).

Output columns in injections.csv:
    file, source, flaw_type, severity, start_sec, end_sec
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

from src.config import (
    FLAWED_DIR,
    INJECT_COMBINED_COUNT,
    LABELS_DIR,
    RAW_DIR,
    SAMPLE_RATE,
    SEED,
)
from src.dataset.generator import INJECTOR_MAP, generate_flawed_file
from src.features.transcript import tokenize, estimate_word_times

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INJECTIONS_CSV: Path = LABELS_DIR / "injections.csv"
CSV_HEADER = ["file", "source", "flaw_type", "severity", "start_sec", "end_sec"]
FLAW_TYPES = list(INJECTOR_MAP.keys())
SEVERITIES = [1, 2, 3, 4, 5]


def _load_transcript(wav_path: Path) -> str:
    """Read the .txt file beside the WAV (same stem)."""
    txt_path = wav_path.with_suffix(".txt")
    if not txt_path.exists():
        raise FileNotFoundError(f"Transcript not found: {txt_path}")
    return txt_path.read_text(encoding="utf-8-sig").strip()


def _words_for(wav_path: Path) -> list:
    """Get word timings via configured alignment backend for accurate region boundaries."""
    from src.features.forced_align import align_transcript
    from src.features.transcript import Word

    transcript = _load_transcript(wav_path)
    aligned = align_transcript(wav_path, transcript)
    return [Word(text=w["word"], start=w["start"], end=w["end"]) for w in aligned]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate flawed speech dataset.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=FLAWED_DIR)
    args = parser.parse_args(argv)

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(raw_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {raw_dir}. Add audio to generate the dataset.")
        return

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    # Single-flaw files
    for wav_path in wav_files:
        try:
            words = _words_for(wav_path)
        except Exception as exc:
            print(f"  SKIP {wav_path.name}: {exc}", file=sys.stderr)
            continue

        for flaw_type, severity in itertools.product(FLAW_TYPES, SEVERITIES):
            stem = wav_path.stem
            out_name = f"{stem}_{flaw_type.lower()}_sev{severity}.wav"
            out_path = out_dir / out_name
            try:
                start_s, end_s = generate_flawed_file(
                    wav_path, words, flaw_type, severity, rng, out_path
                )
                rows.append(
                    {
                        "file": out_name,
                        "source": wav_path.name,
                        "flaw_type": flaw_type,
                        "severity": severity,
                        "start_sec": round(start_s, 3),
                        "end_sec": round(end_s, 3),
                    }
                )
                print(f"  OK  {out_name}  [{start_s:.2f}-{end_s:.2f}s]")
            except Exception as exc:
                print(f"  ERR {out_name}: {exc}", file=sys.stderr)

    # Combined-flaw files
    for i in range(INJECT_COMBINED_COUNT):
        if not wav_files:
            break
        wav_path = wav_files[int(rng.integers(0, len(wav_files)))]
        try:
            words = _words_for(wav_path)
        except Exception as exc:
            print(f"  SKIP combined {i}: {exc}", file=sys.stderr)
            continue

        import librosa
        import soundfile as sf
        from src.dataset.generator import (
            pick_region,
            inject,
            INJECTOR_MAP,
            _fade,
        )

        waveform, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        waveform = waveform.astype(__import__("numpy").float32)
        combined_rows = []
        n_flaws = int(rng.integers(2, 4))
        chosen = list(
            rng.choice(len(FLAW_TYPES), size=min(n_flaws, len(FLAW_TYPES)), replace=False)
        )
        stem = wav_path.stem
        out_name = f"{stem}_combined_{i:02d}.wav"
        out_path = out_dir / out_name

        current_waveform = waveform.copy()
        current_words = list(words)  # crude copy; word timings stay fixed
        time_shift = 0.0

        for idx in chosen:
            flaw_type = FLAW_TYPES[idx]
            severity = int(rng.integers(1, 6))
            try:
                w_start, w_end = pick_region(current_words, rng)
                r_start = current_words[w_start].start + time_shift
                r_end = current_words[w_end].end + time_shift
                fn = INJECTOR_MAP[flaw_type]
                new_waveform, label_start, label_end = inject(
                    current_waveform, r_start, r_end, fn, severity, rng
                )
                current_waveform = new_waveform
                delta = label_end - r_end
                time_shift += delta
                combined_rows.append(
                    {
                        "file": out_name,
                        "source": wav_path.name,
                        "flaw_type": flaw_type,
                        "severity": severity,
                        "start_sec": round(label_start, 3),
                        "end_sec": round(label_end, 3),
                    }
                )
            except Exception as exc:
                print(f"  ERR combined flaw {flaw_type}: {exc}", file=sys.stderr)

        if combined_rows:
            sf.write(str(out_path), current_waveform, SAMPLE_RATE, subtype="PCM_16")
            rows.extend(combined_rows)
            print(f"  OK  {out_name} ({len(combined_rows)} flaws)")

    # Write CSV
    with INJECTIONS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} label rows -> {INJECTIONS_CSV}")


if __name__ == "__main__":
    main()
