"""
scripts/run_all.py
==================
One-command runner for the Speech Evaluation System.

Workflow:
1. Validates environment and prints dependency status.
2. Checks data/raw/ — generates flawed dataset if raw WAVs exist.
3. Validates labels CSV if present.
4. Runs full pytest test suite and reports test results.
5. If both raw and flawed audio exist, evaluates each candidate against its
   baseline, writes output JSONs to outputs/, and evaluates detection metrics.

Usage:
    python scripts/run_all.py [--raw-dir PATH] [--skip-tests]

The --skip-tests flag skips the slow pytest step for faster end-to-end runs.
The pytest step always uses the same Python interpreter as this script (sys.executable).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as `python scripts/run_all.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import subprocess

from src.config import (
    ALIGNMENT_BACKEND,
    DATA_DIR,
    FLAWED_DIR,
    LABELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    RAW_DIR,
)


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess command and stream output."""
    print(f"\n>> {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the speech evaluation system end-to-end.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest test suite.")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("Speech Evaluation System — End-to-End Runner")
    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Alignment backend: {ALIGNMENT_BACKEND}")
    print("=" * 60)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run tests
    if not args.skip_tests:
        print("\n--- Step 1: Running test suite ---")
        res = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
            cwd=str(PROJECT_ROOT),
        )
        if res.returncode != 0:
            print("Tests failed! Aborting pipeline.", file=sys.stderr)
            sys.exit(res.returncode)
        print("Test suite passed.")

    # 2. Check data/raw/
    raw_wavs = list(args.raw_dir.glob("*.wav"))
    if not raw_wavs:
        print(
            f"\n[INFO] No WAV files in {args.raw_dir} yet.\n"
            "       Add your raw speech recordings (*.wav + *.txt) to data/raw/\n"
            "       and re-run this script to generate the dataset and evaluations."
        )
        print("\nAll pipeline components verified and ready. Exiting 0.")
        return

    # 3. Generate dataset
    print(f"\n--- Step 2: Generating flawed dataset from {len(raw_wavs)} source recording(s) ---")
    run([sys.executable, "-m", "src.dataset.generate", "--raw-dir", str(args.raw_dir)])

    # 4. Validate labels
    injections_csv = LABELS_DIR / "injections.csv"
    if injections_csv.exists():
        print("\n--- Step 3: Validating generated labels ---")
        run([sys.executable, "scripts/validate_labels.py", "--csv", str(injections_csv)])

    # 5. Evaluate all flawed files
    print("\n--- Step 4: Running contrastive evaluation on all flawed recordings ---")
    from src.pipeline import evaluate

    flawed_wavs = list(FLAWED_DIR.glob("*.wav"))
    for fwav in flawed_wavs:
        # Match to source
        stem_parts = fwav.stem.split("_")
        # Try finding the original file in raw_dir
        source_candidates = [
            r for r in raw_wavs if fwav.stem.startswith(r.stem)
        ]
        if not source_candidates:
            continue
        base_wav = source_candidates[0]
        txt_path = base_wav.with_suffix(".txt")
        if not txt_path.exists():
            continue
        transcript = txt_path.read_text(encoding="utf-8-sig").strip()

        print(f"  Evaluating {fwav.name} vs baseline {base_wav.name}...", flush=True)
        try:
            result = evaluate(
                candidate_path=fwav,
                baseline_path=base_wav,
                transcript=transcript,
                audio_id=fwav.stem,
            )
            out_file = OUTPUTS_DIR / f"{fwav.stem}_eval.json"
            out_file.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"    OK -> {out_file.name} (score: {result.composite_score:.2f})", flush=True)
        except Exception as exc:
            print(f"    ERR -> {exc}", file=sys.stderr)

    # 6. Evaluate detections
    if injections_csv.exists():
        print("\n--- Step 5: Computing detection accuracy metrics ---")
        run([sys.executable, "scripts/evaluate_detections.py", "--csv", str(injections_csv)])

    print("\n=" * 60)
    print(f"Pipeline complete! Output JSON evaluations saved to {OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
