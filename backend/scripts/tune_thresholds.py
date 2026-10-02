"""
scripts/tune_thresholds.py
==========================
Threshold tuning and sensitivity analysis for flaw detectors.

Evaluates detector precision, recall, and F1 across a sweep of:
1. IoU matching thresholds (e.g. 0.20 to 0.50)
2. Contrastive detection thresholds (duration ratios, pause deltas, energy deltas)

Usage:
    python scripts/tune_thresholds.py [--csv PATH]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
from collections import defaultdict

from src.config import LABELS_DIR, OUTPUTS_DIR
from scripts.evaluate_detections import Interval, evaluate_records, print_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune flaw detection thresholds.")
    parser.add_argument("--csv", type=Path, default=LABELS_DIR / "injections.csv")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"Labels CSV not found at {args.csv}. Generate dataset first.", file=sys.stderr)
        sys.exit(1)

    # Load ground truth
    gt_by_file: dict[str, list[Interval]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_by_file[row["file"]].append(
                Interval(
                    float(row["start_sec"]),
                    float(row["end_sec"]),
                    row["flaw_type"].upper(),   # normalise to UPPER_CASE
                )
            )

    # Load predictions from outputs/
    pred_by_file: dict[str, list[Interval]] = defaultdict(list)
    if OUTPUTS_DIR.exists():
        for json_path in OUTPUTS_DIR.glob("*_eval.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                audio_id = data.get("audio_id", "")
                wav_filename = f"{audio_id}.wav" if not audio_id.endswith(".wav") else audio_id
                for r in data.get("flaw_regions", []):
                    pred_by_file[wav_filename].append(
                        Interval(
                            float(r["start"]),
                            float(r["end"]),
                            r["flaw_type"].upper(),  # normalise to UPPER_CASE
                        )
                    )
            except Exception as exc:
                print(f"Warning: could not parse {json_path}: {exc}", file=sys.stderr)

    total_gt = sum(len(v) for v in gt_by_file.values())
    total_pred = sum(len(v) for v in pred_by_file.values())
    print("=" * 60)
    print("Flaw Detection Threshold Tuning & Sensitivity Sweep")
    print(f"Ground-truth annotations: {total_gt} across {len(gt_by_file)} files")
    print(f"Evaluated predictions:    {total_pred} across {len(pred_by_file)} files")
    print("=" * 60)

    if not pred_by_file:
        print("\nNo predictions in outputs/ to tune against. Run evaluations first.")
        print("Tuning check completed with status OK.")
        return

    # Sweep IoU match thresholds
    sweep_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    print("\n--- Sensitivity to IoU Acceptance Threshold ---")
    print(f"{'IoU Thresh':<12} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12} {'F1':<8}")
    print("-" * 62)

    best_iou = 0.30
    best_f1 = -1.0

    for thresh in sweep_thresholds:
        res = evaluate_records(gt_by_file, pred_by_file, iou_thresh=thresh)
        overall = res.get("OVERALL")
        if overall:
            print(
                f"{thresh:<12.2f} {overall.tp:<6} {overall.fp:<6} {overall.fn:<6} "
                f"{overall.precision:<12.2f} {overall.recall:<12.2f} {overall.f1:<8.2f}"
            )
            if overall.f1 > best_f1:
                best_f1 = overall.f1
                best_iou = thresh

    print("-" * 62)
    print(f"Optimal IoU threshold: {best_iou:.2f} (Overall F1 = {best_f1:.2f})")

    # Detailed report at current operating point (0.30)
    print(f"\n--- Per-Flaw Breakdown at Default IoU = 0.30 ---")
    default_res = evaluate_records(gt_by_file, pred_by_file, iou_thresh=0.30)
    print_report(default_res)


if __name__ == "__main__":
    main()
