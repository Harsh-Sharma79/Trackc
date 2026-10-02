"""
scripts/evaluate_detections.py
==============================
Evaluate flaw detection accuracy against ground-truth labels in injections.csv.

Metrics:
- IoU (Intersection over Union) between predicted and ground-truth intervals.
- A prediction is a TRUE POSITIVE if:
    1. predicted flaw_type == ground-truth flaw_type
    2. IoU(predicted_interval, ground_truth_interval) >= iou_threshold (default 0.3)
- Precision, Recall, F1 computed per flaw type and overall macro/micro average.

Usage:
    python scripts/evaluate_detections.py [--csv PATH] [--iou-thresh FLOAT]
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as `python scripts/evaluate_detections.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from collections import defaultdict
from typing import NamedTuple

from src.config import LABELS_DIR


class Interval(NamedTuple):
    start: float
    end: float
    flaw_type: str


class Metrics(NamedTuple):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def compute_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Compute 1-D temporal Intersection over Union between two intervals.

    Args:
        start1, end1: First interval boundaries in seconds.
        start2, end2: Second interval boundaries in seconds.

    Returns:
        IoU in [0, 1].
    """
    inter_start = max(start1, start2)
    inter_end = min(end1, end2)
    inter_len = max(0.0, inter_end - inter_start)
    if inter_len == 0.0:
        return 0.0

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_len = union_end - union_start
    if union_len <= 0.0:
        return 0.0

    return inter_len / union_len


def match_detections(
    ground_truth: list[Interval],
    predictions: list[Interval],
    iou_thresh: float = 0.3,
) -> tuple[int, int, int]:
    """Greedy bipartite matching of predictions to ground-truth intervals.

    Args:
        ground_truth: True flaw intervals.
        predictions:  Predicted flaw intervals.
        iou_thresh:   Minimum IoU to count as a true positive.

    Returns:
        (true_positives, false_positives, false_negatives)
    """
    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            if gt.flaw_type != pred.flaw_type:
                continue
            iou = compute_iou(pred.start, pred.end, gt.start, gt.end)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(ground_truth) - len(matched_gt)
    return tp, fp, fn


def calc_metrics(tp: int, fp: int, fn: int) -> Metrics:
    """Compute Precision, Recall, F1 from counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return Metrics(tp, fp, fn, round(prec, 4), round(rec, 4), round(f1, 4))


def evaluate_records(
    gt_by_file: dict[str, list[Interval]],
    pred_by_file: dict[str, list[Interval]],
    iou_thresh: float = 0.3,
) -> dict[str, Metrics]:
    """Compute per-flaw-type and overall metrics.

    Args:
        gt_by_file:   Mapping of filename → list of true intervals.
        pred_by_file: Mapping of filename → list of predicted intervals.
        iou_thresh:   IoU match threshold.

    Returns:
        Dictionary of flaw_type → Metrics, plus an 'OVERALL' entry.
    """
    all_files = set(gt_by_file.keys()) | set(pred_by_file.keys())
    per_type_tp: dict[str, int] = defaultdict(int)
    per_type_fp: dict[str, int] = defaultdict(int)
    per_type_fn: dict[str, int] = defaultdict(int)

    # Collect all known flaw types
    all_types: set[str] = set()
    for intervals in gt_by_file.values():
        all_types.update(i.flaw_type for i in intervals)
    for intervals in pred_by_file.values():
        all_types.update(i.flaw_type for i in intervals)

    for fname in all_files:
        gts = gt_by_file.get(fname, [])
        preds = pred_by_file.get(fname, [])

        for ftype in all_types:
            g_subset = [g for g in gts if g.flaw_type == ftype]
            p_subset = [p for p in preds if p.flaw_type == ftype]
            tp, fp, fn = match_detections(g_subset, p_subset, iou_thresh)
            per_type_tp[ftype] += tp
            per_type_fp[ftype] += fp
            per_type_fn[ftype] += fn

    results: dict[str, Metrics] = {}
    total_tp = sum(per_type_tp.values())
    total_fp = sum(per_type_fp.values())
    total_fn = sum(per_type_fn.values())

    for ftype in sorted(all_types):
        results[ftype] = calc_metrics(
            per_type_tp[ftype], per_type_fp[ftype], per_type_fn[ftype]
        )

    results["OVERALL"] = calc_metrics(total_tp, total_fp, total_fn)
    return results


def print_report(results: dict[str, Metrics]) -> None:
    """Print a clean Markdown-compatible metrics table."""
    print("\n| Flaw Type | TP | FP | FN | Precision | Recall | F1 |")
    print("|---|---|---|---|---|---|---|")
    for ftype, m in results.items():
        prefix = "**" if ftype == "OVERALL" else ""
        suffix = "**" if ftype == "OVERALL" else ""
        print(
            f"| {prefix}{ftype}{suffix} | {m.tp} | {m.fp} | {m.fn} | "
            f"{m.precision:.2f} | {m.recall:.2f} | {prefix}{m.f1:.2f}{suffix} |"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate flaw detections.")
    parser.add_argument(
        "--csv", type=Path, default=LABELS_DIR / "injections.csv"
    )
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"Labels CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

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

    # Check for prediction JSONs in outputs/
    from src.config import OUTPUTS_DIR
    pred_by_file: dict[str, list[Interval]] = defaultdict(list)
    if OUTPUTS_DIR.exists():
        for json_path in OUTPUTS_DIR.glob("*_eval.json"):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
                # Match candidate audio file name: audio_id or file stem
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
    print(f"Loaded {total_gt} ground-truth regions across {len(gt_by_file)} files.")
    print(f"Loaded {total_pred} predicted regions across {len(pred_by_file)} evaluated files.")

    if pred_by_file:
        results = evaluate_records(gt_by_file, pred_by_file, iou_thresh=args.iou_thresh)
        print_report(results)
    else:
        print("No predictions found in outputs/. Run pipeline evaluation to generate predictions.")


if __name__ == "__main__":
    import json
    main()
