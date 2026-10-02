"""
scripts/validate_labels.py
===========================
Validate data/labels/injections.csv: check counts, time bounds, and severity range.

Usage:
    python scripts/validate_labels.py [--csv PATH]

Prints a per-flaw-type count summary and reports any anomalies.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as `python scripts/validate_labels.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
from collections import Counter


from src.config import FLAWED_DIR, LABELS_DIR

INJECTIONS_CSV = LABELS_DIR / "injections.csv"


def validate(csv_path: Path) -> bool:
    """Validate the injections CSV.

    Checks:
    1. Required columns are present.
    2. start_sec < end_sec for every row.
    3. severity in [1, 5].
    4. Every flawed file referenced in csv exists in FLAWED_DIR.

    Returns:
        True if all checks pass, False otherwise.
    """
    required_cols = {"file", "source", "flaw_type", "severity", "start_sec", "end_sec"}
    errors: list[str] = []
    counts: Counter = Counter()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return False

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not required_cols.issubset(reader.fieldnames):
            missing = required_cols - set(reader.fieldnames or [])
            print(f"ERROR: Missing columns: {missing}", file=sys.stderr)
            return False

        for i, row in enumerate(reader, start=2):  # line 1 = header
            flaw = row["flaw_type"]
            counts[flaw] += 1
            try:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
                sev = int(row["severity"])
            except ValueError as e:
                errors.append(f"Row {i}: parse error — {e}")
                continue

            if end <= start:
                errors.append(
                    f"Row {i} ({flaw}): end_sec ({end}) <= start_sec ({start})"
                )
            if not (1 <= sev <= 5):
                errors.append(f"Row {i} ({flaw}): severity {sev} out of [1,5]")

            file_path = FLAWED_DIR / row["file"]
            if not file_path.exists():
                errors.append(f"Row {i}: file not found: {file_path}")

    print("\n=== Flaw type counts ===")
    for flaw, count in sorted(counts.items()):
        print(f"  {flaw:<30} {count}")
    print(f"\n  TOTAL rows: {sum(counts.values())}")

    if errors:
        print(f"\n=== {len(errors)} ERROR(S) ===", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return False

    print("\nAll checks passed.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate injections.csv")
    parser.add_argument("--csv", type=Path, default=INJECTIONS_CSV)
    args = parser.parse_args()
    ok = validate(args.csv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
