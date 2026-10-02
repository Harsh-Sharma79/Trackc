"""
tests/test_evaluate_detections.py
==================================
Tests for scripts/evaluate_detections.py: IoU computation, bipartite matching,
and precision/recall/F1 aggregation.
"""
from __future__ import annotations

import pytest

from scripts.evaluate_detections import (
    Interval,
    calc_metrics,
    compute_iou,
    evaluate_records,
    match_detections,
)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------

def test_iou_identical():
    """Identical intervals should have IoU == 1.0."""
    assert abs(compute_iou(1.0, 3.0, 1.0, 3.0) - 1.0) < 1e-6


def test_iou_no_overlap():
    """Disjoint intervals should have IoU == 0.0."""
    assert compute_iou(1.0, 2.0, 3.0, 4.0) == 0.0


def test_iou_abutting():
    """Abutting intervals share an edge but zero interior → IoU == 0.0."""
    assert compute_iou(1.0, 2.0, 2.0, 3.0) == 0.0


def test_iou_half_overlap():
    """[0, 2] and [1, 3]: intersection = [1, 2] (len 1), union = [0, 3] (len 3) → IoU = 1/3."""
    expected = 1.0 / 3.0
    assert abs(compute_iou(0.0, 2.0, 1.0, 3.0) - expected) < 1e-6


def test_iou_subset():
    """[1, 2] inside [0, 4]: inter = 1, union = 4 → IoU = 0.25."""
    assert abs(compute_iou(1.0, 2.0, 0.0, 4.0) - 0.25) < 1e-6


def test_iou_symmetric():
    """compute_iou(A, B) must equal compute_iou(B, A)."""
    assert compute_iou(0.5, 1.5, 1.0, 2.5) == compute_iou(1.0, 2.5, 0.5, 1.5)


# ---------------------------------------------------------------------------
# match_detections
# ---------------------------------------------------------------------------

def test_match_perfect():
    """One identical pred + GT → TP=1, FP=0, FN=0."""
    gt = [Interval(1.0, 2.0, "PACE_TOO_SLOW")]
    pred = [Interval(1.0, 2.0, "PACE_TOO_SLOW")]
    tp, fp, fn = match_detections(gt, pred, iou_thresh=0.3)
    assert (tp, fp, fn) == (1, 0, 0)


def test_match_type_mismatch():
    """Overlapping intervals with DIFFERENT flaw_type must NOT match."""
    gt = [Interval(1.0, 2.0, "PACE_TOO_SLOW")]
    pred = [Interval(1.0, 2.0, "PACE_TOO_FAST")]
    tp, fp, fn = match_detections(gt, pred, iou_thresh=0.3)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_below_iou_threshold():
    """Small overlap with IoU < 0.3 must count as false positive and false negative."""
    # [0, 10] and [9.5, 10.5]: inter=0.5, union=10.5 → IoU ≈ 0.048 < 0.3
    gt = [Interval(0.0, 10.0, "FILLER")]
    pred = [Interval(9.5, 10.5, "FILLER")]
    tp, fp, fn = match_detections(gt, pred, iou_thresh=0.3)
    assert (tp, fp, fn) == (0, 1, 1)


def test_match_one_to_one():
    """Two identical predictions must NOT both match a single ground-truth interval."""
    gt = [Interval(1.0, 2.0, "PACE_TOO_SLOW")]
    pred = [
        Interval(1.0, 2.0, "PACE_TOO_SLOW"),
        Interval(1.0, 2.0, "PACE_TOO_SLOW"),
    ]
    tp, fp, fn = match_detections(gt, pred, iou_thresh=0.3)
    assert tp == 1
    assert fp == 1
    assert fn == 0


# ---------------------------------------------------------------------------
# calc_metrics
# ---------------------------------------------------------------------------

def test_metrics_perfect():
    m = calc_metrics(10, 0, 0)
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0


def test_metrics_zero_predictions():
    m = calc_metrics(0, 0, 5)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0


def test_metrics_f1_harmonic_mean():
    m = calc_metrics(5, 5, 5)  # prec=0.5, rec=0.5 → f1=0.5
    assert abs(m.precision - 0.5) < 1e-4
    assert abs(m.recall - 0.5) < 1e-4
    assert abs(m.f1 - 0.5) < 1e-4


# ---------------------------------------------------------------------------
# evaluate_records (aggregation)
# ---------------------------------------------------------------------------

def test_evaluate_records_overall():
    gt = {
        "file1.wav": [
            Interval(1.0, 3.0, "PACE_TOO_SLOW"),
            Interval(5.0, 6.0, "FILLER"),
        ]
    }
    pred = {
        "file1.wav": [
            Interval(1.0, 3.0, "PACE_TOO_SLOW"),  # match
            Interval(5.0, 6.0, "FILLER"),          # match
        ]
    }
    results = evaluate_records(gt, pred, iou_thresh=0.3)
    assert "PACE_TOO_SLOW" in results
    assert "FILLER" in results
    assert "OVERALL" in results
    assert results["OVERALL"].f1 == 1.0
