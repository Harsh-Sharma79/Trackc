"""
tests/test_features_alignment.py
=================================
Tests for src/features/alignment.py.
"""
import numpy as np
import pytest

from src.features.alignment import (
    _sakoe_chiba_dtw,
    apply_warp,
    dtw_align,
    align_features,
)


def _cost_matrix(n: int, m: int) -> np.ndarray:
    """Simple diagonal cost matrix (identity-like alignment)."""
    i_idx = np.arange(n)[:, None]
    j_idx = np.arange(m)[None, :]
    return np.abs(i_idx / n - j_idx / m).astype(np.float64)


class TestSakoeChiba:
    def test_same_length_diagonal_path(self):
        n = 10
        cost = np.eye(n, dtype=np.float64) * 0.0 + 1.0
        cost[np.arange(n), np.arange(n)] = 0.0
        ref, cand = _sakoe_chiba_dtw(cost, radius=5)
        assert len(ref) == len(cand)
        assert ref[-1] == n - 1
        assert cand[-1] == n - 1

    def test_path_starts_at_origin(self):
        cost = _cost_matrix(8, 8)
        ref, cand = _sakoe_chiba_dtw(cost, radius=4)
        assert ref[0] == 0
        assert cand[0] == 0

    def test_path_ends_at_terminal(self):
        n, m = 12, 10
        cost = _cost_matrix(n, m)
        ref, cand = _sakoe_chiba_dtw(cost, radius=6)
        assert ref[-1] == n - 1
        assert cand[-1] == m - 1

    def test_monotone_path(self):
        cost = _cost_matrix(8, 8)
        ref, cand = _sakoe_chiba_dtw(cost, radius=4)
        assert np.all(np.diff(ref) >= 0)
        assert np.all(np.diff(cand) >= 0)


class TestDtwAlign:
    def test_returns_two_arrays(self):
        a = np.random.rand(20, 2)
        b = np.random.rand(15, 2)
        ref, cand = dtw_align(a, b)
        assert ref.ndim == 1
        assert cand.ndim == 1

    def test_path_lengths_equal(self):
        a = np.random.rand(10, 1)
        b = np.random.rand(10, 1)
        ref, cand = dtw_align(a, b)
        assert len(ref) == len(cand)

    def test_1d_input_accepted(self):
        a = np.arange(10, dtype=float)
        b = np.arange(8, dtype=float)
        ref, cand = dtw_align(a, b)
        assert len(ref) > 0


class TestApplyWarp:
    def test_output_length_matches_target(self):
        feature = np.arange(15, dtype=float)
        cand_path = np.tile(np.arange(15), 2)[:15]
        out = apply_warp(feature, cand_path, target_length=10)
        assert len(out) == 10

    def test_identity_warp(self):
        feature = np.array([1.0, 2.0, 3.0, 4.0])
        path = np.arange(4)
        out = apply_warp(feature, path, target_length=4)
        assert len(out) == 4


class TestAlignFeatures:
    def test_output_shapes_match_baseline(self):
        n_b, n_c = 50, 40
        b_p = np.random.rand(n_b)
        b_e = np.random.rand(n_b)
        c_p = np.random.rand(n_c)
        c_e = np.random.rand(n_c)
        bp, be, wp, we = align_features(b_p, b_e, c_p, c_e)
        assert len(bp) == n_b
        assert len(be) == n_b
        assert len(wp) == n_b
        assert len(we) == n_b

    def test_identical_signals_low_error(self):
        n = 30
        sig = np.sin(np.linspace(0, 2 * np.pi, n))
        _, _, wp, we = align_features(sig, sig, sig, sig)
        # Warped should be very close to original for identical inputs
        assert np.mean(np.abs(wp - sig)) < 0.5
