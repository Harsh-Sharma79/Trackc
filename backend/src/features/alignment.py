"""
src/features/alignment.py
=========================
DTW-based alignment of candidate audio features to a baseline.

Uses a Sakoe-Chiba banded DTW (via scipy.spatial.distance + custom
path search) to warp the candidate feature stream onto the baseline's
time axis, enabling fair frame-by-frame comparison.

Public API:
- dtw_align        – align two 2-D feature matrices, return index mapping
- apply_warp       – resample a feature array using the DTW index mapping
- align_features   – convenience wrapper for the full alignment workflow
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from src.config import DTW_RADIUS


def _sakoe_chiba_dtw(
    cost_matrix: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sakoe-Chiba banded DTW on a pre-computed local cost matrix.

    Args:
        cost_matrix: 2-D array of shape (N, M) where N = reference length,
                     M = candidate length.  Lower values = more similar.
        radius:      Half-bandwidth of the Sakoe-Chiba band (in frames).

    Returns:
        Tuple (ref_path, cand_path) – 1-D integer arrays of equal length
        giving the optimal alignment path indices.
    """
    n, m = cost_matrix.shape
    inf = np.inf

    # Ensure the band covers the length difference so the end cell (n-1, m-1) is reachable
    band_radius = max(radius, abs(n - m) + 5)

    # Vectorized Sakoe-Chiba DTW: process every row in one NumPy ufunc call
    # instead of an inner Python for-loop (reduces ~N*2*radius Python ops → N NumPy calls).
    acc = np.full((n, m), inf, dtype=np.float64)
    acc[0, 0] = cost_matrix[0, 0]

    for i in range(1, n):
        j_lo = max(0, i - band_radius)
        j_hi = min(m - 1, i + band_radius)
        js = np.arange(j_lo, j_hi + 1)

        # Predecessor: diagonal (i-1, j-1)
        prev_diag = acc[i - 1, np.clip(js - 1, 0, m - 1)].copy()
        prev_diag[js == 0] = inf

        # Predecessor: up (i-1, j)
        prev_up = acc[i - 1, js]

        # Predecessor: left (i, j-1) — requires a short left-to-right sweep since
        # acc[i, j-1] may have just been written in this same row.
        prev_left = np.full(len(js), inf, dtype=np.float64)
        if j_lo > 0:
            prev_left[0] = acc[i, j_lo - 1]
        best_predecessor = np.minimum(np.minimum(prev_diag, prev_up), prev_left)
        row_cost = cost_matrix[i, j_lo: j_hi + 1] + best_predecessor

        # Push left-propagation: subsequent cells can use acc[i, j-1] from this row
        # We need a forward-pass for the left-dependency (cumulative min-update).
        for k in range(1, len(js)):
            left_candidate = row_cost[k - 1]
            row_cost[k] = min(row_cost[k], cost_matrix[i, js[k]] + left_candidate)

        acc[i, j_lo: j_hi + 1] = row_cost

    # Backtrack
    ref_path: list[int] = []
    cand_path: list[int] = []
    i, j = n - 1, m - 1
    ref_path.append(i)
    cand_path.append(j)

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            best = min(
                acc[i - 1, j - 1],
                acc[i - 1, j],
                acc[i, j - 1],
            )
            if best == acc[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif best == acc[i - 1, j]:
                i -= 1
            else:
                j -= 1
        ref_path.append(i)
        cand_path.append(j)

    ref_path.reverse()
    cand_path.reverse()
    return np.array(ref_path, dtype=np.int64), np.array(cand_path, dtype=np.int64)


def dtw_align(
    baseline_features: np.ndarray,
    candidate_features: np.ndarray,
    metric: str = "euclidean",
    radius: int = DTW_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    """Align candidate feature stream to baseline using Sakoe-Chiba DTW.

    Args:
        baseline_features:  2-D array (N_baseline, n_features).
        candidate_features: 2-D array (N_candidate, n_features).
        metric:             Distance metric for local cost (default "euclidean").
        radius:             Sakoe-Chiba band radius in frames.

    Returns:
        Tuple (ref_path, cand_path) of integer index arrays representing
        the optimal warping path.
    """
    if baseline_features.ndim == 1:
        baseline_features = baseline_features[:, np.newaxis]
    if candidate_features.ndim == 1:
        candidate_features = candidate_features[:, np.newaxis]

    cost_matrix = cdist(baseline_features, candidate_features, metric=metric)
    return _sakoe_chiba_dtw(cost_matrix, radius=radius)


def apply_warp(
    feature_array: np.ndarray,
    cand_path: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Resample a 1-D feature array to ``target_length`` using the DTW path.

    For each baseline frame index 0..target_length-1, average the candidate frames
    mapped to that baseline frame (excluding NaNs if voiced frames are present).
    If all mapped candidate frames are unvoiced (NaN), the result remains NaN.

    Args:
        feature_array: 1-D candidate feature array of length N_candidate.
        cand_path:     Candidate indices from the DTW path (length = path_len).
        target_length: Desired output length (= baseline frame count).

    Returns:
        1-D float64 array of shape (target_length,).
    """
    buckets: list[list[float]] = [[] for _ in range(target_length)]
    for path_idx, c_idx in enumerate(cand_path):
        buckets[path_idx % target_length].append(float(feature_array[c_idx]))

    output = np.zeros(target_length, dtype=np.float64)
    has_nan_input = bool(np.any(np.isnan(feature_array)))

    for idx, vals in enumerate(buckets):
        if not vals:
            output[idx] = np.nan if has_nan_input else 0.0
            continue
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            output[idx] = float(np.mean(finite_vals))
        else:
            output[idx] = np.nan if has_nan_input else 0.0

    return output


def align_features(
    baseline_pitch: np.ndarray,
    baseline_energy: np.ndarray,
    candidate_pitch: np.ndarray,
    candidate_energy: np.ndarray,
    radius: int = DTW_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full alignment workflow: DTW-align candidate to baseline frame grid.

    Stacks pitch + energy into a joint feature space for alignment, then
    applies the resulting warp to each feature separately.

    Args:
        baseline_pitch:   Baseline pitch array (semitones), length N_b.
        baseline_energy:  Baseline energy array (z-score), length N_b.
        candidate_pitch:  Candidate pitch array, length N_c.
        candidate_energy: Candidate energy array, length N_c.
        radius:           Sakoe-Chiba band radius.

    Returns:
        Tuple of four arrays all of length N_b:
          (baseline_pitch, baseline_energy,
           warped_candidate_pitch, warped_candidate_energy)
    """
    # Clean NaNs in pitch for finite distance computation in DTW
    bp_dtw = np.nan_to_num(baseline_pitch, nan=0.0)
    cp_dtw = np.nan_to_num(candidate_pitch, nan=0.0)

    baseline_feat = np.stack([bp_dtw, baseline_energy], axis=1)
    candidate_feat = np.stack([cp_dtw, candidate_energy], axis=1)

    _ref_path, cand_path = dtw_align(baseline_feat, candidate_feat, radius=radius)

    n_b = len(baseline_pitch)
    warped_pitch = apply_warp(candidate_pitch, cand_path, n_b)
    warped_energy = apply_warp(candidate_energy, cand_path, n_b)

    return baseline_pitch, baseline_energy, warped_pitch, warped_energy

