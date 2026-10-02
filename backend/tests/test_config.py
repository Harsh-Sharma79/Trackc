"""
tests/test_config.py
====================
Tests for src/config.py – ensure all constants are present,
correctly typed, and within reasonable ranges.
"""
import pytest
import src.config as cfg
from pathlib import Path


def test_seed_is_int():
    assert isinstance(cfg.SEED, int)
    assert cfg.SEED == 42


def test_sample_rate():
    assert cfg.SAMPLE_RATE == 16_000
    assert cfg.CHANNELS == 1


def test_frame_hop_sizes():
    assert cfg.FRAME_LENGTH > 0
    assert cfg.HOP_LENGTH > 0
    assert cfg.HOP_LENGTH <= cfg.FRAME_LENGTH


def test_fmin_fmax():
    assert 0 < cfg.FMIN_HZ < cfg.FMAX_HZ


def test_pause_thresholds():
    assert cfg.PAUSE_ENERGY_THRESHOLD < 0        # must be negative dBFS
    assert cfg.PAUSE_MIN_DURATION > 0
    assert cfg.PAUSE_IDEAL_MAX > cfg.PAUSE_MIN_DURATION


def test_pace_range():
    assert 0 < cfg.PACE_IDEAL_WPM_MIN < cfg.PACE_IDEAL_WPM_MAX


def test_pitch_var_range():
    assert 0 <= cfg.PITCH_VAR_IDEAL_MIN < cfg.PITCH_VAR_IDEAL_MAX


def test_paths_are_path_objects():
    for attr in ("PROJECT_ROOT", "DATA_DIR", "RAW_DIR", "FLAWED_DIR",
                 "LABELS_DIR", "OUTPUTS_DIR", "ALIGNMENT_CACHE_DIR"):
        assert isinstance(getattr(cfg, attr), Path), f"{attr} should be a Path"


def test_alignment_backend_valid():
    assert cfg.ALIGNMENT_BACKEND in ("whisperx", "torchaudio", "proportional")


def test_dtw_radius():
    assert cfg.DTW_RADIUS > 0

