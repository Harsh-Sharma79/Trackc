"""
tests/test_explain_templates.py
================================
Tests for src/explain/templates.py.
"""
import pytest
from src.explain.templates import _severity_adverb, explain
from src.schema import FlawType


class TestSeverityAdverb:
    def test_high_severity(self):
        assert _severity_adverb(0.9) == "severely"

    def test_medium_severity(self):
        assert _severity_adverb(0.5) == "noticeably"

    def test_low_severity(self):
        assert _severity_adverb(0.25) == "slightly"

    def test_very_low_severity(self):
        assert _severity_adverb(0.1) == "marginally"


class TestExplain:
    def test_returns_string(self):
        result = explain(FlawType.PACE_TOO_FAST, 0.7, {"wpm": 200.0})
        assert isinstance(result, str)
        assert len(result) > 10

    def test_all_flaw_types_produce_output(self):
        metadata_map = {
            FlawType.PACE_TOO_FAST: {"wpm": 200.0},
            FlawType.PACE_TOO_SLOW: {"wpm": 80.0},
            FlawType.PITCH_MONOTONE: {"pitch_std_st": 0.5},
            FlawType.PITCH_ERRATIC: {"pitch_std_st": 12.0},
            FlawType.ENERGY_LOW: {"mean_energy_z": -2.5},
            FlawType.ENERGY_INCONSISTENT: {"energy_std_z": 3.5},
            FlawType.PAUSE_MISSING: {},
            FlawType.PAUSE_EXCESSIVE: {"pause_duration_s": 2.0},
            FlawType.PAUSE_MISPLACED: {},
            FlawType.FILLER: {},
            FlawType.UNCLEAR: {},
        }
        for flaw_type, meta in metadata_map.items():
            result = explain(flaw_type, 0.5, meta)
            assert isinstance(result, str), f"Failed for {flaw_type}"
            assert len(result) > 5, f"Too short for {flaw_type}"


    def test_severity_affects_adverb(self):
        fast_high = explain(FlawType.PACE_TOO_FAST, 0.9, {"wpm": 250.0})
        fast_low = explain(FlawType.PACE_TOO_FAST, 0.1, {"wpm": 250.0})
        assert fast_high != fast_low

    def test_no_metadata_does_not_crash(self):
        # Templates that don't require metadata should still work
        result = explain(FlawType.PAUSE_MISPLACED, 0.4)
        assert isinstance(result, str)

    def test_missing_metadata_key_returns_fallback(self):
        # PACE_TOO_FAST template needs {wpm} – omit it
        result = explain(FlawType.PACE_TOO_FAST, 0.5, {})
        # Should return the fallback string, not raise
        assert isinstance(result, str)
        assert len(result) > 0

    def test_adverb_present_in_output(self):
        result = explain(FlawType.ENERGY_LOW, 0.8, {"mean_energy_z": -2.0})
        assert "severely" in result
