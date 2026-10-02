"""
tests/test_schema.py
====================
Tests for src/schema.py – validate Pydantic models, validators, and enums.
"""
import pytest
from pydantic import ValidationError

from src.schema import (
    EvaluationResult,
    FlawRegion,
    FlawType,
    RubricDimension,
    RubricScore,
)


# ── FlawRegion ──────────────────────────────────────────────────────────────

class TestFlawRegion:
    def test_valid_flaw_region(self):
        fr = FlawRegion(
            start=1.0,
            end=2.0,
            flaw_type=FlawType.PACE_TOO_FAST,
            severity=0.5,
            explanation="Test explanation.",
        )
        assert fr.start == 1.0
        assert fr.end == 2.0
        assert fr.severity == 0.5

    def test_end_must_be_after_start(self):
        with pytest.raises(ValidationError):
            FlawRegion(
                start=2.0,
                end=1.0,
                flaw_type=FlawType.PACE_TOO_FAST,
                severity=0.5,
                explanation="Bad.",
            )

    def test_end_equal_start_invalid(self):
        with pytest.raises(ValidationError):
            FlawRegion(
                start=1.0,
                end=1.0,
                flaw_type=FlawType.PACE_TOO_SLOW,
                severity=0.3,
                explanation="Equal times are invalid.",
            )

    def test_severity_bounds(self):
        with pytest.raises(ValidationError):
            FlawRegion(
                start=0.0, end=1.0,
                flaw_type=FlawType.PITCH_MONOTONE,
                severity=1.5,   # > 1
                explanation="x",
            )
        with pytest.raises(ValidationError):
            FlawRegion(
                start=0.0, end=1.0,
                flaw_type=FlawType.PITCH_MONOTONE,
                severity=-0.1,  # < 0
                explanation="x",
            )

    def test_metadata_defaults_to_empty_dict(self):
        fr = FlawRegion(
            start=0.0, end=0.5,
            flaw_type=FlawType.ENERGY_LOW,
            severity=0.2,
            explanation="low energy",
        )
        assert fr.metadata == {}

    def test_all_flaw_types_are_valid(self):
        for flaw_type in FlawType:
            fr = FlawRegion(
                start=0.0, end=1.0,
                flaw_type=flaw_type,
                severity=0.5,
                explanation="test",
            )
            assert fr.flaw_type == flaw_type


# ── RubricScore ─────────────────────────────────────────────────────────────

class TestRubricScore:
    def test_valid_score(self):
        rs = RubricScore(dimension=RubricDimension.PACE, score=0.8)
        assert rs.score == 0.8
        assert rs.weight == 1.0

    def test_score_out_of_bounds(self):
        with pytest.raises(ValidationError):
            RubricScore(dimension=RubricDimension.PACE, score=1.1)

    def test_all_dimensions_constructible(self):
        for dim in RubricDimension:
            rs = RubricScore(dimension=dim, score=0.5)
            assert rs.dimension == dim


# ── EvaluationResult ────────────────────────────────────────────────────────

def _make_rubric(scores: list[float]) -> list[RubricScore]:
    dims = list(RubricDimension)
    return [
        RubricScore(dimension=dims[i % len(dims)], score=s)
        for i, s in enumerate(scores)
    ]


def _composite(scores: list[float]) -> float:
    return round(sum(scores) / len(scores), 4)


class TestEvaluationResult:
    def test_valid_result(self):
        rubric = _make_rubric([0.8, 0.6, 0.7, 0.9])
        comp = _composite([0.8, 0.6, 0.7, 0.9])
        er = EvaluationResult(
            audio_id="test",
            transcript="hello",
            duration=10.0,
            rubric_scores=rubric,
            composite_score=comp,
        )
        assert er.composite_score == comp

    def test_composite_mismatch_raises(self):
        rubric = _make_rubric([0.8, 0.6])
        with pytest.raises(ValidationError):
            EvaluationResult(
                audio_id="test",
                transcript="hello",
                duration=10.0,
                rubric_scores=rubric,
                composite_score=0.9999,   # wrong
            )

    def test_empty_rubric_invalid(self):
        with pytest.raises(ValidationError):
            EvaluationResult(
                audio_id="test",
                transcript="hello",
                duration=10.0,
                rubric_scores=[],
                composite_score=0.5,
            )

    def test_json_roundtrip(self):
        rubric = _make_rubric([1.0])
        er = EvaluationResult(
            audio_id="x",
            transcript="t",
            duration=5.0,
            rubric_scores=rubric,
            composite_score=1.0,
        )
        restored = EvaluationResult.model_validate_json(er.model_dump_json())
        assert restored.audio_id == er.audio_id
        assert restored.composite_score == er.composite_score
