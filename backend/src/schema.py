"""
src/schema.py
=============
Pydantic v2 data models for the Speech Evaluation System output contract.

All time values are in **seconds**.
Pitch values are in **semitones** relative to the speaker's median F0.
Energy values are **z-scores**.
Severity is a float in [0, 1].
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class FlawType(str, Enum):
    """Taxonomy of detectable speech flaws."""

    PACE_TOO_FAST = "PACE_TOO_FAST"
    PACE_TOO_SLOW = "PACE_TOO_SLOW"
    PITCH_MONOTONE = "PITCH_MONOTONE"
    PITCH_ERRATIC = "PITCH_ERRATIC"
    ENERGY_LOW = "ENERGY_LOW"
    ENERGY_INCONSISTENT = "ENERGY_INCONSISTENT"
    PAUSE_MISSING = "PAUSE_MISSING"
    PAUSE_EXCESSIVE = "PAUSE_EXCESSIVE"
    PAUSE_MISPLACED = "PAUSE_MISPLACED"
    FILLER = "FILLER"
    UNCLEAR = "UNCLEAR"


class FlawRegion(BaseModel):
    """A time-stamped interval where a speech flaw was detected."""

    start: float = Field(..., ge=0.0, description="Region start time in seconds.")
    end: float = Field(..., ge=0.0, description="Region end time in seconds.")
    flaw_type: FlawType = Field(..., description="Taxonomic flaw category.")
    severity: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised severity in [0, 1]."
    )
    explanation: str = Field(
        ..., min_length=1, description="Human-readable causal explanation."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra diagnostic values (optional)."
    )

    @model_validator(mode="after")
    def _end_after_start(self) -> "FlawRegion":
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be strictly greater than start ({self.start})."
            )
        return self


class RubricDimension(str, Enum):
    """Top-level rubric dimensions shown on the score card."""

    PACE = "pace"
    PITCH_VARIATION = "pitch_variation"
    ENERGY_CONSISTENCY = "energy_consistency"
    PAUSE_PATTERN = "pause_pattern"


class RubricScore(BaseModel):
    """Score for a single rubric dimension."""

    dimension: RubricDimension
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Normalised score in [0, 1]; 1 is ideal."
    )
    weight: float = Field(
        default=1.0, gt=0.0, description="Relative weight for composite score."
    )
    details: str = Field(default="", description="One-line rationale for the score.")


class EvaluationResult(BaseModel):
    """Top-level output of the speech evaluation pipeline."""

    audio_id: str = Field(..., description="Identifier for the evaluated audio file.")
    transcript: str = Field(..., description="The spoken transcript text.")
    duration: float = Field(..., gt=0.0, description="Audio duration in seconds.")
    rubric_scores: list[RubricScore] = Field(
        ..., min_length=1, description="One entry per rubric dimension."
    )
    composite_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted average of rubric scores."
    )
    flaw_regions: list[FlawRegion] = Field(
        default_factory=list, description="Time-stamped flaw detections."
    )
    summary: str = Field(default="", description="Overall human-readable summary.")

    @model_validator(mode="after")
    def _composite_matches_rubric(self) -> "EvaluationResult":
        """Composite score must equal the weighted mean of rubric scores."""
        total_weight = sum(r.weight for r in self.rubric_scores)
        expected = sum(r.score * r.weight for r in self.rubric_scores) / total_weight
        if abs(self.composite_score - expected) > 1e-4:
            raise ValueError(
                f"composite_score {self.composite_score:.4f} does not match "
                f"weighted mean of rubric scores {expected:.4f}."
            )
        return self
