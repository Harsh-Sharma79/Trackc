"""
src/explain/templates.py
========================
Template-driven human-readable explanation generator.

Maps (FlawType, severity, metadata) → a clear, causal English sentence.
All templates are pure string operations – no LLM calls, no I/O.

Public API:
- explain(flaw_type, severity, metadata) -> str
"""
from __future__ import annotations

from src.schema import FlawType

# Severity label thresholds
_SEVERITY_LABEL = [
    (0.75, "severely"),
    (0.45, "noticeably"),
    (0.20, "slightly"),
    (0.0,  "marginally"),
]


def _severity_adverb(severity: float) -> str:
    """Return an adverb that qualifies the severity of a flaw."""
    for threshold, label in _SEVERITY_LABEL:
        if severity >= threshold:
            return label
    return "marginally"


_TEMPLATES: dict[FlawType, str] = {
    FlawType.PACE_TOO_FAST: (
        "The speaker is {adverb} rushing: the local pace of {wpm:.0f} WPM exceeds the "
        "ideal ceiling of {ideal_max:.0f} WPM, making it harder for listeners to follow."
    ),
    FlawType.PACE_TOO_SLOW: (
        "The speaker is {adverb} dragging: the local pace of {wpm:.0f} WPM falls below "
        "the ideal floor of {ideal_min:.0f} WPM, risking listener disengagement."
    ),
    FlawType.PITCH_MONOTONE: (
        "The speaker sounds {adverb} monotone in this segment: pitch variation is only "
        "{pitch_std_st:.2f} st (ideal ≥ {ideal_min:.1f} st), conveying little expressiveness."
    ),
    FlawType.PITCH_ERRATIC: (
        "Pitch changes are {adverb} erratic here: std-dev of {pitch_std_st:.2f} st exceeds "
        "the ideal ceiling of {ideal_max:.1f} st, which can distract listeners."
    ),
    FlawType.ENERGY_LOW: (
        "The speaker is {adverb} too quiet in this segment: mean energy z-score is "
        "{mean_energy_z:.2f}, well below the neutral baseline."
    ),
    FlawType.ENERGY_INCONSISTENT: (
        "Volume is {adverb} inconsistent here: energy std-dev of {energy_std_z:.2f} "
        "indicates erratic loudness swings."
    ),
    FlawType.PAUSE_MISSING: (
        "A {adverb} noticeable lack of pauses was detected in this region; adding "
        "deliberate pauses would improve listener comprehension."
    ),
    FlawType.PAUSE_EXCESSIVE: (
        "A {adverb} long pause of {pause_duration_s:.1f} s was detected, exceeding the "
        "ideal maximum of {ideal_max:.1f} s; this may signal hesitation or loss of flow."
    ),
    FlawType.PAUSE_MISPLACED: (
        "A {adverb} poorly placed pause was detected here; the break does not coincide "
        "with a natural sentence boundary."
    ),
    FlawType.FILLER: (
        "The speaker used a {adverb} distracting filler sound or word here, which disrupts "
        "the flow of speech."
    ),
    FlawType.UNCLEAR: (
        "Speech clarity is {adverb} degraded in this segment, making words difficult to distinguish."
    ),
}


def explain(
    flaw_type: FlawType,
    severity: float,
    metadata: dict | None = None,
) -> str:
    """Generate a human-readable causal explanation for a detected flaw.

    Args:
        flaw_type: The type of speech flaw.
        severity:  Severity score in [0, 1].
        metadata:  Optional dict of extra values referenced by the template
                   (e.g. {"wpm": 195.3}).

    Returns:
        Formatted explanation string.
    """
    from src.config import (
        PACE_IDEAL_WPM_MAX,
        PACE_IDEAL_WPM_MIN,
        PAUSE_IDEAL_MAX,
        PITCH_VAR_IDEAL_MAX,
        PITCH_VAR_IDEAL_MIN,
    )

    template = _TEMPLATES.get(flaw_type, "A speech flaw was detected: {flaw_type}.")
    ctx: dict = {
        "adverb": _severity_adverb(severity),
        "flaw_type": flaw_type.value,
        "ideal_min": PACE_IDEAL_WPM_MIN,
        "ideal_max": PACE_IDEAL_WPM_MAX,
        # override ideal_max/ideal_min for pitch
        **(
            {"ideal_min": PITCH_VAR_IDEAL_MIN, "ideal_max": PITCH_VAR_IDEAL_MAX}
            if flaw_type in (FlawType.PITCH_MONOTONE, FlawType.PITCH_ERRATIC)
            else {}
        ),
        # override for pause
        **(
            {"ideal_max": PAUSE_IDEAL_MAX}
            if flaw_type in (FlawType.PAUSE_EXCESSIVE, FlawType.PAUSE_MISSING)
            else {}
        ),
    }
    if metadata:
        ctx.update(metadata)

    try:
        return template.format_map(ctx)
    except KeyError as exc:
        return (
            f"[Explanation unavailable – missing template key {exc}] "
            f"Flaw: {flaw_type.value}, severity: {severity:.2f}."
        )
