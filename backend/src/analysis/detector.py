"""
src/analysis/detector.py
========================
Flaw-region detection: scan aligned feature streams for time-stamped flaws.

Detectors:
- detect_pace_flaws        – fast/slow segments based on local WPM
- detect_pitch_flaws       – monotone or erratic pitch windows
- detect_energy_flaws      – low or inconsistent energy windows
- detect_pause_flaws       – missing, excessive, or misplaced pauses
- detect_all               – run all detectors and return merged list
"""
from __future__ import annotations

import numpy as np

from src.config import (
    ENERGY_Z_CLIP,
    FILLER_ENERGY_Z_MAX,
    FILLER_MIN_DURATION,
    FILLER_PITCH_STD_MAX,
    HOP_LENGTH,
    FLAW_MERGE_GAP_SEC,
    MIN_FLAW_DURATION_SEC,
    REGION_MERGE_EPS_S,
    MONOTONE_MIN_REGION_SEC,
    PACE_IDEAL_WPM_MAX,
    PACE_IDEAL_WPM_MIN,
    PAUSE_IDEAL_MAX,
    PITCH_VAR_IDEAL_MAX,
    PITCH_VAR_IDEAL_MIN,
    SAMPLE_RATE,
)
from src.features.transcript import Word
from src.schema import FlawRegion, FlawType


def merge_and_filter_regions(
    regions: list[FlawRegion],
    *,
    merge_gap: float = FLAW_MERGE_GAP_SEC,
    min_duration: float = MIN_FLAW_DURATION_SEC,
) -> list[FlawRegion]:
    """Merge nearby same-type regions and discard imperceptibly short spans."""
    merged: list[FlawRegion] = []
    for region in sorted(regions, key=lambda r: (r.flaw_type.value, r.start)):
        if (
            merged
            and merged[-1].flaw_type == region.flaw_type
            # The epsilon keeps float noise (measured gap 0.2000000000000028 s)
            # from splitting a single flaw into two adjacent regions.
            and region.start - merged[-1].end <= merge_gap + REGION_MERGE_EPS_S
        ):
            previous = merged[-1]
            # Preserve the explanation/metadata of the more severe detection.
            source = region if region.severity > previous.severity else previous
            merged[-1] = source.model_copy(
                update={
                    "start": min(previous.start, region.start),
                    "end": max(previous.end, region.end),
                    "severity": max(previous.severity, region.severity),
                }
            )
        else:
            merged.append(region)
    return [r for r in merged if r.end - r.start >= min_duration - REGION_MERGE_EPS_S]


def _severity(deviation: float, scale: float = 1.0) -> float:
    """Map a raw deviation value to [0, 1] via tanh.

    Args:
        deviation: Non-negative deviation from the ideal boundary.
        scale:     Scaling factor before tanh (larger → steeper).

    Returns:
        Severity in [0, 1].
    """
    return float(np.clip(np.tanh(deviation * scale), 0.0, 1.0))


def _frames_to_seconds(frame_idx: int) -> float:
    """Convert a frame index to time in seconds."""
    return float(frame_idx * HOP_LENGTH / SAMPLE_RATE)


def detect_pace_flaws(words: list[Word]) -> list[FlawRegion]:
    """Detect pace flaws using a sliding window over word timings.

    A window of 10 consecutive words is scanned; if the local WPM is
    outside the ideal range, a FlawRegion is created.

    Args:
        words: List of Word objects with start/end timing.

    Returns:
        List of FlawRegion for pace flaws.
    """
    from src.explain.templates import explain  # local import to avoid circular

    flaws: list[FlawRegion] = []
    window = 10
    step = 5

    for i in range(0, max(1, len(words) - window + 1), step):
        chunk = words[i : i + window]
        if len(chunk) < 2:
            continue
        span_s = chunk[-1].end - chunk[0].start
        if span_s <= 0:
            continue
        wpm = len(chunk) / (span_s / 60.0)

        if wpm > PACE_IDEAL_WPM_MAX:
            deviation = (wpm - PACE_IDEAL_WPM_MAX) / PACE_IDEAL_WPM_MAX
            sev = _severity(deviation, scale=2.0)
            flaw_type = FlawType.PACE_TOO_FAST
        elif wpm < PACE_IDEAL_WPM_MIN:
            deviation = (PACE_IDEAL_WPM_MIN - wpm) / PACE_IDEAL_WPM_MIN
            sev = _severity(deviation, scale=2.0)
            flaw_type = FlawType.PACE_TOO_SLOW
        else:
            continue

        flaws.append(
            FlawRegion(
                start=round(chunk[0].start, 3),
                end=round(chunk[-1].end, 3),
                flaw_type=flaw_type,
                severity=round(sev, 4),
                explanation=explain(flaw_type, sev, {"wpm": wpm}),
                metadata={"wpm": round(wpm, 1)},
            )
        )
    return merge_and_filter_regions(flaws)


def detect_pitch_flaws(
    pitch_st: np.ndarray,
    window_frames: int = 50,
    step_frames: int = 25,
) -> list[FlawRegion]:
    """Detect monotone or erratic pitch over sliding windows.

    Args:
        pitch_st:      Per-frame pitch in semitones.
        window_frames: Window size in frames.
        step_frames:   Step size in frames.

    Returns:
        List of FlawRegion for pitch flaws.
    """
    from src.explain.templates import explain  # local import

    flaws: list[FlawRegion] = []

    # Split on unvoiced frames.  NaNs are neither zero-pitch nor low-variance
    # samples, so a monotone run can contain voiced frames only.
    voiced_indices = np.flatnonzero(np.isfinite(pitch_st))
    if voiced_indices.size == 0:
        return flaws
    voiced_runs = np.split(voiced_indices, np.where(np.diff(voiced_indices) > 1)[0] + 1)

    for run_indices in voiced_runs:
        if run_indices.size < 5:
            continue
        run_window = min(window_frames, run_indices.size)
        for offset in range(0, run_indices.size - run_window + 1, step_frames):
            frame_indices = run_indices[offset : offset + run_window]
            voiced = pitch_st[frame_indices]
            std = float(np.std(voiced))
            start_s = _frames_to_seconds(int(frame_indices[0]))
            end_s = _frames_to_seconds(int(frame_indices[-1]) + 1)

            if std < PITCH_VAR_IDEAL_MIN:
                deviation = (PITCH_VAR_IDEAL_MIN - std) / max(PITCH_VAR_IDEAL_MIN, 1e-6)
                sev = _severity(deviation, scale=2.5)
                flaw_type = FlawType.PITCH_MONOTONE
            elif std > PITCH_VAR_IDEAL_MAX:
                deviation = (std - PITCH_VAR_IDEAL_MAX) / max(PITCH_VAR_IDEAL_MAX, 1e-6)
                sev = _severity(deviation, scale=1.5)
                flaw_type = FlawType.PITCH_ERRATIC
            else:
                continue
            flaws.append(FlawRegion(
                start=round(start_s, 3), end=round(end_s, 3), flaw_type=flaw_type,
                severity=round(sev, 4),
                explanation=explain(flaw_type, sev, {"pitch_std_st": std}),
                metadata={"pitch_std_st": round(std, 3)},
            ))
    return merge_and_filter_regions(
        flaws,
        min_duration=MONOTONE_MIN_REGION_SEC,
    )


def detect_energy_flaws(
    energy_z: np.ndarray,
    window_frames: int = 50,
    step_frames: int = 25,
) -> list[FlawRegion]:
    """Detect low or inconsistent energy over sliding windows.

    Args:
        energy_z:      Per-frame energy z-scores.
        window_frames: Window size in frames.
        step_frames:   Step size in frames.

    Returns:
        List of FlawRegion for energy flaws.
    """
    from src.explain.templates import explain  # local import

    flaws: list[FlawRegion] = []
    clipped = np.clip(energy_z, -ENERGY_Z_CLIP, ENERGY_Z_CLIP)

    for i in range(0, max(1, len(clipped) - window_frames + 1), step_frames):
        window = clipped[i : i + window_frames]
        mean_e = float(np.mean(window))
        std_e = float(np.std(window))

        start_s = _frames_to_seconds(i)
        end_s = _frames_to_seconds(i + len(window))

        if mean_e < -1.5:
            sev = _severity(abs(mean_e) - 1.5, scale=0.8)
            flaws.append(
                FlawRegion(
                    start=round(start_s, 3),
                    end=round(end_s, 3),
                    flaw_type=FlawType.ENERGY_LOW,
                    severity=round(sev, 4),
                    explanation=explain(
                        FlawType.ENERGY_LOW, sev, {"mean_energy_z": mean_e}
                    ),
                    metadata={"mean_energy_z": round(mean_e, 3)},
                )
            )
        elif std_e > 2.0:
            sev = _severity(std_e - 2.0, scale=0.5)
            flaws.append(
                FlawRegion(
                    start=round(start_s, 3),
                    end=round(end_s, 3),
                    flaw_type=FlawType.ENERGY_INCONSISTENT,
                    severity=round(sev, 4),
                    explanation=explain(
                        FlawType.ENERGY_INCONSISTENT, sev, {"energy_std_z": std_e}
                    ),
                    metadata={"energy_std_z": round(std_e, 3)},
                )
            )
    return merge_and_filter_regions(flaws)


def detect_pause_flaws(
    pauses: list[tuple[float, float]],
    duration: float,
) -> list[FlawRegion]:
    """Detect pause-related flaws (excessive duration or misplaced gaps).

    Args:
        pauses:   List of (start_s, end_s) detected pause intervals.
        duration: Total audio duration in seconds.

    Returns:
        List of FlawRegion for pause flaws.
    """
    from src.explain.templates import explain  # local import

    flaws: list[FlawRegion] = []

    for start_s, end_s in pauses:
        pause_dur = end_s - start_s
        if pause_dur > PAUSE_IDEAL_MAX:
            deviation = (pause_dur - PAUSE_IDEAL_MAX) / PAUSE_IDEAL_MAX
            sev = _severity(deviation, scale=1.5)
            flaws.append(
                FlawRegion(
                    start=round(start_s, 3),
                    end=round(end_s, 3),
                    flaw_type=FlawType.PAUSE_EXCESSIVE,
                    severity=round(sev, 4),
                    explanation=explain(
                        FlawType.PAUSE_EXCESSIVE,
                        sev,
                        {"pause_duration_s": round(pause_dur, 2)},
                    ),
                    metadata={"pause_duration_s": round(pause_dur, 3)},
                )
            )

    return merge_and_filter_regions(flaws)


def detect_filler_flaws(
    words: list[Word],
    pitch_st: np.ndarray,
    energy_z: np.ndarray,
) -> list[FlawRegion]:
    """Detect probable filler sounds (uh, um) in inter-word gaps.

    A gap between two consecutive words is classified as a FILLER region when:
      1. Gap duration >= FILLER_MIN_DURATION (filters out normal co-articulation).
      2. Mean energy z-score in the gap < FILLER_ENERGY_Z_MAX (low energy).
      3. Voiced pitch std in the gap < FILLER_PITCH_STD_MAX (flat, steady pitch).

    Condition 2 ensures silence is excluded (silence has very low or zero energy
    but also no voiced pitch); condition 3 confirms voiced activity without the
    intonation variability of real speech content.

    Args:
        words:    List of Word objects with start/end timing.
        pitch_st: Full-utterance pitch in semitones (NaN = unvoiced).
        energy_z: Full-utterance energy z-score.

    Returns:
        List of FlawRegion with flaw_type=FILLER.
    """
    from src.explain.templates import explain  # local import to avoid circular

    flaws: list[FlawRegion] = []

    for i in range(len(words) - 1):
        gap_start = words[i].end
        gap_end = words[i + 1].start
        gap_dur = gap_end - gap_start
        if gap_dur < FILLER_MIN_DURATION:
            continue

        # Convert gap to frame indices
        f_start = int(round(gap_start * SAMPLE_RATE / HOP_LENGTH))
        f_end = int(round(gap_end * SAMPLE_RATE / HOP_LENGTH))
        f_start = max(0, min(f_start, len(pitch_st)))
        f_end = max(f_start, min(f_end, len(pitch_st)))

        if f_end <= f_start:
            continue

        gap_pitch = pitch_st[f_start:f_end]
        gap_energy = energy_z[f_start:f_end]

        mean_energy = float(np.mean(gap_energy))
        voiced = gap_pitch[np.isfinite(gap_pitch)]
        pitch_std = float(np.std(voiced)) if voiced.size >= 2 else float("nan")

        # Condition: low energy AND voiced AND flat pitch
        if mean_energy >= FILLER_ENERGY_Z_MAX:
            continue
        if voiced.size == 0:
            continue  # purely silent gap — not a filler sound
        if np.isnan(pitch_std) or pitch_std >= FILLER_PITCH_STD_MAX:
            continue

        sev = _severity(FILLER_MIN_DURATION / max(gap_dur, 1e-6), scale=1.0)
        sev = round(float(np.clip(sev, 0.0, 1.0)), 4)

        flaws.append(
            FlawRegion(
                start=round(gap_start, 3),
                end=round(gap_end, 3),
                flaw_type=FlawType.FILLER,
                severity=sev,
                explanation=explain(
                    FlawType.FILLER, sev, {"gap_duration_s": round(gap_dur, 2)}
                ),
                metadata={
                    "gap_duration_s": round(gap_dur, 3),
                    "mean_energy_z": round(mean_energy, 3),
                    "pitch_std_st": round(pitch_std, 3) if not np.isnan(pitch_std) else None,
                },
            )
        )
    return flaws


def detect_all(
    pitch_st: np.ndarray,
    energy_z: np.ndarray,
    pauses: list[tuple[float, float]],
    words: list[Word],
    duration: float,
) -> list[FlawRegion]:
    """Run all detectors and return a time-sorted merged list of FlawRegions.

    Args:
        pitch_st: Per-frame pitch array (semitones).
        energy_z: Per-frame energy array (z-score).
        pauses:   Detected pause intervals.
        words:    Word-level timing objects.
        duration: Audio duration in seconds.

    Returns:
        Chronologically sorted list of FlawRegion objects.
    """
    all_flaws: list[FlawRegion] = []
    all_flaws.extend(detect_pace_flaws(words))
    all_flaws.extend(detect_pitch_flaws(pitch_st))
    all_flaws.extend(detect_energy_flaws(energy_z))
    all_flaws.extend(detect_pause_flaws(pauses, duration))
    all_flaws.extend(detect_filler_flaws(words, pitch_st, energy_z))
    all_flaws.sort(key=lambda r: r.start)
    return all_flaws
