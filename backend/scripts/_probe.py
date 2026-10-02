"""Temporary diagnostic probe (not part of the shipped pipeline).

Measures, for every ground-truth flaw region, the raw signals that a
contrastive detector would key on, plus the same signals outside the region
(used as pseudo-negatives). Prints separation statistics per flaw type.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import FLAWED_DIR, HOP_LENGTH, LABELS_DIR, RAW_DIR, SAMPLE_RATE
from src.features.audio import extract_all
from src.features.forced_align import align_transcript
from src.features.spectral import (
    compute_clarity_snr,
    compute_spectral_centroid,
    compute_spectral_flatness,
)
from src.features.transcript import Word
from src.analysis.match import compare_words, match_words

ROWS = list(csv.DictReader((LABELS_DIR / "injections.csv").open(encoding="utf-8")))


def word_pitch_values(word: Word, pitch: np.ndarray) -> np.ndarray:
    fs = int(round(word.start * SAMPLE_RATE / HOP_LENGTH))
    fe = max(fs + 1, int(round(word.end * SAMPLE_RATE / HOP_LENGTH)))
    fs = max(0, min(fs, len(pitch)))
    fe = max(fs, min(fe, len(pitch)))
    v = pitch[fs:fe]
    return v[np.isfinite(v)]


def window_stats(arr: np.ndarray, t0: float, t1: float, derive=("mean", "std")) -> dict:
    fs = int(round(t0 * SAMPLE_RATE / HOP_LENGTH))
    fe = max(fs + 1, int(round(t1 * SAMPLE_RATE / HOP_LENGTH)))
    fs = max(0, min(fs, len(arr)))
    fe = max(fs, min(fe, len(arr)))
    seg = arr[fs:fe]
    seg = seg[np.isfinite(seg)]
    out = {"n": int(seg.size)}
    if seg.size:
        if "mean" in derive:
            out["mean"] = float(np.mean(seg))
        if "std" in derive:
            out["std"] = float(np.std(seg))
    return out


def main() -> None:
    by_type: dict[str, list[dict]] = defaultdict(list)
    neg_by_type: dict[str, list[dict]] = defaultdict(list)

    cache: dict[str, object] = {}

    for row in ROWS:
        fname = row["file"]
        src = row["source"]
        ftype = row["flaw_type"].upper()
        t0, t1 = float(row["start_sec"]), float(row["end_sec"])
        cand_path = FLAWED_DIR / fname
        base_path = RAW_DIR / src
        if not cand_path.exists() or not base_path.exists():
            print(f"MISSING {fname}", file=sys.stderr)
            continue

        transcript = base_path.with_suffix(".txt").read_text(encoding="utf-8-sig").strip()
        if src not in cache:
            cache[src] = extract_all(base_path)
        base = cache[src]  # type: ignore[assignment]

        key = str(cand_path)
        if key not in cache:
            cache[key] = extract_all(cand_path)
        cand = cache[key]  # type: ignore[assignment]

        a_cand = align_transcript(cand_path, transcript, duration=cand.duration)
        a_base = align_transcript(base_path, transcript, duration=base.duration)
        cw = [Word(w["word"], w["start"], w["end"]) for w in a_cand]
        bw = [Word(w["word"], w["start"], w["end"]) for w in a_base]
        matched = match_words(bw, cw)
        comps = compare_words(matched, base.pitch_st, cand.pitch_st, base.energy_z, cand.energy_z)

        centroid_c = compute_spectral_centroid(cand.waveform)
        centroid_b = compute_spectral_centroid(base.waveform)
        flat_c = compute_spectral_flatness(cand.waveform)
        flat_b = compute_spectral_flatness(base.waveform)
        snr_c, clar_c = compute_clarity_snr(cand.waveform)
        snr_b, clar_b = compute_clarity_snr(base.waveform)

        rec: dict = {"file": fname, "sev": row["severity"], "t0": t0, "t1": t1}
        rec["centroid_ratio"] = window_stats(centroid_c, t0, t1)["mean"] / max(
            window_stats(centroid_b, t0, t1)["mean"], 1e-6
        )
        rec["flatness_delta"] = window_stats(flat_c, t0, t1)["mean"] - window_stats(
            flat_b, t0, t1
        )["mean"]
        rec["clarity_cand"] = clar_c
        rec["clarity_base"] = clar_b
        rec["snr_cand"] = snr_c
        rec["snr_base"] = snr_b

        # word-level signals inside vs outside the label window
        in_ratios, out_ratios = [], []
        in_pitch_std, out_pitch_std = [], []
        in_energy_delta, out_energy_delta = [], []
        in_dur_ratio, out_dur_ratio = [], []
        in_pause_delta, out_pause_delta = [], []
        f_start = int(round(t0 * SAMPLE_RATE / HOP_LENGTH))
        f_end = max(f_start + 1, int(round(t1 * SAMPLE_RATE / HOP_LENGTH)))

        for c in comps:
            w = c.candidate
            overlap = min(w.end, t1) - max(w.start, t0)
            inside = overlap > 0.5 * max(w.end - w.start, 1e-6)
            ratios = []
            if not np.isnan(c.pitch_range_ratio):
                ratios.append(c.pitch_range_ratio)
            pv = word_pitch_values(w, cand.pitch_st)
            std = float(np.std(pv)) if pv.size >= 2 else float("nan")
            if inside:
                in_ratios.extend(ratios)
                in_pitch_std.append(std)
                in_energy_delta.append(c.energy_delta)
                in_dur_ratio.append(c.duration_ratio)
                in_pause_delta.append(c.pause_delta_s)
            else:
                out_ratios.extend(ratios)
                out_pitch_std.append(std)
                out_energy_delta.append(c.energy_delta)
                out_dur_ratio.append(c.duration_ratio)
                out_pause_delta.append(c.pause_delta_s)

        rec["in_pitch_ratio"] = [round(float(x), 3) for x in in_ratios]
        rec["out_pitch_ratio_median"] = (
            round(float(np.nanmedian(out_ratios)), 3) if out_ratios else None
        )
        rec["in_pitch_std"] = round(float(np.nanmean(in_pitch_std)), 3) if in_pitch_std else None
        rec["out_pitch_std_median"] = (
            round(float(np.nanmedian([s for s in out_pitch_std if not np.isnan(s)])), 3)
            if any(not np.isnan(s) for s in out_pitch_std)
            else None
        )
        rec["in_energy_delta"] = round(float(np.nanmean(in_energy_delta)), 3) if in_energy_delta else None
        rec["out_energy_delta_median"] = (
            round(float(np.nanmedian([d for d in out_energy_delta if not np.isnan(d)])), 3)
            if any(not np.isnan(d) for d in out_energy_delta)
            else None
        )
        rec["in_dur_ratio"] = round(float(np.nanmean(in_dur_ratio)), 3) if in_dur_ratio else None
        rec["out_dur_ratio_median"] = (
            round(float(np.nanmedian([d for d in out_dur_ratio if not np.isnan(d)])), 3)
            if out_dur_ratio
            else None
        )
        rec["in_pause_delta"] = round(float(np.nanmean(in_pause_delta)), 3) if in_pause_delta else None

        # Gaps (inter-word) that overlap the label window: filler evidence
        gaps = []
        for i in range(len(cw) - 1):
            gs, ge = cw[i].end, cw[i + 1].start
            ov = min(ge, t1) - max(gs, t0)
            if ov <= 0:
                continue
            gd = ge - gs
            if gd <= 0:
                continue
            fs = int(round(gs * SAMPLE_RATE / HOP_LENGTH))
            fe = max(fs + 1, int(round(ge * SAMPLE_RATE / HOP_LENGTH)))
            fs = max(0, min(fs, len(cand.pitch_st)))
            fe = max(fs, min(fe, len(cand.pitch_st)))
            pit = cand.pitch_st[fs:fe]
            en = cand.energy_z[fs:fe]
            voiced = pit[np.isfinite(pit)]
            gaps.append(
                {
                    "dur": round(gd, 3),
                    "energy_z": round(float(np.mean(en)), 3) if en.size else None,
                    "voiced": int(voiced.size),
                    "pitch_std": round(float(np.std(voiced)), 3) if voiced.size >= 2 else None,
                }
            )
        rec["gaps_in_window"] = gaps
        rec["total_gaps"] = sum(1 for i in range(len(cw) - 1) if cw[i + 1].start - cw[i].end >= 0.2)

        by_type[ftype].append(rec)

        # pseudo-negatives: same file, everything outside the window
        neg = dict(rec)
        neg["in_pitch_ratio"] = []
        neg.pop("gaps_in_window", None)
        neg_by_type[ftype].append(neg)

    for ftype in sorted(by_type):
        print(f"\n===== {ftype} ({len(by_type[ftype])} regions) =====")
        for rec in by_type[ftype]:
            print(
                f"  sev{rec['sev']} {rec['file'][:34]:34s} "
                f"pitchRatio={rec['in_pitch_ratio']} "
                f"pitchStd={rec['in_pitch_std']} (out med {rec['out_pitch_std_median']}) "
                f"energyDelta={rec['in_energy_delta']} (out med {rec['out_energy_delta_median']}) "
                f"durRatio={rec['in_dur_ratio']} (out {rec['out_dur_ratio_median']}) "
                f"pauseDelta={rec['in_pause_delta']}"
            )
            print(
                f"      centroidRatio={rec['centroid_ratio']:.3f} flatDelta={rec['flatness_delta']:+.4f} "
                f"snr={rec['snr_cand']} (base {rec['snr_base']}) clarity={rec['clarity_cand']} "
                f"gaps={rec.get('gaps_in_window')}"
            )


if __name__ == "__main__":
    main()
