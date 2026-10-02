"""
src/features/forced_align.py
============================
Forced alignment module supporting WhisperX, torchaudio MMS_FA, and
proportional fallback. Caches results as JSON in data/labels/alignments/.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.config import (
    ALIGNMENT_BACKEND,
    ALIGNMENT_CACHE_DIR,
    SAMPLE_RATE,
)
from src.features.audio import load_audio
from src.features.transcript import estimate_word_times, tokenize

logger = logging.getLogger(__name__)


class AlignedWord(TypedDict):
    """Word boundary timing output contract."""
    word: str
    start: float
    end: float
    aligned: bool


def compute_alignment_hash(audio_path: str | Path, transcript: str) -> str:
    """Compute a deterministic SHA-256 hash from audio content and transcript text."""
    p = Path(audio_path)
    hasher = hashlib.sha256()
    if p.exists():
        hasher.update(p.read_bytes())
    else:
        hasher.update(str(p).encode("utf-8"))
    hasher.update(transcript.strip().encode("utf-8"))
    return hasher.hexdigest()[:24]


def interpolate_unaligned_words(
    words: list[str],
    raw_aligned: list[dict[str, Any]],
    duration: float,
) -> list[AlignedWord]:
    """Fill timing gaps for words missed by acoustic aligner using proportional interpolation.

    Ensures every transcript word has a non-empty, strictly monotonic interval.
    Unaligned words have aligned=False.
    """
    if not words:
        return []

    # Map raw aligned words by index/order
    results: list[AlignedWord] = []
    # Match raw_aligned words sequentially to transcript words
    raw_idx = 0
    n_raw = len(raw_aligned)

    aligned_slots: list[dict[str, Any] | None] = [None] * len(words)
    raw_ptr = 0
    for i, w in enumerate(words):
        w_clean = w.lower().strip()
        for j in range(raw_ptr, n_raw):
            cand = raw_aligned[j]
            cand_w = str(cand.get("word", "")).lower().strip()
            if cand_w == w_clean and "start" in cand and "end" in cand:
                aligned_slots[i] = cand
                raw_ptr = j + 1
                break


    # Now interpolate gaps between matched anchors
    n = len(words)
    last_end = 0.0

    i = 0
    while i < n:
        if aligned_slots[i] is not None:
            anchor = aligned_slots[i]
            s = float(anchor["start"])
            e = float(anchor["end"])
            # Ensure monotonicity
            s = max(s, last_end)
            e = max(e, s + 0.01)
            results.append(AlignedWord(word=words[i], start=round(s, 4), end=round(e, 4), aligned=True))
            last_end = e
            i += 1
        else:
            # Find span of unaligned words: [i, next_anchor_idx)
            next_anchor_idx = i
            while next_anchor_idx < n and aligned_slots[next_anchor_idx] is None:
                next_anchor_idx += 1

            gap_start = last_end
            if next_anchor_idx < n and aligned_slots[next_anchor_idx] is not None:
                gap_end = float(aligned_slots[next_anchor_idx]["start"])
                if gap_end <= gap_start + 0.05:
                    gap_end = gap_start + 0.1 * (next_anchor_idx - i)
            else:
                gap_end = max(duration, gap_start + 0.1 * (next_anchor_idx - i))

            unaligned_sub = words[i:next_anchor_idx]
            sub_times = estimate_word_times(
                unaligned_sub,
                duration=gap_end,
                speech_start=gap_start,
                speech_end=gap_end,
            )
            for wt in sub_times:
                results.append(
                    AlignedWord(
                        word=wt.text,
                        start=round(wt.start, 4),
                        end=round(wt.end, 4),
                        aligned=False,
                    )
                )
            last_end = gap_end
            i = next_anchor_idx

    return results


def _align_with_whisperx(
    audio_path: str | Path,
    words: list[str],
    transcript: str,
    duration: float,
) -> list[AlignedWord]:
    """Align using WhisperX forced alignment model.

    Raises:
        ImportError: If whisperx is not installed.
        RuntimeError: If whisperx alignment fails.
    """
    try:
        import whisperx  # noqa: F401
    except ImportError:
        raise ImportError(
            "whisperx is not installed. Install it or set ALIGNMENT_BACKEND='torchaudio'."
        )

    device = "cpu"
    align_model, metadata = whisperx.load_align_model(
        language_code="en",
        device=device,
    )
    audio = whisperx.load_audio(str(audio_path))
    fake_transcript = [{"text": transcript, "start": 0.0, "end": duration}]
    result = whisperx.align(
        fake_transcript,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    raw_words: list[dict[str, Any]] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                raw_words.append(w)

    return interpolate_unaligned_words(words, raw_words, duration)


_MMS_FA_MODEL = None
_MMS_FA_DICT = None


def _get_mms_fa_components():
    global _MMS_FA_MODEL, _MMS_FA_DICT
    import torchaudio
    if _MMS_FA_MODEL is None:
        bundle = torchaudio.pipelines.MMS_FA
        _MMS_FA_MODEL = bundle.get_model()
        _MMS_FA_DICT = bundle.get_dict()
    return _MMS_FA_MODEL, _MMS_FA_DICT, torchaudio.pipelines.MMS_FA.sample_rate


def _align_with_torchaudio(
    audio_path: str | Path,
    words: list[str],
    duration: float,
) -> list[AlignedWord]:
    """Align using Torchaudio MMS_FA forced alignment pipeline.

    Uses blank=0 (not blank_id=0 which is not a valid parameter).
    Character-level token spans are grouped back into per-word intervals by
    tracking how many tokens each word contributes.

    inspect.signature(torchaudio.functional.forced_align):
        (log_probs, targets, input_lengths=None, target_lengths=None, blank=0)

    Raises:
        RuntimeError: On any torchaudio alignment failure.
    """
    import torch
    import torchaudio

    model, dictionary, target_sr = _get_mms_fa_components()

    waveform, sr = torchaudio.load(str(audio_path))
    # Convert stereo → mono
    if waveform.ndim > 1 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    # Build word→token mapping, preserving only words with at least 1 known char
    BLANK_CHAR = "-"
    word_token_lists: list[list[int]] = []
    aligned_words_texts: list[str] = []
    all_tokens: list[int] = []

    for w in words:
        chars = [c for c in w.lower() if c in dictionary and c != BLANK_CHAR]
        if chars:
            toks = [dictionary[c] for c in chars]
            word_token_lists.append(toks)
            aligned_words_texts.append(w)
            all_tokens.extend(toks)

    if not all_tokens:
        raise RuntimeError("No mappable characters found in transcript for torchaudio MMS_FA alignment.")

    with torch.inference_mode():
        emission, _ = model(waveform)
        targets = torch.tensor([all_tokens], dtype=torch.int32)
        # FIX: use blank=0, not blank_id=0
        aligned_tokens, scores = torchaudio.functional.forced_align(
            emission, targets, blank=0
        )

    # merge_tokens: merges consecutive identical tokens into spans (char-level)
    token_spans = torchaudio.functional.merge_tokens(aligned_tokens[0], scores[0], blank=0)

    # Group character-level spans back into per-word time intervals
    num_frames = emission.size(1)
    time_per_frame = duration / max(num_frames, 1)

    raw_words: list[dict[str, Any]] = []
    span_idx = 0
    for w, toks in zip(aligned_words_texts, word_token_lists):
        n = len(toks)
        spans_w = token_spans[span_idx: span_idx + n]
        span_idx += n
        if spans_w:
            raw_words.append({
                "word": w,
                "start": spans_w[0].start * time_per_frame,
                "end": spans_w[-1].end * time_per_frame,
            })

    return interpolate_unaligned_words(words, raw_words, duration)


def _align_proportional(
    words: list[str],
    duration: float,
) -> list[AlignedWord]:
    """Deterministic character-proportional timing fallback."""
    proportional_words = estimate_word_times(words, duration=duration)
    return [
        AlignedWord(
            word=pw.text,
            start=round(pw.start, 4),
            end=round(pw.end, 4),
            aligned=False,
        )
        for pw in proportional_words
    ]


def align_transcript(
    audio_path: str | Path,
    transcript: str,
    duration: float | None = None,
    backend: str | None = None,
    use_cache: bool = True,
) -> list[AlignedWord]:
    """Force-align audio and transcript to return exact word boundaries.

    Results are cached in data/labels/alignments/ by audio-content + transcript hash.

    Backend semantics (Bug 3 fix — NO silent fallbacks):
    - An explicitly selected backend that fails RAISES with the real error.
    - "proportional" is only used when ALIGNMENT_BACKEND == "proportional" or
      backend parameter is "proportional".
    - "whisperx" raises ImportError immediately if whisperx is not installed.
    - Cache is only written after a SUCCESSFUL alignment (not on fallback or error).

    Args:
        audio_path: Path to the target audio file.
        transcript: Full text transcript.
        duration: Optional audio duration in seconds.
        backend: "whisperx" | "torchaudio" | "proportional" (default from config).
        use_cache: Whether to read/write cached JSON alignments.

    Returns:
        List of AlignedWord dicts with keys {word, start, end, aligned}.
    """
    if duration is None:
        waveform = load_audio(audio_path)
        duration = float(len(waveform) / SAMPLE_RATE)

    words = tokenize(transcript)
    if not words:
        return []

    # Check cache
    ALIGNMENT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    file_hash = compute_alignment_hash(audio_path, transcript)
    cache_path = ALIGNMENT_CACHE_DIR / f"{file_hash}.json"

    if use_cache and cache_path.exists():
        try:
            cached_data = json.loads(cache_path.read_text(encoding="utf-8"))
            return [
                AlignedWord(
                    word=item["word"],
                    start=float(item["start"]),
                    end=float(item["end"]),
                    aligned=bool(item["aligned"]),
                )
                for item in cached_data
            ]
        except Exception as e:
            # Cache file is corrupt: delete it and re-align
            logger.warning("Corrupt alignment cache %s, deleting: %s", cache_path, e)
            cache_path.unlink(missing_ok=True)

    chosen_backend = backend or ALIGNMENT_BACKEND

    # Validate backend value early
    valid_backends = {"whisperx", "torchaudio", "proportional"}
    if chosen_backend not in valid_backends:
        raise ValueError(
            f"Unknown ALIGNMENT_BACKEND {chosen_backend!r}. Choose from {valid_backends}."
        )

    # Dispatch — NO silent fallbacks; failures propagate
    aligned_words: list[AlignedWord]
    if chosen_backend == "whisperx":
        # Raises ImportError if not installed, RuntimeError on failure
        aligned_words = _align_with_whisperx(audio_path, words, transcript, duration)
    elif chosen_backend == "torchaudio":
        # Raises RuntimeError on failure — no fallback to proportional
        aligned_words = _align_with_torchaudio(audio_path, words, duration)
    else:
        # chosen_backend == "proportional"
        aligned_words = _align_proportional(words, duration)

    # Write cache only on success
    if use_cache:
        try:
            cache_path.write_text(json.dumps(aligned_words, indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to write alignment cache %s: %s", cache_path, e)

    return aligned_words
