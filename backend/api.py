"""HTTP adapter for the speech-evaluation pipeline and optional built frontend."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from src.pipeline import evaluate_with_report

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("speech-eval-api")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"
TEMP_DIR = Path(os.getenv("SPEECH_EVAL_TEMP_DIR", tempfile.gettempdir())) / "speech-eval-uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_BYTES = int(os.getenv("SPEECH_EVAL_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_TRANSCRIPT_CHARS = int(os.getenv("SPEECH_EVAL_MAX_TRANSCRIPT_CHARS", "20000"))
ALIGNMENT_BACKEND = os.getenv("SPEECH_EVAL_ALIGNMENT_BACKEND", "torchaudio")
ALLOW_ALIGNMENT_FALLBACK = os.getenv("SPEECH_EVAL_ALLOW_ALIGNMENT_FALLBACK", "true").lower() in {"1", "true", "yes"}

app = FastAPI(title="Speech Evaluation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_evaluation_lock = asyncio.Semaphore(1)


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return ".wav"
    return suffix


async def _store_upload(upload: UploadFile) -> Path:
    path: Path | None = None
    total = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=_safe_suffix(upload.filename), prefix="upload-", dir=TEMP_DIR, delete=False
        ) as output:
            path = Path(output.name)
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Audio uploads must be no larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB each.",
                    )
                output.write(chunk)
        if total == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio file is empty.")
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def _severity_band(value: float) -> str:
    if value >= 0.80:
        return "critical"
    if value >= 0.60:
        return "high"
    if value >= 0.35:
        return "medium"
    return "low"


def _friendly_flaw_name(value: str) -> str:
    return value.replace("_", " ").title()


def _frontend_result(result: Any, report: Any, candidate_name: str, reference_name: str | None) -> dict[str, Any]:
    audio_id = result.audio_id or str(uuid.uuid4())
    flaws = []
    for index, region in enumerate(result.flaw_regions, start=1):
        flaw_type = region.flaw_type.value
        flaws.append({
            "id": f"{audio_id}-flaw-{index}",
            "type": _friendly_flaw_name(flaw_type),
            "start": region.start,
            "end": region.end,
            "severity": _severity_band(region.severity),
            "explanation": region.explanation,
            "feature": flaw_type.replace("_", " ").title(),
            "metadata": region.metadata,
        })

    return {
        "id": audio_id,
        "status": "complete",
        "overall_score": round(result.composite_score * 100, 1),
        "summary": result.summary,
        "duration_seconds": result.duration,
        "transcript": result.transcript,
        "flaws": flaws,
        "rubric_scores": [
            {
                "name": score.dimension.value.replace("_", " ").title(),
                "score": round(score.score * 100, 1),
                "weight": score.weight,
                "explanation": score.details,
            }
            for score in result.rubric_scores
        ],
        "participant_audio": {"name": candidate_name, "duration_seconds": result.duration},
        "reference_audio": {"name": reference_name} if reference_name else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (["Word alignment used proportional timing because the requested alignment backend was unavailable."] if report.alignment_degraded else []),
        "metadata": {
            "alignment_backend": report.alignment_backend,
            "alignment_degraded": report.alignment_degraded,
            "has_baseline": report.has_baseline,
            "words_total": report.words_total,
            "words_unaligned": report.words_unaligned,
            "stage_seconds": report.stage_seconds,
            "backend_contract": "speech-eval-v1",
        },
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "speech-evaluation"}


@app.get("/api/capabilities")
def capabilities() -> dict[str, Any]:
    return {
        "microphone": False,
        "reference_audio": True,
        "features": ["pace", "pause_pattern", "pitch_variation", "energy_consistency", "temporal_flaw_regions"],
        "max_duration_seconds": 300,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "alignment_backend": ALIGNMENT_BACKEND,
    }


@app.post("/api/evaluate")
async def evaluate(
    participant_audio: UploadFile = File(...),
    transcript: str = Form(...),
    reference_audio: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=422, detail="Please provide the spoken transcript.")
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(status_code=413, detail=f"Transcript must be at most {MAX_TRANSCRIPT_CHARS} characters.")

    candidate_path: Path | None = None
    baseline_path: Path | None = None
    candidate_name = Path(participant_audio.filename or "participant-audio").name
    reference_name = Path(reference_audio.filename or "reference-audio").name if reference_audio else None
    try:
        candidate_path = await _store_upload(participant_audio)
        if reference_audio is not None:
            baseline_path = await _store_upload(reference_audio)

        async with _evaluation_lock:
            result, report = await run_in_threadpool(
                evaluate_with_report,
                candidate_path,
                baseline_path,
                transcript,
                Path(candidate_name).stem[:64] or str(uuid.uuid4()),
                alignment_backend=ALIGNMENT_BACKEND,
                allow_alignment_fallback=ALLOW_ALIGNMENT_FALLBACK,
            )
        return _frontend_result(result, report, candidate_name, reference_name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Speech evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc.__class__.__name__}. Check backend dependencies and alignment-model setup, then retry.") from exc
    finally:
        if candidate_path is not None:
            candidate_path.unlink(missing_ok=True)
        if baseline_path is not None:
            baseline_path.unlink(missing_ok=True)


if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")


@app.get("/{requested_path:path}", include_in_schema=False)
def frontend_or_not_found(requested_path: str) -> Any:
    if requested_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found.")
    candidate = (FRONTEND_DIST / requested_path).resolve()
    if FRONTEND_DIST.is_dir() and candidate.is_relative_to(FRONTEND_DIST.resolve()) and candidate.is_file():
        return FileResponse(candidate)
    index = FRONTEND_DIST / "index.html"
    if index.is_file():
        return FileResponse(index)
    if not requested_path:
        return {"service": "Speech Evaluation API", "docs": "/docs", "health": "/api/health"}
    raise HTTPException(status_code=404, detail="Not found.")
