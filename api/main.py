"""HTTP API for the jury and the local demo UI."""
from __future__ import annotations

import base64
import hmac
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from bvd.config import CFG  # noqa: E402
from bvd.predictor import Predictor  # noqa: E402

logger = logging.getLogger(__name__)
app = FastAPI(title="Bank cash-desk violation detector", version="2.0")

MAX_UPLOAD_BYTES = 100 * 1024 * 1024
JOB_TTL_SEC = 60 * 60
MAX_RETAINED_JOBS = 256
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_predictor: Predictor | None = None
_predictor_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-analysis")


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    configured_key = os.environ.get("JURY_API_KEY", "")
    if not configured_key:
        raise HTTPException(status_code=503, detail="API key is not configured")
    if not x_api_key or not hmac.compare_digest(x_api_key, configured_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = Predictor()
    return _predictor


def _prune_jobs(now: float) -> None:
    expired = [
        job_id for job_id, job in _jobs.items()
        if job["status"] != "processing" and now - job["created_at"] > JOB_TTL_SEC
    ]
    for job_id in expired:
        del _jobs[job_id]

    if len(_jobs) >= MAX_RETAINED_JOBS:
        completed = sorted(
            ((job["created_at"], job_id) for job_id, job in _jobs.items()
             if job["status"] != "processing")
        )
        while len(_jobs) >= MAX_RETAINED_JOBS and completed:
            _, job_id = completed.pop(0)
            _jobs.pop(job_id, None)


def _jury_result(job_id: str, prediction: dict) -> dict:
    verdict = "violation" if prediction["violation"] else "no_violation"
    violations = []
    if prediction["violation"]:
        evidence = [
            {"timestamp_sec": round(t, 1), "image_base64": b64}
            for t, img in prediction.get("evidence_rgb", [])[:3]
            if (b64 := Predictor.encode_jpg(img))
        ]
        violations.append({
            "type": prediction.get("violation_type_ru") or prediction.get("violation_type") or "Возможное нарушение",
            "timecode_sec": prediction["timestamp_sec"],
            "description": prediction.get("description") or "",
            "evidence_frames": evidence,
        })
    return {
        "job_id": job_id,
        "status": "done",
        "verdict": verdict,
        "violations": violations,
    }


def _analyze_job(job_id: str, video_path: str) -> None:
    try:
        prediction = get_predictor().predict(video_path)
        result = _jury_result(job_id, prediction)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job.update(status="done", result=result)
    except Exception:
        logger.exception("Video analysis failed for job %s", job_id)
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job.update(status="failed", error="Video analysis failed")
    finally:
        Path(video_path).unlink(missing_ok=True)


async def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "video.mov").suffix.lower()
    if suffix != ".mov":
        raise HTTPException(status_code=400, detail="Only .mov video files are accepted")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mov")
    temp_path = temp.name
    size = 0
    try:
        with temp:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 100 MB limit")
                temp.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video is empty")
        return temp_path
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


@app.get("/health")
def health():
    return {"status": "ok", "model": CFG.model_path.exists()}


@app.post("/api/v1/analyze", status_code=202, dependencies=[Depends(require_api_key)])
async def analyze(file: UploadFile = File(...)):
    video_path = await _save_upload(file)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _prune_jobs(time.time())
        if len(_jobs) >= MAX_RETAINED_JOBS:
            Path(video_path).unlink(missing_ok=True)
            raise HTTPException(status_code=503, detail="Analysis queue is full; retry later")
        _jobs[job_id] = {"status": "processing", "created_at": time.time()}
    try:
        _executor.submit(_analyze_job, job_id, video_path)
    except Exception:
        Path(video_path).unlink(missing_ok=True)
        with _jobs_lock:
            _jobs.pop(job_id, None)
        raise HTTPException(status_code=503, detail="Could not queue video for analysis")
    return {"job_id": job_id}


@app.get("/api/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str):
    with _jobs_lock:
        _prune_jobs(time.time())
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        if job["status"] == "done":
            return job["result"]
        response = {"job_id": job_id, "status": job["status"]}
        if job["status"] == "failed":
            response["error"] = job.get("error", "Video analysis failed")
        return response


@app.post("/predict", dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile = File(...)):
    """Synchronous compatibility endpoint; also requires the jury API key."""
    video_path = await _save_upload(file)
    try:
        prediction = get_predictor().predict(video_path)
        return Predictor.to_json(prediction)
    except (OSError, IOError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        Path(video_path).unlink(missing_ok=True)
