"""Extraction API (Phases 5 and 8).

    uvicorn api.app:app --host 0.0.0.0 --port 8000

    GET  /health          liveness + configuration + which models are wired
    POST /extract         one file (multipart "file")  -> ExtractionOut
    POST /extract/batch   many files (multipart "files") -> {results, summary}
    GET  /metrics         in-process counters: failures, abstention, escalation, $, latency

Pipeline calls are blocking network I/O, so handlers run them in the threadpool rather
than on the event loop. One bounded semaphore caps concurrent extractions across BOTH
endpoints: the free tier allows ~10 requests/minute, and unbounded fan-out only converts
load into 429 retries and timeouts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from api import service
from api.batch import summarize_results
from api.schemas import ExtractionOut, from_error
from api.telemetry import Telemetry

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20")) * 1024 * 1024
MAX_BATCH_FILES = int(os.environ.get("MAX_BATCH_FILES", "20"))
MAX_CONCURRENCY = int(os.environ.get("API_MAX_CONCURRENCY", "3"))
# A threading semaphore, taken inside the worker thread: an asyncio.Semaphore created at
# import binds to whichever event loop first contends for it, which breaks under test
# clients and reloaders that each run their own loop.
_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)
telemetry = Telemetry()

app = FastAPI(
    title="Receipt extraction",
    version="0.8.0",
    description="VLM reader + text-LLM reasoner with deterministic validation, "
                "per-field confidence and abstention.",
)


class BatchOut(BaseModel):
    results: list[ExtractionOut]
    summary: dict


class UploadRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _extract_bounded(path: Path, doc_id: str) -> ExtractionOut:
    with _slots:
        return service.extract_document(path, doc_id)


def _require_config() -> None:
    missing = service.config_problems()["missing"]
    if missing:
        telemetry.reject()
        raise HTTPException(503, f"service not configured: {', '.join(missing)} unset")


async def _read_upload(file: UploadFile) -> tuple[Path, str]:
    """Validate and persist one upload. Raises UploadRejected with an HTTP status."""
    name = Path(file.filename or "upload")
    if name.suffix.lower() not in service.SUPPORTED_SUFFIXES:
        raise UploadRejected(415, f"unsupported file type '{name.suffix}'; "
                                  f"expected one of {sorted(service.SUPPORTED_SUFFIXES)}")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected(413, f"file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
    if not data:
        raise UploadRejected(400, "empty file")
    # A unique name per upload, never the client's filename: two clients sending
    # "receipt.jpg" at once must not share a path on disk.
    tmp = Path(tempfile.gettempdir()) / f"upload_{uuid.uuid4().hex}{name.suffix.lower()}"
    tmp.write_bytes(data)
    return tmp, name.stem


async def _extract_upload(path: Path, doc_id: str) -> ExtractionOut:
    try:
        result = await run_in_threadpool(_extract_bounded, path, doc_id)
    finally:
        path.unlink(missing_ok=True)
    telemetry.record(result)
    return result


@app.get("/health")
def health() -> dict:
    problems = service.config_problems()
    return {
        "status": "misconfigured" if problems["missing"] else
                  "degraded" if problems["degraded"] else "ok",
        "mode": service.mode(),
        "missing_config": problems["missing"],
        "degraded_config": problems["degraded"],
        "model_a": os.environ.get("VLM_A_MODEL"),
        "model_b": os.environ.get("LLM_B_MODEL"),
        "cascade_model": os.environ.get("CASCADE_MODEL") if service.mode() == "cascade" else None,
        "confidence_model": "fitted" if service.confidence_model() else "fallback weights",
        "threshold": service.threshold(),
        "limits": {"max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
                   "max_batch_files": MAX_BATCH_FILES, "max_concurrency": MAX_CONCURRENCY},
    }


@app.get("/metrics")
def metrics() -> dict:
    return telemetry.snapshot()


@app.post("/extract", response_model=ExtractionOut, response_model_exclude_none=False)
async def extract(file: UploadFile = File(...)) -> ExtractionOut:
    telemetry.request()
    _require_config()
    try:
        path, doc_id = await _read_upload(file)
    except UploadRejected as exc:
        telemetry.reject()
        raise HTTPException(exc.status, exc.message) from None

    result = await _extract_upload(path, doc_id)
    if result.status == "error":
        # The document was accepted but could not be processed: a failure of this
        # request, not a malformed request, so 502 — with the structured body.
        raise HTTPException(502, detail=result.model_dump())
    return result


@app.post("/extract/batch", response_model=BatchOut)
async def extract_batch(files: list[UploadFile] = File(...)) -> BatchOut:
    """Always 200 once the batch itself is acceptable. Per-file problems (a bad type, an
    empty file, a failed extraction) come back as status="error" entries in `results`,
    in upload order, so one bad receipt never costs the caller the other nineteen."""
    telemetry.request()
    _require_config()
    if len(files) > MAX_BATCH_FILES:
        telemetry.reject()
        raise HTTPException(413, f"{len(files)} files exceeds the batch limit of {MAX_BATCH_FILES}")

    started = time.perf_counter()
    seen: dict[str, int] = {}
    jobs = []
    for upload in files:
        stem = Path(upload.filename or "upload").stem
        seen[stem] = seen.get(stem, 0) + 1
        # Two files named receipt.jpg in one batch must stay distinguishable in results.
        doc_id = stem if seen[stem] == 1 else f"{stem}#{seen[stem]}"
        try:
            path, _ = await _read_upload(upload)
        except UploadRejected as exc:
            rejected = from_error(doc_id, ValueError(exc.message))
            telemetry.record(rejected)
            jobs.append(asyncio.sleep(0, result=rejected))
            continue
        jobs.append(_extract_upload(path, doc_id))

    results = list(await asyncio.gather(*jobs))
    summary = summarize_results(results, wall_s=time.perf_counter() - started,
                                concurrency=MAX_CONCURRENCY)
    return BatchOut(results=results, summary=summary)
