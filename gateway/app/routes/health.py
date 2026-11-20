"""``GET /healthz`` (liveness, never touches ComfyUI) and ``GET /readyz`` (readiness).

``/readyz`` is unauthenticated, so its ComfyUI probe is cached for ``READY_CACHE_S``
(success and failure alike; at most one ``/system_stats`` call per window) and its body
carries only queue depth numbers, no job identifiers.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.comfy.client import ComfyError

router = APIRouter()

READY_CHECK_TIMEOUT_S = 5.0
READY_CACHE_S = 2.0
_PUBLIC_QUEUE_FIELDS = (
    "active",
    "queued",
    "comfy_external_pending",
    "estimated_wait_s",
    "max_depth",
)


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


async def _comfy_status(request: Request) -> str:
    state = request.app.state
    cached = getattr(state, "readyz_cache", None)
    now = time.monotonic()
    if cached is not None and now - cached[0] < READY_CACHE_S:
        return cached[1]
    lock = getattr(state, "readyz_lock", None)
    if lock is None:
        lock = state.readyz_lock = asyncio.Lock()
    async with lock:  # concurrent probes share one /system_stats call
        cached = getattr(state, "readyz_cache", None)
        if cached is not None and time.monotonic() - cached[0] < READY_CACHE_S:
            return cached[1]
        comfy = "ok"
        try:
            async with asyncio.timeout(READY_CHECK_TIMEOUT_S):
                await state.client.health()
        except (ComfyError, TimeoutError) as exc:
            comfy = f"unavailable: {type(exc).__name__}"
        state.readyz_cache = (time.monotonic(), comfy)
        return comfy


@router.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    comfy = await _comfy_status(request)
    ready = comfy == "ok"
    snap = request.app.state.queue.snapshot()
    body = {
        "status": "ready" if ready else "not_ready",
        "comfyui": comfy,
        "queue": {k: snap[k] for k in _PUBLIC_QUEUE_FIELDS},
    }
    return JSONResponse(body, status_code=200 if ready else 503)
