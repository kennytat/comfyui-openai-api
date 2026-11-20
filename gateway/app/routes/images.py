"""``POST /v1/images/generations``.

Flow: validate -> resolve model (forced availability refresh on not-found) / size /
steps -> estimate + admit (synchronous) -> enqueue -> race the job future against a
client-disconnect watcher -> fetch outputs -> Pillow re-encode without metadata and
base64/store (``asyncio.to_thread``) -> response.

Everything after admission runs inside ``asyncio.timeout_at(t0 + REQUEST_TIMEOUT_S)``; the
worker's generation deadline is ``t0 + REQUEST_TIMEOUT_S - POSTPROCESS_RESERVE_S``. The
``finally`` sits *outside* the timeout scope and awaits a shielded cleanup task that
cancels unfinished ComfyUI work (targeted), deletes outputs and the history entry, releases
the admission slot exactly once and emits one structured log line per job. On any non-200
outcome the cleanup also aborts the encode thread through its ``EncodeGuard`` (checked
between blobs and before every ``store.put``) and deletes hosted files it already stored,
so a 504 or disconnect leaves no orphaned ``/images`` files. Cleanup tasks are held in a
module-level set until done.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from app import errors
from app.auth import require_api_key
from app.comfy.client import (
    ComfyError,
    UpstreamInterrupted,
    UpstreamOutputMissing,
    UpstreamUnavailable,
    WaitResult,
)
from app.comfy.workflows import GenerationParams, build_prompt, resolve_params
from app.postprocess import reencode
from app.queue import Job
from app.schemas import ImageGenerationRequest
from app.storage import ImageStore, delete_comfy_outputs

log = logging.getLogger("gateway.images")
job_log = logging.getLogger("gateway.jobs")

router = APIRouter(dependencies=[Depends(require_api_key)])

CLEANUP_TIMEOUT_S = 5.0
_CLEANUP_TASKS: set[asyncio.Future[None]] = set()
CLIENT_GONE_STATUS = 499  # nginx convention; never actually delivered to the client


class ClientDisconnected(Exception):
    """The HTTP client went away while its job was pending."""


class EncodeAborted(Exception):
    """The encode thread stopped because the request was aborted or ran out of time."""


class EncodeGuard:
    """Shared by the encode thread and the route so no hosted file outlives an abort.

    The thread checks it between blobs and stores through ``put`` (check + store under one
    lock); ``abort`` (route cleanup on any non-200 outcome) flips the flag under the same
    lock and deletes whatever was already stored. ``deadline`` (monotonic) makes the thread
    abort itself once the request budget is spent.
    """

    def __init__(self, store: ImageStore, deadline: float) -> None:
        self.store = store
        self.deadline = deadline
        self._lock = threading.Lock()
        self._aborted = False
        self._stored: list[tuple[str, str]] = []

    def check(self) -> None:
        if self._aborted:
            raise EncodeAborted()
        if time.monotonic() >= self.deadline:
            self.abort()
            raise EncodeAborted()

    def put(self, data: bytes, ext: str) -> str:
        self.check()
        with self._lock:
            if self._aborted:
                raise EncodeAborted()
            image_id = self.store.put(data, ext)
            self._stored.append((image_id, ext))
            return image_id

    def abort(self) -> int:
        """Stop the thread at its next check and delete stored files. Returns the count."""
        with self._lock:
            self._aborted = True
            stored, self._stored = self._stored, []
        for image_id, ext in stored:
            self.store.delete(image_id, ext)
        return len(stored)


def map_comfy_error(exc: ComfyError) -> errors.OpenAIError:
    if isinstance(exc, UpstreamUnavailable):
        return errors.upstream_unavailable()
    if isinstance(exc, UpstreamInterrupted):
        return errors.upstream_interrupted()
    if exc.oom or exc.code == "upstream_oom":
        return errors.upstream_oom()
    if isinstance(exc, UpstreamOutputMissing):
        return errors.upstream_output_missing()
    return errors.upstream_error(retryable=not exc.non_retryable)


async def _watch_disconnect(request: Request, interval_s: float) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(interval_s)


async def _wait_job(request: Request, job: Job, interval_s: float) -> WaitResult:
    watcher = asyncio.create_task(_watch_disconnect(request, interval_s))
    try:
        done, _ = await asyncio.wait({job.future, watcher}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        watcher.cancel()
    if job.future in done:
        return job.future.result()
    raise ClientDisconnected()


def _encode_all(
    blobs: list[bytes],
    output_format: str,
    compression: int | None,
    as_b64: bool,
    guard: EncodeGuard,
    public_base_url: str,
) -> list[dict[str, Any]]:
    """Blocking (Pillow + base64 / disk); runs via asyncio.to_thread."""
    items: list[dict[str, Any]] = []
    for blob in blobs:
        guard.check()
        data = reencode(blob, output_format, compression)
        if as_b64:
            items.append(
                {"b64_json": base64.b64encode(data).decode("ascii"), "revised_prompt": None}
            )
        else:
            image_id = guard.put(data, output_format)
            url = f"{public_base_url}/images/{image_id}.{output_format}"
            items.append({"url": url, "revised_prompt": None})
    return items


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)


@router.post("/v1/images/generations")
async def create_image(body: ImageGenerationRequest, request: Request) -> Response:
    state = request.app.state
    settings = state.settings
    t0 = getattr(request.state, "t0", None) or time.monotonic()
    total_deadline = t0 + settings.request_timeout_s
    gen_deadline = total_deadline - settings.postprocess_reserve_s

    body.check_limits(settings.max_prompt_chars, settings.max_n)
    response_format = body.response_format or settings.default_response_format
    if response_format == "url" and not settings.enable_url_format:
        raise errors.unsupported_value(
            "response_format 'url' is disabled on this server.", "response_format"
        )
    model_id = body.model or settings.default_model
    try:
        entry = await state.catalog.require(model_id)
    except UpstreamUnavailable as exc:
        raise errors.upstream_unavailable() from exc
    wh = entry.resolve_size(body.size)
    if wh is None:
        raise errors.invalid_value(
            f"Invalid size '{body.size}'. Supported sizes: {', '.join(entry.sizes)}, auto.",
            "size",
            code="invalid_size",
        )
    width, height = wh
    output_format = body.output_format or "png"
    job_id = uuid.uuid4().hex
    params = resolve_params(
        entry,
        GenerationParams(
            prompt=body.prompt,
            width=width,
            height=height,
            n=body.n or 1,
            seed=body.seed,
            steps=body.steps,
            quality=body.quality,
            cfg_scale=body.cfg_scale,
            negative_prompt=body.negative_prompt,
            filename_prefix=f"{settings.comfy_output_subfolder}/{job_id}",
        ),
    )
    graph = build_prompt(entry, params)
    job = state.queue.admit(entry, params, graph, gen_deadline, job_id)  # may raise 400/429

    record: dict[str, Any] = {
        "job_id": job.id,
        "model": entry.id,
        "n": params.n,
        "size": f"{width}x{height}",
        "steps": params.steps,
        "seed": params.seed,
        "est_job_s": round(job.est_s, 2),
        "est_wait_s": round(job.est_wait_s, 2),
        "cold_load_applied": job.cold_load_applied,
        "comfy_external_pending": job.comfy_external_pending,
    }
    history: dict[str, Any] | None = None
    guard = EncodeGuard(state.store, total_deadline)
    status, outcome = 500, "internal_error"
    loop = asyncio.get_running_loop()
    try:
        try:
            async with asyncio.timeout_at(loop.time() + (total_deadline - time.monotonic())):
                result = await _wait_job(request, job, state.disconnect_poll_s)
                history = result.history
                record["comfy_wait_ms"] = _ms(result.comfy_wait_s)
                record["gpu_ms"] = _ms(result.gpu_s)
                blobs = await state.client.fetch_outputs(history, entry.output_node)
                pp0 = time.monotonic()
                items = await asyncio.to_thread(
                    _encode_all,
                    blobs,
                    output_format,
                    body.output_compression,
                    response_format == "b64_json",
                    guard,
                    settings.public_base_url,
                )
                record["postprocess_ms"] = _ms(time.monotonic() - pp0)
        except (TimeoutError, EncodeAborted) as exc:
            raise errors.timeout() from exc
        except ComfyError as exc:
            raise map_comfy_error(exc) from exc
        status, outcome = 200, "ok"
        return JSONResponse(
            {
                "created": int(state.clock()),
                "data": items,
                "output_format": output_format,
                "size": f"{width}x{height}",
                "quality": body.quality or "auto",
            }
        )
    except errors.OpenAIError as exc:
        status, outcome = exc.status, exc.code or "error"
        raise
    except ClientDisconnected:
        status, outcome = CLIENT_GONE_STATUS, "client_disconnected"
        return Response(status_code=CLIENT_GONE_STATUS)
    except asyncio.CancelledError:
        status, outcome = CLIENT_GONE_STATUS, "cancelled"
        raise
    finally:
        record.update(outcome=outcome, http_status=status)
        cleanup = asyncio.ensure_future(
            _cleanup(request, job, history, t0, record, None if status == 200 else guard)
        )
        _CLEANUP_TASKS.add(cleanup)  # strong ref: survives a cancelled route task
        cleanup.add_done_callback(_CLEANUP_TASKS.discard)
        await asyncio.shield(cleanup)


async def _cleanup(
    request: Request,
    job: Job,
    history: dict[str, Any] | None,
    t0: float,
    record: dict[str, Any],
    abort_encode: EncodeGuard | None = None,
) -> None:
    """Cancel unfinished ComfyUI work, delete outputs + history, release the slot once."""
    state = request.app.state
    settings = state.settings
    client = state.client
    queue = state.queue
    try:
        queue.abandon(job)  # stop the worker's wait / skip it if still queued
        if abort_encode is not None:
            # stop a still-running encode thread; delete hosted files it already stored
            removed = await asyncio.to_thread(abort_encode.abort)
            if removed:
                record["hosted_files_removed"] = removed
        pid = job.prompt_id
        if pid is not None and job.claim_cleanup():
            async with asyncio.timeout(CLEANUP_TIMEOUT_S):
                if not job.finished_upstream:
                    try:
                        record["cancel"] = await client.cancel(pid)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("cancel of prompt %s failed: %s", pid, exc)
                if settings.cleanup_comfy_outputs:
                    try:
                        if history is None:
                            history = await client.get_history(pid)
                        if history:
                            images = client.output_images(history, job.model.output_node)
                            await asyncio.to_thread(
                                delete_comfy_outputs,
                                [(i.subfolder, i.filename) for i in images],
                                settings.comfy_output_dir,
                                settings.comfy_output_subfolder,
                            )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("output cleanup for prompt %s failed: %s", pid, exc)
                try:
                    await client.delete_history(pid)
                except Exception as exc:  # noqa: BLE001
                    log.warning("history delete for prompt %s failed: %s", pid, exc)
    except Exception as exc:  # noqa: BLE001 - incl. cleanup timeout
        log.warning("cleanup of job %s incomplete: %s", job.id, exc)
    finally:
        queue.release(job)
        if job.prompt_id is not None:
            client.forget(job.prompt_id)
        record["prompt_id"] = job.prompt_id
        if job.submitted_at is not None:
            record["queue_wait_ms"] = _ms(job.submitted_at - t0)
        record["total_ms"] = _ms(time.monotonic() - t0)
        job_log.info(json.dumps(record, sort_keys=True))
