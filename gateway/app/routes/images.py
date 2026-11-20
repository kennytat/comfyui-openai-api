"""``POST /v1/images/generations`` (JSON), ``/v1/images/edits`` and
``/v1/images/variations`` (multipart).

Flow: validate -> resolve model (forced availability refresh on not-found) / operation /
size / steps -> (edits/variations: cheap capacity pre-check before the body is read, then
Pillow preprocessing of the uploads in a thread, at most ``MAX_CONCURRENT_PREPROCESS`` at
once process-wide; extra requests fail fast with 429 ``queue_full``) ->
estimate + admit (synchronous) -> upload inputs to ComfyUI (``<job_id>_image.png``,
``<job_id>_mask.png``; the worker waits for ``job.inputs_ready`` before submitting) ->
race the job future against a client-disconnect watcher -> fetch outputs -> Pillow
re-encode without metadata and base64/store (``asyncio.to_thread``) -> response. All three
routes share ``_run_job`` (admission, deadline, disconnect watcher, shielded cleanup,
encode guard, per-job log line).

Everything after admission runs inside ``asyncio.timeout_at(t0 + REQUEST_TIMEOUT_S)``; the
worker's generation deadline is ``t0 + REQUEST_TIMEOUT_S - POSTPROCESS_RESERVE_S``. The
``finally`` sits *outside* the timeout scope and awaits a shielded cleanup task that
cancels unfinished ComfyUI work (targeted), deletes outputs and the history entry, releases
the admission slot exactly once and emits one structured log line per job. Uploaded inputs
are deleted from the ComfyUI input mount (``COMFY_INPUT_DIR``) after every outcome (also a
name ComfyUI renamed an upload to), and once more ``app.state.input_retry_delay_s`` later
in a tracked background task (``app.state.background_tasks``, cancelled at shutdown) to
catch uploads that land after a cancelled request. While uploading, the route also
watches the job future: if the worker gave up waiting (``UPLOAD_WAIT_S``) the upload is
cancelled and the job's 503 is returned. On any non-200
outcome the cleanup also aborts the encode thread through its ``EncodeGuard`` (checked
between blobs and before every ``store.put``) and deletes hosted files it already stored,
so a 504 or disconnect leaves no orphaned ``/images`` files. Cleanup tasks are held in a
module-level set until done.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from app import errors
from app.auth import require_api_key
from app.comfy.client import (
    ComfyError,
    UpstreamError,
    UpstreamInterrupted,
    UpstreamOutputMissing,
    UpstreamUnavailable,
    WaitResult,
)
from app.comfy.workflows import GenerationParams, ModelEntry, build_prompt, resolve_params
from app.postprocess import reencode
from app.queue import Job
from app.ratelimit import enforce_rate_limit
from app.schemas import (
    ImageEditRequest,
    ImageGenerationRequest,
    ImageOptions,
    ImageVariationRequest,
)
from app.storage import ImageStore, delete_comfy_outputs
from app.uploads import parse_multipart, prepare_inputs

log = logging.getLogger("gateway.images")
job_log = logging.getLogger("gateway.jobs")

router = APIRouter(dependencies=[Depends(require_api_key), Depends(enforce_rate_limit)])

CLEANUP_TIMEOUT_S = 5.0
_CLEANUP_TASKS: set[asyncio.Future[None]] = set()
PREPROCESS_BUSY_RETRY_S = 1.0
CLIENT_GONE_STATUS = 499  # nginx convention; never actually delivered to the client
MULTIPART_PATHS = ("/v1/images/edits", "/v1/images/variations")


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


def _response_format(body: ImageOptions, settings: Any) -> str:
    response_format = body.response_format or settings.default_response_format
    if response_format == "url" and not settings.enable_url_format:
        raise errors.unsupported_value(
            "response_format 'url' is disabled on this server.", "response_format"
        )
    return response_format


async def _resolve_model(request: Request, body: ImageOptions, operation: str) -> ModelEntry:
    state = request.app.state
    model_id = body.model or state.settings.default_model
    try:
        entry: ModelEntry = await state.catalog.require(model_id)
    except UpstreamUnavailable as exc:
        raise errors.upstream_unavailable() from exc
    if entry.operation(operation) is None:
        verb = {"edit_masked": "masked edits", "edit": "edits", "variation": "variations"}
        raise errors.unsupported_value(
            f"The model '{entry.id}' does not support {verb.get(operation, operation)}.",
            "model",
        )
    return entry


def _resolve_size(entry: ModelEntry, size: str | None) -> tuple[int, int]:
    wh = entry.resolve_size(size)
    if wh is None:
        raise errors.invalid_value(
            f"Invalid size '{size}'. Supported sizes: {', '.join(entry.sizes)}, auto.",
            "size",
            code="invalid_size",
        )
    return wh


@dataclass(frozen=True)
class _Upload:
    """An input file the job uploads to ComfyUI after admission."""

    filename: str  # server-generated, e.g. "<job_id>_image.png"
    name: str  # ComfyUI input name the graph references ("<subfolder>/<filename>")
    data: bytes


@router.post("/v1/images/generations")
async def create_image(body: ImageGenerationRequest, request: Request) -> Response:
    settings = request.app.state.settings
    t0 = getattr(request.state, "t0", None) or time.monotonic()
    body.check_limits(settings.max_prompt_chars, settings.max_n)
    response_format = _response_format(body, settings)
    entry = await _resolve_model(request, body, "generate")
    width, height = _resolve_size(entry, body.size)
    job_id = uuid.uuid4().hex
    params = GenerationParams(
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
    )
    return await _run_job(request, t0, job_id, entry, params, body, response_format, [])


def _form(model: type[ImageOptions], fields: dict[str, str]) -> ImageOptions:
    try:
        return model.model_validate(fields)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc


def _single_file(
    files: dict[str, list[bytes]], fields: dict[str, str], names: tuple[str, ...], required: bool
) -> bytes | None:
    param = names[0]
    found = [blob for n in names for blob in files.get(n, [])]
    if not found:
        if any(n in fields for n in names):
            raise errors.invalid_value(f"'{param}' must be uploaded as a file.", param)
        if required:
            raise errors.OpenAIError(
                400,
                f"Missing required parameter: '{param}'.",
                code="missing_required_parameter",
                param=param,
            )
        return None
    if len(found) > 1:
        raise errors.unsupported_value(f"Only one '{param}' file is supported per request.", param)
    return found[0]


async def _image_request(
    request: Request,
    form_model: type[ImageOptions],
    operation: str,
    with_mask: bool,
) -> Response:
    """Shared front half of edits/variations: parse, validate, preprocess, then run."""
    state = request.app.state
    settings = state.settings
    t0 = getattr(request.state, "t0", None) or time.monotonic()
    # Cheap rejection before reading/decoding the upload when admission cannot succeed.
    state.queue.precheck(t0 + settings.generation_budget_s)
    fields, files = await parse_multipart(request, settings.max_upload_bytes)
    body = _form(form_model, fields)
    image = _single_file(files, fields, ("image", "image[]"), required=True)
    mask = _single_file(files, fields, ("mask", "mask[]"), required=False) if with_mask else None
    body.check_limits(settings.max_prompt_chars, settings.max_n)
    response_format = _response_format(body, settings)
    if mask is not None:
        operation = "edit_masked"
    entry = await _resolve_model(request, body, operation)
    width, height = _resolve_size(entry, body.size)
    assert image is not None
    sem: asyncio.Semaphore = state.preprocess_sem
    if sem.locked():  # fail fast: never queue CPU/memory-heavy decodes behind each other
        raise errors.queue_full(
            PREPROCESS_BUSY_RETRY_S, "The server is busy preprocessing uploads. Try again later."
        )
    async with sem:
        image_png, mask_png = await asyncio.to_thread(
            prepare_inputs, image, mask, width, height, settings.max_input_pixels
        )
    job_id = uuid.uuid4().hex
    sub = settings.comfy_input_subfolder.strip("/")
    uploads = [_Upload(f"{job_id}_image.png", f"{sub}/{job_id}_image.png", image_png)]
    if mask_png is not None:
        uploads.append(_Upload(f"{job_id}_mask.png", f"{sub}/{job_id}_mask.png", mask_png))
    params = GenerationParams(
        prompt=getattr(body, "prompt", ""),
        width=width,
        height=height,
        n=body.n or 1,
        seed=body.seed,
        steps=body.steps,
        quality=body.quality,
        cfg_scale=body.cfg_scale,
        negative_prompt=body.negative_prompt,
        filename_prefix=f"{settings.comfy_output_subfolder}/{job_id}",
        operation=operation,
        image=uploads[0].name,
        mask=uploads[1].name if len(uploads) > 1 else None,
        denoise=body.strength,  # type: ignore[attr-defined]
    )
    return await _run_job(request, t0, job_id, entry, params, body, response_format, uploads)


@router.post("/v1/images/edits")
async def edit_image(request: Request) -> Response:
    return await _image_request(request, ImageEditRequest, "edit", with_mask=True)


@router.post("/v1/images/variations")
async def create_variation(request: Request) -> Response:
    return await _image_request(request, ImageVariationRequest, "variation", with_mask=False)


async def _upload_all(client: Any, job: Job, uploads: list[_Upload]) -> None:
    for upload in uploads:
        job.inputs.append(upload.name)  # deleted in cleanup even if partial
        name = await client.upload_image(upload.data, upload.filename)
        if name != upload.name:
            job.inputs.append(name)  # the renamed file exists too: delete it as well
            raise UpstreamError(f"ComfyUI renamed upload to {name!r}")


async def _upload_inputs(client: Any, job: Job, uploads: list[_Upload]) -> None:
    """Upload, but give up as soon as the worker failed the job (e.g. UPLOAD_WAIT_S)."""
    task = asyncio.ensure_future(_upload_all(client, job, uploads))
    try:
        await asyncio.wait({task, job.future}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    if task.cancelled():
        job.future.result()  # the worker's error (503 / timeout)
    task.result()


async def _retry_input_delete(inputs: list[str], settings: Any, delay_s: float) -> None:
    """Second pass for uploads that landed after the first delete (cancelled request)."""
    await asyncio.sleep(delay_s)
    try:
        await asyncio.to_thread(
            delete_comfy_outputs, inputs, settings.comfy_input_dir, settings.comfy_input_subfolder
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("delayed input cleanup failed: %s", exc)


async def _run_job(
    request: Request,
    t0: float,
    job_id: str,
    entry: ModelEntry,
    params: GenerationParams,
    body: ImageOptions,
    response_format: str,
    uploads: list[_Upload],
) -> Response:
    """Shared pipeline: estimate + admit -> upload inputs -> wait (deadline, disconnect)
    -> fetch -> encode -> response; shielded cleanup outside the timeout scope."""
    state = request.app.state
    settings = state.settings
    total_deadline = t0 + settings.request_timeout_s
    gen_deadline = total_deadline - settings.postprocess_reserve_s
    output_format = body.output_format or "png"
    params = resolve_params(entry, params)
    graph = build_prompt(entry, params)
    job = state.queue.admit(  # may raise 400/429
        entry, params, graph, gen_deadline, job_id, needs_inputs=bool(uploads)
    )

    record: dict[str, Any] = {
        "job_id": job.id,
        "model": entry.id,
        "operation": params.operation,
        "n": params.n,
        "size": f"{params.width}x{params.height}",
        "steps": params.steps,
        "seed": params.seed,
        "est_job_s": round(job.est_s, 2),
        "est_scale": round(job.est_scale, 3),
        "est_wait_s": round(job.est_wait_s, 2),
        "cold_load_applied": job.cold_load_applied,
        "comfy_external_pending": job.comfy_external_pending,
    }
    if params.denoise is not None:
        record["denoise"] = params.denoise
    history: dict[str, Any] | None = None
    guard = EncodeGuard(state.store, total_deadline)
    status, outcome = 500, "internal_error"
    loop = asyncio.get_running_loop()
    try:
        try:
            async with asyncio.timeout_at(loop.time() + (total_deadline - time.monotonic())):
                if uploads:
                    up0 = time.monotonic()
                    await _upload_inputs(state.client, job, uploads)
                    record["upload_ms"] = _ms(time.monotonic() - up0)
                    assert job.inputs_ready is not None
                    job.inputs_ready.set()
                result = await _wait_job(request, job, state.disconnect_poll_s)
                history = result.history
                record["comfy_wait_ms"] = _ms(result.comfy_wait_s)
                record["gpu_ms"] = _ms(result.gpu_s)
                blobs = await state.client.fetch_outputs(history, job.output_node)
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
                "size": f"{params.width}x{params.height}",
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
                            images = client.output_images(history, job.output_node)
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
        if job.inputs:  # uploaded user images: always removed once ComfyUI is done
            try:
                deleted = await asyncio.to_thread(
                    delete_comfy_outputs,
                    job.inputs,
                    settings.comfy_input_dir,
                    settings.comfy_input_subfolder,
                )
                record["inputs_deleted"] = len(deleted)
            except Exception as exc:  # noqa: BLE001
                log.warning("input cleanup for job %s failed: %s", job.id, exc)
            retry = asyncio.ensure_future(
                _retry_input_delete(list(job.inputs), settings, state.input_retry_delay_s)
            )
            tasks: set[asyncio.Future[None]] = state.background_tasks
            tasks.add(retry)  # tracked; cancelled at shutdown
            retry.add_done_callback(tasks.discard)
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
        metrics = getattr(state, "metrics", None)
        if metrics is not None:
            metrics.observe_job(record)
