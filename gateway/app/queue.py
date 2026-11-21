"""Single-GPU job queue: deterministic admission, FIFO worker, idempotent slot release.

Public API:

- ``Job``: one admitted generation. The route owns its lifecycle; the worker only
  submits it to ComfyUI, waits for completion and resolves ``job.future`` with a
  ``WaitResult`` or an exception (``ComfyError`` subclasses, ``TimeoutError``).
  ``inputs_ready`` (set for jobs with uploaded inputs, e.g. edits): the worker waits for
  it (bounded by ``min(generation deadline, now + UPLOAD_WAIT_S)``) before submitting, so
  the route can upload after admission without racing the worker; ``abandon`` sets it to
  wake the worker. A job whose uploads miss ``UPLOAD_WAIT_S`` fails with
  ``UpstreamUnavailable`` (503) so a slow upload never holds the FIFO longer.
  ``inputs`` lists uploaded ComfyUI input names the route deletes during cleanup.
  Flags: ``cancelled`` (route abandoned it; worker skips/abandons), ``submitted``,
  ``finished_upstream`` (history reached; no cancel needed), ``released``
  (admission slot already returned), ``cleanup_claimed`` (ComfyUI-side cleanup owner
  decided, so cancel/history-delete happen once).
- ``JobQueue(settings, client)``:
  - ``admit(model, params, graph, gen_deadline, job_id=None, needs_inputs=False) -> Job``:
    **synchronous**
    (no ``await`` between the checks and ``active += 1``). Raises
    ``request_too_expensive`` (400) when the job alone (warm checkpoint) exceeds the
    generation budget, ``queue_full`` (429) when ``active + comfy_external_pending >=
    QUEUE_MAX_DEPTH + 1``, ``estimated_wait_exceeded`` (429) when
    ``estimated_wait + own_estimate`` exceeds the time remaining until ``gen_deadline``
    (elapsed time since request receipt is charged). Exception: with nothing to wait for,
    a cold-load estimate that does not fit is still admitted when the warm estimate fits
    (a 429 would only loop). Retry-After = estimated wait.
  - ``precheck(gen_deadline)``: cheap capacity check run before reading a multipart body
    (same ``queue_full`` threshold as ``admit``; ``estimated_wait_exceeded`` when the
    current estimated wait alone exceeds the remaining budget). Not a reservation.
  - ``release(job)``: idempotent; decrements ``active`` once. Called only from the
    route's shielded ``finally``.
  - ``abandon(job)``: mark cancelled, stop the worker's wait, cancel the future.
  - ``estimated_wait() -> (seconds, tail_checkpoint)``: running remaining + queued
    estimates + ``comfy_external_pending * EST_EXTERNAL_JOB_S``.
  - ``refresh_external()``: one ``/queue`` poll (external + stale-gateway entries
    counted by ``client_id``); ``run_poller()`` loops it every ``COMFY_QUEUE_POLL_S``.
  - ``run_worker()``: the single FIFO worker (skips cancelled or already-expired jobs).
    ``/prompt`` is bounded by the job's generation deadline; after a submit timeout or
    transport error one ``/queue`` snapshot is taken and owned prompts no job tracks
    (orphans accepted after we stopped waiting) are cancelled.
  - ``snapshot() -> dict`` for ``/readyz`` (includes the calibration per model).
- ``EstimateCalibrator(alpha, scale_min, scale_max, min_samples)``: per-model EWMA of
  ``measured / estimated`` sampling time (``gpu_s - est_extra_s`` over ``compute_seconds``)
  and of its absolute deviation, fed by warm successful jobs (the model's weights were
  already the last ones submitted). ``scale(model_id)`` is ``1.0`` until ``min_samples``
  observations, then ``clamp(mean + deviation, scale_min, scale_max)``: biased upward so
  estimates stay at or above typical real times. ``JobQueue`` applies it to every estimate
  when ``EST_AUTOCALIBRATE`` is on. Observations are kept in memory only (reset on restart).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.comfy.client import ComfyClient, ComfyError, UpstreamUnavailable, WaitResult
from app.comfy.workflows import (
    GenerationParams,
    ModelEntry,
    compute_seconds,
    estimate_seconds,
)
from app.config import Settings
from app.errors import estimated_wait_exceeded, queue_full, request_too_expensive

log = logging.getLogger("gateway.queue")

UPSTREAM_CLEANUP_TIMEOUT_S = 5.0


@dataclass(eq=False)
class Job:
    id: str
    model: ModelEntry
    params: GenerationParams
    graph: dict[str, Any]
    gen_deadline: float  # time.monotonic() deadline for generation
    est_s: float  # own estimate at admission (incl. cold load if applied)
    est_wait_s: float
    cold_load_applied: bool
    comfy_external_pending: int
    future: asyncio.Future[WaitResult]
    admitted_at: float = field(default_factory=time.monotonic)
    state: str = "queued"
    prompt_id: str | None = None
    submitted: bool = False
    submitted_at: float | None = None
    run_started_at: float | None = None
    finished_upstream: bool = False
    cancelled: bool = False
    released: bool = False
    cleanup_claimed: bool = False
    wait_task: asyncio.Task[WaitResult] | None = None
    inputs_ready: asyncio.Event | None = None
    inputs: list[str] = field(default_factory=list)
    warm_at_submit: bool = False  # the previous submitted job used the same weights
    est_scale: float = 1.0  # learned estimate correction applied at admission
    extra_data: dict[str, Any] | None = None  # sent with /prompt (e.g. preview_method)

    @property
    def checkpoint(self) -> str:
        return self.model.checkpoint

    @property
    def output_node(self) -> str:
        op = self.model.operation(self.params.operation)
        return op.output_node if op is not None else self.model.output_node

    def claim_cleanup(self) -> bool:
        """Return True exactly once: the caller owns ComfyUI-side cleanup."""
        if self.cleanup_claimed:
            return False
        self.cleanup_claimed = True
        return True


@dataclass
class _Ewma:
    mean: float
    dev: float
    count: int


class EstimateCalibrator:
    def __init__(self, alpha: float, scale_min: float, scale_max: float, min_samples: int) -> None:
        self.alpha = alpha
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.min_samples = min_samples
        self._stats: dict[str, _Ewma] = {}

    def scale(self, model_id: str) -> float:
        st = self._stats.get(model_id)
        if st is None or st.count < self.min_samples:
            return 1.0
        return min(self.scale_max, max(self.scale_min, st.mean + st.dev))

    def observe(self, model_id: str, estimated_s: float, measured_s: float) -> None:
        if estimated_s <= 0 or measured_s <= 0:
            return
        ratio = measured_s / estimated_s
        st = self._stats.get(model_id)
        if st is None:
            self._stats[model_id] = _Ewma(mean=ratio, dev=0.0, count=1)
            return
        a = self.alpha
        st.dev = (1 - a) * st.dev + a * abs(ratio - st.mean)
        st.mean = (1 - a) * st.mean + a * ratio
        st.count += 1

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            mid: {
                "ratio_mean": round(st.mean, 3),
                "ratio_dev": round(st.dev, 3),
                "samples": st.count,
                "scale": round(self.scale(mid), 3),
            }
            for mid, st in self._stats.items()
        }


class JobQueue:
    def __init__(self, settings: Settings, client: ComfyClient) -> None:
        self.settings = settings
        self.client = client
        self.active = 0
        self.comfy_external_pending = 0
        self.comfy_reachable: bool | None = None
        self.last_checkpoint: str | None = None
        self.running: Job | None = None
        self._pending: deque[Job] = deque()
        self._wakeup = asyncio.Event()
        self.calibrator = EstimateCalibrator(
            settings.est_ewma_alpha,
            settings.est_scale_min,
            settings.est_scale_max,
            settings.est_calibration_min_samples,
        )

    def scale(self, model: ModelEntry) -> float:
        """Learned estimate correction for ``model`` (1.0 when auto-calibration is off)."""
        return self.calibrator.scale(model.id) if self.settings.est_autocalibrate else 1.0

    def _observe(self, job: Job, result: WaitResult) -> None:
        """Feed a warm, successful job's measured GPU time into the calibration."""
        gpu_s = result.gpu_s
        if not job.warm_at_submit or gpu_s is None:
            return
        op = job.model.operation(job.params.operation)
        extra = op.est_extra_s if op is not None else 0.0
        estimated = compute_seconds(job.model, job.params)
        self.calibrator.observe(job.model.id, estimated, max(0.0, gpu_s - extra))

    # ------------------------------------------------------------------ estimates

    @property
    def budget_s(self) -> float:
        return self.settings.generation_budget_s

    def _queued(self) -> list[Job]:
        return [j for j in self._pending if not j.cancelled]

    def estimated_wait(self) -> tuple[float, str | None]:
        now = time.monotonic()
        wait = 0.0
        tail = self.last_checkpoint
        run = self.running
        if run is not None and not run.cancelled:
            started = run.run_started_at or now
            wait += max(0.0, run.est_s - (now - started))
            tail = run.checkpoint
        for job in self._queued():
            wait += job.est_s
            tail = job.checkpoint
        wait += self.comfy_external_pending * self.settings.est_external_job_s
        return wait, tail

    # ------------------------------------------------------------------ admission

    def precheck(self, gen_deadline: float) -> None:
        """Reject early (before the upload is read/decoded) when admission cannot succeed."""
        wait, _ = self.estimated_wait()
        if self.active + self.comfy_external_pending >= self.settings.queue_max_depth + 1:
            raise queue_full(wait)
        if wait > gen_deadline - time.monotonic():
            raise estimated_wait_exceeded(wait)

    def admit(
        self,
        model: ModelEntry,
        params: GenerationParams,
        graph: dict[str, Any],
        gen_deadline: float,
        job_id: str | None = None,
        needs_inputs: bool = False,
        budget_s: float | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> Job:
        """Synchronous admission: checks and the counter increment happen atomically.
        ``budget_s`` overrides the generation budget (streaming requests get a longer one)."""
        overhead = self.settings.est_job_overhead_s
        scale = self.scale(model)
        warm = estimate_seconds(model, params, model.checkpoint, overhead, scale)
        budget = self.budget_s if budget_s is None else budget_s
        if warm > budget:
            param = "n" if params.n > 1 else "steps"
            raise request_too_expensive(
                f"This request is estimated to take {warm:.0f}s, more than the "
                f"{budget:.0f}s generation budget. Lower n, steps or size.",
                param,
            )
        wait, tail = self.estimated_wait()
        external = self.comfy_external_pending
        if self.active + external >= self.settings.queue_max_depth + 1:
            raise queue_full(wait)
        own = estimate_seconds(model, params, tail, overhead, scale)
        # Charge the time already spent since request receipt (body read, model refresh).
        remaining = gen_deadline - time.monotonic()
        if wait + own > remaining:
            # Cold start on an idle queue: nothing to wait for, so a 429 would only loop
            # (Retry-After 0). Admit when the warm estimate fits; the 504 is the safety net.
            if not (wait <= 0 and warm <= remaining):
                raise estimated_wait_exceeded(wait)
        loop = asyncio.get_running_loop()
        job = Job(
            id=job_id or uuid.uuid4().hex,
            model=model,
            params=params,
            graph=graph,
            gen_deadline=gen_deadline,
            est_s=own,
            est_wait_s=wait,
            cold_load_applied=model.checkpoint != tail,
            comfy_external_pending=external,
            future=loop.create_future(),
            inputs_ready=asyncio.Event() if needs_inputs else None,
            est_scale=scale,
            extra_data=extra_data,
        )
        self.active += 1
        self._pending.append(job)
        self._wakeup.set()
        return job

    def release(self, job: Job) -> bool:
        """Return the admission slot exactly once. Returns True if it took effect."""
        if job.released:
            return False
        job.released = True
        self.active -= 1
        return True

    def abandon(self, job: Job) -> None:
        """The route gave up on the job: stop waiting for it and skip it if still queued."""
        job.cancelled = True
        if job.inputs_ready is not None:
            job.inputs_ready.set()  # wake a worker waiting for the uploads
        if job.wait_task is not None and not job.wait_task.done():
            job.wait_task.cancel()
        if not job.future.done():
            job.future.cancel()
        elif not job.future.cancelled():
            job.future.exception()  # mark retrieved

    # ------------------------------------------------------------------ external load

    async def refresh_external(self) -> None:
        try:
            snap = await self.client.queue_snapshot()
        except ComfyError as exc:
            if self.comfy_reachable is not False:
                log.warning("comfy queue poll failed: %s", exc)
            self.comfy_reachable = not isinstance(exc, UpstreamUnavailable)
            self.comfy_external_pending = 0
            return
        self.comfy_reachable = True
        self.comfy_external_pending = snap.external_count

    async def run_poller(self) -> None:
        while True:
            try:
                await self.refresh_external()
            except Exception:
                log.exception("comfy queue poller failed")
            await asyncio.sleep(self.settings.comfy_queue_poll_s)

    # ------------------------------------------------------------------ worker

    async def run_worker(self) -> None:
        while True:
            while not self._pending:
                self._wakeup.clear()
                await self._wakeup.wait()
            job = self._pending.popleft()
            try:
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("worker failed on job %s", job.id)
                _set_exception(job, RuntimeError("worker failure"))

    async def _run(self, job: Job) -> None:
        if job.cancelled:
            job.state = "skipped"
            return
        if time.monotonic() >= job.gen_deadline:
            job.state = "expired"
            _set_exception(job, TimeoutError("deadline passed while queued"))
            return
        self.running = job
        job.run_started_at = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            if job.inputs_ready is not None and not job.inputs_ready.is_set():
                job.state = "awaiting_inputs"
                now = time.monotonic()
                upload_deadline = now + self.settings.upload_wait_s
                wait_until = min(job.gen_deadline, upload_deadline)
                try:
                    async with asyncio.timeout_at(loop.time() + (wait_until - now)):
                        await job.inputs_ready.wait()
                except TimeoutError:
                    job.state = "timeout"
                    if upload_deadline < job.gen_deadline:
                        exc: BaseException = UpstreamUnavailable(
                            "input upload did not complete within UPLOAD_WAIT_S"
                        )
                    else:
                        exc = TimeoutError("deadline passed while uploading inputs")
                    _set_exception(job, exc)
                    return
            if job.cancelled:
                job.state = "skipped"
                return
            job.state = "submitting"
            try:
                # a hung /prompt must not block the FIFO past this job's own deadline
                async with asyncio.timeout_at(loop.time() + (job.gen_deadline - time.monotonic())):
                    prompt_id = await self.client.submit(job.graph, extra_data=job.extra_data)
            except TimeoutError:
                job.state = "timeout"
                _set_exception(job, TimeoutError("deadline passed while submitting"))
                await self._cancel_orphans()
                return
            except ComfyError as exc:
                job.state = "failed"
                _set_exception(job, exc)
                if isinstance(exc, UpstreamUnavailable):
                    # the prompt may have been accepted before the connection dropped
                    await self._cancel_orphans()
                return
            job.prompt_id = prompt_id
            job.submitted = True
            job.submitted_at = time.monotonic()
            job.warm_at_submit = self.last_checkpoint == job.checkpoint
            self.last_checkpoint = job.checkpoint
            if job.cancelled:
                # the route left while /prompt was in flight: clean up ComfyUI ourselves
                await self._abandon_upstream(job)
                return
            job.state = "running"
            job.wait_task = asyncio.create_task(self.client.wait_for(prompt_id, job.gen_deadline))
            try:
                result = await job.wait_task
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise  # the worker itself is shutting down
                job.state = "abandoned"
                return
            except ComfyError as exc:
                job.state = "failed"
                job.finished_upstream = not isinstance(exc, UpstreamUnavailable)
                _set_exception(job, exc)
                return
            except TimeoutError as exc:
                job.state = "timeout"
                _set_exception(job, exc)
                return
            job.state = "done"
            job.finished_upstream = True
            self._observe(job, result)
            if not job.future.done():
                job.future.set_result(result)
        finally:
            self.running = None

    async def _abandon_upstream(self, job: Job) -> None:
        if job.prompt_id is None or not job.claim_cleanup():
            return
        job.state = "abandoned"
        try:
            async with asyncio.timeout(UPSTREAM_CLEANUP_TIMEOUT_S):
                await self.client.cancel(job.prompt_id)
                await self.client.delete_history(job.prompt_id)
        except Exception as exc:  # noqa: BLE001 - best effort
            log.warning("could not clean up abandoned prompt %s: %s", job.prompt_id, exc)
        finally:
            self.client.forget(job.prompt_id)

    def _tracked_prompt_ids(self) -> set[str]:
        jobs = [self.running, *self._pending]
        return {j.prompt_id for j in jobs if j is not None and j.prompt_id is not None}

    async def _cancel_orphans(self) -> None:
        """After a failed/timed-out /prompt: cancel owned prompts no job is tracking.

        One ``/queue`` snapshot; any entry carrying this boot's ``client_id`` whose prompt id
        no queued/running job knows is a prompt ComfyUI accepted after we stopped waiting.
        """
        try:
            async with asyncio.timeout(UPSTREAM_CLEANUP_TIMEOUT_S):
                snap = await self.client.queue_snapshot()
                tracked = self._tracked_prompt_ids()
                for entry in snap.owned:
                    if entry.prompt_id in tracked:
                        continue
                    log.warning("cancelling orphan prompt %s", entry.prompt_id)
                    try:
                        await self.client.cancel(entry.prompt_id)
                        await self.client.delete_history(entry.prompt_id)
                    finally:
                        self.client.forget(entry.prompt_id)
        except Exception as exc:  # noqa: BLE001 - best effort
            log.warning("orphan prompt check failed: %s", exc)

    # ------------------------------------------------------------------ introspection

    def snapshot(self) -> dict[str, Any]:
        wait, _ = self.estimated_wait()
        return {
            "active": self.active,
            "queued": len(self._queued()),
            "running_job_id": self.running.id if self.running else None,
            "comfy_external_pending": self.comfy_external_pending,
            "estimated_wait_s": round(wait, 2),
            "max_depth": self.settings.queue_max_depth,
            "estimate_calibration": {
                "enabled": self.settings.est_autocalibrate,
                "models": self.calibrator.snapshot(),
            },
        }


def _set_exception(job: Job, exc: BaseException) -> None:
    if not job.future.done():
        job.future.set_exception(exc)
        # the route may never await it (e.g. client gone); avoid "never retrieved" noise
        with contextlib.suppress(BaseException):
            job.future.add_done_callback(_retrieve)


def _retrieve(fut: asyncio.Future[Any]) -> None:
    if not fut.cancelled():
        fut.exception()
