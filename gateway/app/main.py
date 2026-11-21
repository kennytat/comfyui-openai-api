"""Application factory, lifespan wiring and the streamed body-size limit.

``create_app(settings=None, comfy_transport=None, clock=None, workflows_dir=None)``:

- ``settings``: defaults to ``load_settings()`` (environment).
- ``comfy_transport``: ``httpx.AsyncBaseTransport`` injected into ``ComfyClient`` (tests
  pass a respx router, leaving the ASGI transport untouched).
- ``clock``: wall clock (``() -> float``) for the image store expiry and ``created``
  timestamps; exposed as ``app.state.clock``.

The registry is loaded (and validated) at creation, so a broken workflow fails fast. The
lifespan never waits on ComfyUI: it starts the FIFO worker, the external-queue poller, the
TTL and orphan sweepers (ComfyUI output mount: ``ORPHAN_MAX_AGE_S``; input mount:
``INPUT_ORPHAN_MAX_AGE_S``, swept at least that often) and spawns ``reconcile_startup``
as a background task, so ``/healthz`` answers while ComfyUI is down or still booting.

``BodySizeLimitMiddleware`` is raw ASGI: it rejects a too-large ``Content-Length`` up front
and counts the bytes actually delivered by ``receive()``, so chunked bodies without a
length are cut at ``MAX_BODY_BYTES`` as well (413 ``request_too_large``). The multipart
routes (``/v1/images/edits``, ``/v1/images/variations``) get ``MAX_MULTIPART_BYTES``
instead; the per-file ``MAX_UPLOAD_BYTES`` is enforced by the route's parser. The path is
matched with ``root_path`` stripped and trailing slashes removed.

``create_app`` also sets Pillow's process-global decompression-bomb limit once
(``configure_pillow``), the preprocessing semaphore (``app.state.preprocess_sem``,
``MAX_CONCURRENT_PREPROCESS``) and the set of tracked background tasks
(``app.state.background_tasks``, cancelled on shutdown), the per-key rate limiter
(``app.state.rate_limiter``, ``RATE_LIMIT_RPM``) and, with ``ENABLE_METRICS``, the
Prometheus registry (``app.state.metrics``, ``GET /metrics``, ``MetricsMiddleware``).

The module attribute ``app`` (for ``uvicorn app.main:app``) is built lazily on first access.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.comfy.client import ComfyClient
from app.comfy.events import ComfyEvents
from app.comfy.workflows import Registry, parse_size
from app.config import Settings, load_settings
from app.errors import install_error_handlers, request_too_large
from app.metrics import Metrics, MetricsMiddleware
from app.queue import JobQueue
from app.ratelimit import RateLimiter
from app.routes import files, health, images, metrics, models
from app.routes.models import ModelCatalog
from app.storage import ImageStore, run_orphan_sweeper, run_ttl_sweeper
from app.uploads import configure_pillow

log = logging.getLogger("gateway")

TTL_SWEEP_INTERVAL_S = 60.0
ORPHAN_SWEEP_INTERVAL_S = 600.0
DISCONNECT_POLL_S = 1.0
INPUT_RETRY_DELAY_S = 2.0


def default_workflows_dir() -> Path:
    here = Path(__file__).resolve().parent.parent / "workflows"
    return here if here.is_dir() else Path.cwd() / "workflows"


class _BodyTooLarge(StarletteHTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=413)


def _route_path(scope: Scope) -> str:
    path: str = scope.get("path", "")
    root = scope.get("root_path", "").rstrip("/")
    if root and (path == root or path.startswith(root + "/")):
        path = path[len(root) :]
    return path.rstrip("/") or "/"


class BodySizeLimitMiddleware:
    def __init__(
        self, app: ASGIApp, max_bytes: int, path_limits: dict[str, int] | None = None
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.path_limits = {p.rstrip("/") or "/": v for p, v in (path_limits or {}).items()}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        scope.setdefault("state", {})["t0"] = time.monotonic()
        limit = self.path_limits.get(_route_path(scope), self.max_bytes)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    too_big = int(value) > limit
                except ValueError:
                    too_big = False
                if too_big:
                    await request_too_large(limit).response()(scope, receive, send)
                    return

        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    raise _BodyTooLarge()
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if exceeded:
                return  # swallow the app's own error response; ours is sent below
            started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded and not started:
            await request_too_large(limit).response()(scope, receive, send)


def _configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for name in ("gateway", "app"):
        logging.getLogger(name).setLevel(level.upper())


def create_app(
    settings: Settings | None = None,
    comfy_transport: httpx.AsyncBaseTransport | None = None,
    clock: Callable[[], float] | None = None,
    workflows_dir: str | Path | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    clock = clock or time.time
    _configure_logging(settings.log_level)
    registry = Registry.load(workflows_dir or default_workflows_dir())

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = ComfyClient(
            settings.comfyui_url,
            transport=comfy_transport,
            poll_interval_s=settings.poll_interval_s,
            output_subfolder=settings.comfy_output_subfolder,
            input_subfolder=settings.comfy_input_subfolder,
        )
        queue = JobQueue(settings, client)
        app.state.events = ComfyEvents(client)
        app.state.client = client
        app.state.queue = queue
        app.state.store = ImageStore(settings.image_store_dir, settings.image_url_ttl_s, clock)
        app.state.catalog = ModelCatalog(
            registry,
            client,
            settings.model_check_ttl_s,
            settings.model_refresh_min_interval_s,
        )
        app.state.created_ts = int(clock())
        log.info("gateway starting: client_id=%s comfyui=%s", client.client_id, client.base_url)
        tasks: list[asyncio.Task[Any]] = [
            asyncio.create_task(queue.run_worker(), name="job-worker"),
            asyncio.create_task(queue.run_poller(), name="comfy-queue-poller"),
            asyncio.create_task(
                run_ttl_sweeper(app.state.store, TTL_SWEEP_INTERVAL_S), name="ttl-sweeper"
            ),
            asyncio.create_task(client.reconcile_startup(), name="startup-reconcile"),
        ]
        if settings.comfy_ws:
            tasks.append(asyncio.create_task(app.state.events.run(), name="comfy-websocket"))
        # (dir, max age, interval); uploaded user images: always swept, and soon
        input_age = settings.input_orphan_max_age_s
        sweeps = [(settings.comfy_input_dir, input_age, min(ORPHAN_SWEEP_INTERVAL_S, input_age))]
        if settings.cleanup_comfy_outputs:
            sweeps.insert(
                0, (settings.comfy_output_dir, settings.orphan_max_age_s, ORPHAN_SWEEP_INTERVAL_S)
            )
        for sweep_dir, max_age, interval in sweeps:
            tasks.append(
                asyncio.create_task(
                    run_orphan_sweeper(sweep_dir, max_age, interval, clock),
                    name=f"orphan-sweeper:{sweep_dir}",
                )
            )
        try:
            yield
        finally:
            background = list(app.state.background_tasks)
            for task in [*tasks, *background]:
                task.cancel()
            await asyncio.gather(*tasks, *background, return_exceptions=True)
            await client.aclose()

    app = FastAPI(
        title="ComfyUI OpenAI-compatible Images API",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.clock = clock
    app.state.disconnect_poll_s = DISCONNECT_POLL_S
    app.state.input_retry_delay_s = INPUT_RETRY_DELAY_S
    app.state.background_tasks = set()
    app.state.preprocess_sem = asyncio.Semaphore(settings.max_concurrent_preprocess)
    app.state.rate_limiter = RateLimiter(
        settings.rate_limit_rpm, settings.rate_limit_burst or max(1, int(settings.rate_limit_rpm))
    )
    app.state.metrics = (
        Metrics(lambda: getattr(app.state, "queue", None)) if settings.enable_metrics else None
    )
    # Process-global, set once (thread-safe); the per-upload header check stays authoritative.
    # Never below the largest output size, which the encoder re-opens with Pillow too.
    largest_output = max(
        (w * h for m in registry.models for w, h in map(parse_size, m.sizes) if w and h),
        default=0,
    )
    configure_pillow(max(settings.max_input_pixels, largest_output))
    install_error_handlers(app)
    app.include_router(images.router)
    app.include_router(models.router)
    app.include_router(files.router)
    app.include_router(health.router)
    if app.state.metrics is not None:
        app.include_router(metrics.router)
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=settings.max_body_bytes,
        path_limits={path: settings.max_multipart_bytes for path in images.MULTIPART_PATHS},
    )
    if app.state.metrics is not None:
        app.add_middleware(MetricsMiddleware, metrics=app.state.metrics)
    return app


_app: FastAPI | None = None


def __getattr__(name: str) -> Any:
    """Lazily build ``app`` for ``uvicorn app.main:app`` (tests use ``create_app``)."""
    global _app
    if name == "app":
        if _app is None:
            _app = create_app()
        return _app
    raise AttributeError(name)
