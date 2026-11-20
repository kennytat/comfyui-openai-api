"""``GET /v1/models`` and ``GET /v1/models/{id}`` plus the model availability catalog.

``ModelCatalog`` decides which registered models are *available*: registered, enabled and
every ``required_files`` entry listed by ComfyUI (``GET /object_info/<loader>``, both combo
formats). Results are cached for ``MODEL_CHECK_TTL_S``. When the cache is expired and a
refresh fails because ComfyUI is unreachable, ``UpstreamUnavailable`` propagates (503); no
stale list is served. ``require(model_id)`` forces one refresh (rate-limited to
``MODEL_REFRESH_MIN_INTERVAL_S``) before declaring a registered model unavailable (404).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from fastapi import APIRouter, Depends, Request

from app.auth import require_api_key
from app.comfy.client import ComfyClient, ComfyError, UpstreamUnavailable
from app.comfy.workflows import ModelEntry, Registry
from app.errors import model_not_found, upstream_unavailable
from app.schemas import Model, ModelList

log = logging.getLogger("gateway.models")

router = APIRouter(dependencies=[Depends(require_api_key)])


class ModelCatalog:
    def __init__(
        self,
        registry: Registry,
        client: ComfyClient,
        ttl_s: float,
        min_refresh_interval_s: float,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.registry = registry
        self.client = client
        self.ttl_s = ttl_s
        self.min_refresh_interval_s = min_refresh_interval_s
        self._now = monotonic
        self._available: set[str] | None = None
        self._fetched_at = 0.0
        self._last_forced: float | None = None
        self._lock = asyncio.Lock()

    def _fresh(self) -> bool:
        return self._available is not None and self._now() - self._fetched_at < self.ttl_s

    async def refresh(self, force: bool = False) -> set[str]:
        """Query ComfyUI (single-flight). Raises ``UpstreamUnavailable`` when unreachable.

        Concurrent callers queue on the lock; once inside, a caller reuses the result of a
        sweep that completed while it waited (fresh cache, or for ``force`` a sweep that
        finished after the call started) instead of sweeping again. An unreachable ComfyUI
        clears the cache, so no stale list is ever served afterwards.
        """
        requested_at = self._now()
        async with self._lock:
            if self._available is not None and (
                self._fetched_at >= requested_at if force else self._fresh()
            ):
                return self._available
            try:
                available = await self._sweep()
            except UpstreamUnavailable:
                self._available = None  # never serve a stale list
                raise
            self._available = available
            self._fetched_at = self._now()
            return available

    async def _sweep(self) -> set[str]:
        files: dict[tuple[str, str], set[str] | None] = {}
        available: set[str] = set()
        for model in self.registry.enabled():
            ok = True
            for req in model.required_files:
                key = (req.loader, req.input)
                if key not in files:
                    try:
                        values = await self.client.available_files(req.loader, req.input)
                    except UpstreamUnavailable:
                        raise
                    except ComfyError as exc:
                        log.warning("cannot list %s.%s: %s", req.loader, req.input, exc)
                        values = None
                    files[key] = set(values) if values is not None else None
                listed = files[key]
                if listed is None or req.value not in listed:
                    ok = False
                    break
            if ok:
                available.add(model.id)
        return available

    async def available(self) -> set[str]:
        if self._fresh():
            assert self._available is not None
            return self._available
        return await self.refresh()

    async def available_models(self) -> list[ModelEntry]:
        avail = await self.available()
        return [m for m in self.registry.enabled() if m.id in avail]

    async def require(self, model_id: str) -> ModelEntry:
        entry = self.registry.get(model_id)
        if entry is None or not entry.enabled:
            raise model_not_found(model_id)
        if model_id in await self.available():
            return entry
        now = self._now()
        if self._last_forced is None or now - self._last_forced >= self.min_refresh_interval_s:
            self._last_forced = now
            if model_id in await self.refresh(force=True):
                return entry
        raise model_not_found(model_id)


def _catalog(request: Request) -> ModelCatalog:
    return request.app.state.catalog


def _model(request: Request, entry: ModelEntry) -> Model:
    return Model(id=entry.id, created=request.app.state.created_ts)


async def _available_or_503(catalog: ModelCatalog) -> list[ModelEntry]:
    try:
        return await catalog.available_models()
    except UpstreamUnavailable as exc:
        raise upstream_unavailable() from exc


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    models = await _available_or_503(_catalog(request))
    return ModelList(data=[_model(request, m) for m in models])


@router.get("/v1/models/{model_id}", response_model=Model)
async def get_model(model_id: str, request: Request) -> Model:
    try:
        entry = await _catalog(request).require(model_id)
    except UpstreamUnavailable as exc:
        raise upstream_unavailable() from exc
    return _model(request, entry)
