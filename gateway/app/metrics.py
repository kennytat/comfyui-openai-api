"""Prometheus metrics (``GET /metrics``, Bearer API key, ``ENABLE_METRICS``).

``Metrics(queue_getter)`` owns a private ``CollectorRegistry`` (one per app, so tests can
build many apps):

- ``gateway_http_requests_total{route,method,status}`` and
  ``gateway_http_request_duration_seconds{route,method}``: every HTTP request, labelled by
  the matched route template (``unmatched`` otherwise) - recorded by ``MetricsMiddleware``.
- ``gateway_jobs_total{model,operation,outcome}`` and
  ``gateway_job_duration_seconds{model,operation,phase}`` (``phase``: ``total``, ``gpu``,
  ``queue_wait``), ``gateway_images_total{model,operation}``: from each finished job's
  record (``observe_job``).
- ``gateway_errors_total{status,code}``: every OpenAI-style error response (admission
  400/429, rate limiting, 404, 5xx/504 after a job ran ...; ``error``).
- Scrape-time gauges from the live queue: ``gateway_queue_active``, ``gateway_queue_queued``,
  ``gateway_comfy_external_pending``, ``gateway_estimated_wait_seconds`` and
  ``gateway_estimate_scale{model}`` (learned calibration factor).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from starlette.types import ASGIApp, Message, Receive, Scope, Send

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
_JOB_BUCKETS = (1, 2.5, 5, 10, 20, 30, 45, 60, 75, 90, 120, 180, 300)


class _QueueCollector(Collector):
    def __init__(self, queue_getter: Callable[[], Any]) -> None:
        self.queue_getter = queue_getter

    def collect(self) -> Iterator[GaugeMetricFamily]:
        queue = self.queue_getter()
        if queue is None:
            return
        snap = queue.snapshot()
        for name, key, doc in (
            ("gateway_queue_active", "active", "Admitted jobs (queued + running)"),
            ("gateway_queue_queued", "queued", "Jobs waiting for the GPU worker"),
            (
                "gateway_comfy_external_pending",
                "comfy_external_pending",
                "ComfyUI jobs not submitted by this gateway",
            ),
            ("gateway_estimated_wait_seconds", "estimated_wait_s", "Estimated wait for a new job"),
        ):
            yield GaugeMetricFamily(name, doc, value=float(snap[key]))
        scale = GaugeMetricFamily(
            "gateway_estimate_scale", "Learned estimate correction factor", labels=["model"]
        )
        for model_id, stats in snap["estimate_calibration"]["models"].items():
            scale.add_metric([model_id], float(stats["scale"]))
        yield scale


class Metrics:
    def __init__(self, queue_getter: Callable[[], Any]) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.http_requests = Counter(
            "gateway_http_requests", "HTTP requests", ["route", "method", "status"], registry=r
        )
        self.http_duration = Histogram(
            "gateway_http_request_duration_seconds",
            "HTTP request duration",
            ["route", "method"],
            buckets=_JOB_BUCKETS,
            registry=r,
        )
        self.jobs = Counter(
            "gateway_jobs", "Finished jobs", ["model", "operation", "outcome"], registry=r
        )
        self.job_duration = Histogram(
            "gateway_job_duration_seconds",
            "Job duration by phase",
            ["model", "operation", "phase"],
            buckets=_JOB_BUCKETS,
            registry=r,
        )
        self.images = Counter(
            "gateway_images", "Images returned", ["model", "operation"], registry=r
        )
        self.errors = Counter(
            "gateway_errors", "OpenAI-style error responses", ["status", "code"], registry=r
        )
        r.register(_QueueCollector(queue_getter))

    def observe_job(self, record: dict[str, Any]) -> None:
        model, op = record.get("model", "?"), record.get("operation", "?")
        outcome = str(record.get("outcome", "unknown"))
        self.jobs.labels(model, op, outcome).inc()
        for phase, key in (
            ("total", "total_ms"),
            ("gpu", "gpu_ms"),
            ("queue_wait", "queue_wait_ms"),
        ):
            ms = record.get(key)
            if isinstance(ms, int | float):
                self.job_duration.labels(model, op, phase).observe(ms / 1000.0)
        if outcome == "ok":
            self.images.labels(model, op).inc(int(record.get("n", 1)))

    def error(self, status: int, code: str | None) -> None:
        self.errors.labels(str(status), code or "error").inc()

    def render(self) -> bytes:
        return generate_latest(self.registry)


class MetricsMiddleware:
    """Raw ASGI (streaming-safe): counts every HTTP request by route template and status."""

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        t0 = time.monotonic()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            method = scope.get("method", "?")
            self.metrics.http_requests.labels(path, method, str(status)).inc()
            self.metrics.http_duration.labels(path, method).observe(time.monotonic() - t0)
