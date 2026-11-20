"""Per-API-key request rate limiting (token bucket) for the image endpoints.

``RateLimiter(rpm, burst, clock=time.monotonic)``: each key gets a bucket of ``burst``
tokens refilled at ``rpm / 60`` per second; ``take(key_id) -> float`` consumes one token and
returns ``0.0``, or the seconds until a token is available (nothing consumed). ``rpm <= 0``
disables limiting. Buckets live in memory (per process; reset on restart).

``enforce_rate_limit`` is a FastAPI dependency (after ``require_api_key``, which stores the
matched key's index in ``request.state.api_key_id``) that raises 429 ``rate_limit_exceeded``
with ``Retry-After``. Keys are identified by their position in ``API_KEYS``, never by value.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import Request

from app.errors import rate_limit_exceeded


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, rpm: float, burst: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = rpm / 60.0
        self.burst = max(1, burst)
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def take(self, key_id: str) -> float:
        if not self.enabled:
            return 0.0
        now = self.clock()
        b = self._buckets.get(key_id)
        if b is None:
            b = self._buckets[key_id] = _Bucket(float(self.burst), now)
        b.tokens = min(float(self.burst), b.tokens + (now - b.updated) * self.rate)
        b.updated = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return 0.0
        return (1.0 - b.tokens) / self.rate


async def enforce_rate_limit(request: Request) -> None:
    limiter: RateLimiter = request.app.state.rate_limiter
    key_id = str(getattr(request.state, "api_key_id", "unknown"))
    retry_after = limiter.take(key_id)
    if retry_after > 0:
        raise rate_limit_exceeded(retry_after)
