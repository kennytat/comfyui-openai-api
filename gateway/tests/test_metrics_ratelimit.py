"""Phase 4: Prometheus metrics and per-key rate limits."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ratelimit import RateLimiter
from tests.helpers import AUTH, KEY, FakeClock, make_settings

OTHER = {"Authorization": "Bearer other-key-456"}


# ------------------------------------------------------------------ rate limiter (unit)


def test_token_bucket_refills_at_rpm():
    clock = FakeClock(0.0)
    rl = RateLimiter(rpm=6, burst=2, clock=clock)  # one token every 10 s
    assert rl.take("a") == 0.0 and rl.take("a") == 0.0
    assert rl.take("a") == pytest.approx(10.0)
    assert rl.take("b") == 0.0  # independent bucket per key
    clock.advance(4)
    assert rl.take("a") == pytest.approx(6.0)  # nothing consumed by the refused call
    clock.advance(6)
    assert rl.take("a") == 0.0
    clock.advance(1000)  # capped at burst
    assert [rl.take("a") for _ in range(3)][-1] > 0


def test_disabled_limiter_never_limits():
    rl = RateLimiter(rpm=0, burst=1)
    assert not rl.enabled and all(rl.take("a") == 0.0 for _ in range(100))


def test_rate_limit_settings_validated(tmp_path):
    with pytest.raises(ValidationError, match="RATE_LIMIT_RPM"):
        make_settings(tmp_path, rate_limit_rpm=-1)


# ------------------------------------------------------------------ rate limiter (routes)


async def test_rate_limit_429_per_key(gateway_factory):
    async with gateway_factory(
        rate_limit_rpm=1, rate_limit_burst=2, api_keys_raw=f"{KEY},other-key-456"
    ) as gw:
        assert (await gw.generate()).status_code == 200
        assert (await gw.variation()).status_code == 200
        r = await gw.generate()
        assert r.status_code == 429
        body = r.json()["error"]
        assert (body["code"], body["type"]) == ("rate_limit_exceeded", "rate_limit_error")
        assert int(r.headers["Retry-After"]) >= 50
        assert len(gw.fake.prompt_bodies) == 2  # refused before any work
        r = await gw.client.post("/v1/images/generations", json={"prompt": "x"}, headers=OTHER)
        assert r.status_code == 200  # the other key has its own bucket
        # models/health are not rate limited
        assert (await gw.client.get("/v1/models", headers=AUTH)).status_code == 200


async def test_rate_limit_off_by_default(gw):
    for _ in range(5):
        assert (await gw.generate()).status_code == 200


async def test_rate_limit_runs_after_auth(gateway_factory):
    async with gateway_factory(rate_limit_rpm=1, rate_limit_burst=1) as gw:
        bad = {"Authorization": "Bearer nope"}
        for _ in range(3):
            r = await gw.client.post("/v1/images/generations", json={"prompt": "x"}, headers=bad)
            assert r.status_code == 401
        assert (await gw.generate()).status_code == 200


# ------------------------------------------------------------------ metrics


def _value(text: str, prefix: str) -> float:
    lines = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert lines, prefix
    return sum(float(ln.rsplit(" ", 1)[1]) for ln in lines)


async def test_metrics_require_auth(gw):
    assert (await gw.client.get("/metrics")).status_code == 401
    r = await gw.client.get("/metrics", headers=AUTH)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")


async def test_metrics_count_jobs_errors_and_requests(gw):
    assert (await gw.generate(n=2)).status_code == 200
    assert (await gw.generate(model="nope")).status_code == 404
    assert (await gw.generate(n=99)).status_code == 400
    text = (await gw.client.get("/metrics", headers=AUTH)).text
    labels = 'model="sdxl",operation="generate"'
    assert _value(text, f'gateway_jobs_total{{{labels},outcome="ok"}}') == 1
    assert _value(text, f"gateway_images_total{{{labels}}}") == 2
    assert _value(text, f'gateway_job_duration_seconds_count{{{labels},phase="total"}}') == 1
    assert _value(text, 'gateway_errors_total{code="model_not_found",status="404"}') == 1
    assert "gateway_errors_total{code=" in text and 'status="400"' in text
    route = 'gateway_http_requests_total{method="POST",route="/v1/images/generations"'
    assert _value(text, route) == 3
    assert _value(text, "gateway_queue_active ") == 0
    assert "gateway_estimated_wait_seconds" in text and "gateway_queue_queued" in text


async def test_metrics_rate_limited_and_scale_gauge(gateway_factory):
    async with gateway_factory(rate_limit_rpm=1, rate_limit_burst=1) as gw:
        await gw.generate()
        assert (await gw.generate()).status_code == 429
        gw.queue.calibrator.observe("sdxl", 10.0, 12.0)
        gw.queue.calibrator.min_samples = 1
        text = (await gw.client.get("/metrics", headers=AUTH)).text
        assert _value(text, 'gateway_errors_total{code="rate_limit_exceeded"') == 1
        assert _value(text, 'gateway_estimate_scale{model="sdxl"}') == pytest.approx(1.2)


async def test_metrics_disabled(gateway_factory):
    async with gateway_factory(enable_metrics=False) as gw:
        assert gw.app.state.metrics is None
        assert (await gw.client.get("/metrics", headers=AUTH)).status_code == 404
        assert (await gw.generate()).status_code == 200
