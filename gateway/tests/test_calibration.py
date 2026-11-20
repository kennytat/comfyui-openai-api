"""Phase 4: EWMA auto-calibration of job-duration estimates."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.comfy.client import WaitResult
from app.comfy.workflows import GenerationParams, compute_seconds, estimate_seconds
from app.errors import OpenAIError
from app.queue import EstimateCalibrator, JobQueue
from tests.helpers import make_settings
from tests.test_queue import _params, _Settings, registry  # noqa: F401 - fixture


def test_scale_is_neutral_until_min_samples():
    c = EstimateCalibrator(alpha=0.5, scale_min=0.75, scale_max=1.5, min_samples=3)
    assert c.scale("m") == 1.0
    c.observe("m", 10.0, 12.0)
    c.observe("m", 10.0, 12.0)
    assert c.scale("m") == 1.0
    c.observe("m", 10.0, 12.0)
    assert c.scale("m") == pytest.approx(1.2)  # steady ratio, no deviation
    assert c.scale("other") == 1.0


def test_scale_is_mean_plus_deviation_and_clamped():
    c = EstimateCalibrator(alpha=0.5, scale_min=0.75, scale_max=1.5, min_samples=1)
    c.observe("m", 10.0, 10.0)  # mean 1.0
    c.observe("m", 10.0, 12.0)  # dev 0.5*0.2 = 0.1, mean 1.1
    assert c.scale("m") == pytest.approx(1.2)
    snap = c.snapshot()["m"]
    assert snap == {"ratio_mean": 1.1, "ratio_dev": 0.1, "samples": 2, "scale": 1.2}
    for _ in range(20):
        c.observe("m", 10.0, 40.0)
    assert c.scale("m") == 1.5
    for _ in range(40):
        c.observe("m", 10.0, 1.0)
    assert c.scale("m") == 0.75


def test_invalid_observations_are_ignored():
    c = EstimateCalibrator(alpha=0.5, scale_min=0.75, scale_max=1.5, min_samples=1)
    c.observe("m", 0.0, 5.0)
    c.observe("m", 5.0, 0.0)
    assert c.snapshot() == {}


def test_estimate_scale_applies_to_sampling_only(registry):  # noqa: F811
    m = registry.get("sdxl")
    p = GenerationParams(prompt="x", width=1024, height=1024, n=2, quality="low")
    base = compute_seconds(m, p)
    assert base == pytest.approx(15 * 0.5 * 2)
    cold = estimate_seconds(m, p, None, overhead_s=3.0, scale=1.2)
    assert cold == pytest.approx(3 + base * 1.2 + m.cold_load_s)


def _result(gpu_s: float) -> WaitResult:
    return WaitResult(
        prompt_id="p", history={}, submitted_at=0.0, started_at=1.0, completed_at=1.0 + gpu_s
    )


def _job(model, params, warm=True):
    return SimpleNamespace(model=model, params=params, warm_at_submit=warm)


async def test_queue_learns_from_warm_jobs_and_changes_admission(registry):  # noqa: F811
    s = _Settings()
    s.queue_max_depth = 8
    q = JobQueue(s, client=None)  # type: ignore[arg-type]
    m = registry.get("sdxl")
    p = _params(m, n=4)  # 25 steps * 0.5 * 4 = 50 s sampling, warm estimate 53 s
    q.last_checkpoint = m.checkpoint
    assert q.admit(m, p, {}, time.monotonic() + 80).est_s == pytest.approx(53)
    # cold or missing timings are not learned from
    q._observe(_job(m, p, warm=False), _result(100.0))
    q._observe(_job(m, p), WaitResult("p", {}, 0.0, None, 5.0))
    assert q.calibrator.snapshot() == {}
    for _ in range(3):
        q._observe(_job(m, p), _result(80.0))  # really 1.6x slower -> clamped to 1.5
    assert q.scale(m) == 1.5
    assert q.snapshot()["estimate_calibration"]["models"]["sdxl"]["samples"] == 3
    with pytest.raises(OpenAIError) as exc:  # 3 + 50 * 1.5 = 78 s > 80 s - charged wait
        q.admit(m, p, {}, time.monotonic() + 80)
    assert exc.value.code in ("request_too_expensive", "estimated_wait_exceeded")
    s.est_autocalibrate = False
    assert q.scale(m) == 1.0


async def test_operation_extra_is_subtracted(registry):  # noqa: F811
    q = JobQueue(_Settings(), client=None)  # type: ignore[arg-type]
    m = registry.get("sdxl")
    p = _params(m, operation="edit", image="openai-api/x.png")
    est = compute_seconds(m, p)
    extra = m.operation("edit").est_extra_s
    q._observe(_job(m, p), _result(est + extra))
    assert q.calibrator.snapshot()["sdxl"]["ratio_mean"] == pytest.approx(1.0)


async def test_worker_feeds_calibration_end_to_end(gw):
    """Warm jobs completed by the worker are observed; the first (cold) one is not."""
    for _ in range(3):
        gw.fake.hold = asyncio.Event()
        task = asyncio.create_task(gw.generate(quality="low"))
        await gw.wait_until(lambda: gw.fake.queue)
        await asyncio.sleep(0.2)
        gw.fake.hold.set()
        r = await task
        assert r.status_code == 200, r.text
    stats = gw.queue.calibrator.snapshot()["sdxl"]
    assert stats["samples"] == 2 and stats["ratio_mean"] > 0
    snap = gw.queue.snapshot()["estimate_calibration"]
    assert snap["enabled"] is True and snap["models"]["sdxl"] == stats


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("est_ewma_alpha", 0, "EST_EWMA_ALPHA"),
        ("est_ewma_alpha", 1.5, "EST_EWMA_ALPHA"),
        ("est_scale_min", 1.1, "EST_SCALE_MIN"),
        ("est_scale_max", 0.9, "EST_SCALE_MIN"),
        ("est_calibration_min_samples", 0, "EST_CALIBRATION_MIN_SAMPLES"),
    ],
)
def test_calibration_settings_validated(tmp_path, field, value, match):
    with pytest.raises(ValidationError, match=match):
        make_settings(tmp_path, **{field: value})
