"""Admission, FIFO worker, deadline and slot-release tests (Step 1.3 / 1.5, AC5)."""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from app.comfy.workflows import GenerationParams, Registry, build_prompt, resolve_params
from app.errors import OpenAIError
from app.queue import JobQueue
from tests.helpers import AUTH, set_model_estimates

K = 2  # QUEUE_MAX_DEPTH used by the admission tests


async def _fire(gw, count: int, **body) -> list[asyncio.Task]:
    return [asyncio.create_task(gw.generate(quality="low", **body)) for _ in range(count)]


async def _first_done(tasks: list[asyncio.Task], timeout: float = 5.0) -> None:
    done, _ = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    assert done, "no request finished"


# ------------------------------------------------------------------ AC5 admission


async def test_exactly_one_queue_full(gateway_factory):
    async with gateway_factory(queue_max_depth=K) as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
        gw.fake.hold = asyncio.Event()
        tasks = await _fire(gw, K + 2)
        await _first_done(tasks)
        await gw.wait_until(lambda: gw.queue.active == K + 1)
        gw.fake.hold.set()
        responses = await asyncio.gather(*tasks)
        codes = sorted(r.status_code for r in responses)
        assert codes == [200] * (K + 1) + [429]
        rejected = next(r for r in responses if r.status_code == 429)
        assert rejected.json()["error"]["code"] == "queue_full"
        assert rejected.json()["error"]["type"] == "rate_limit_error"
        assert int(rejected.headers["retry-after"]) >= 1
        assert gw.fake.prompt_count == K + 1
        assert gw.queue.active == 0


async def test_estimated_wait_exceeded(gateway_factory):
    async with gateway_factory(queue_max_depth=8) as gw:
        # one 1024x1024 job at 25 steps ~ 1 + 25*2 = 51 s < 80 s budget; two do not fit
        set_model_estimates(gw.app, est_seconds_per_step=2.0, cold_load_s=0)
        gw.fake.hold = asyncio.Event()
        first = asyncio.create_task(gw.generate())
        await gw.wait_until(lambda: gw.fake.prompt_count == 1)
        r = await gw.generate()
        assert r.status_code == 429
        err = r.json()["error"]
        assert err["code"] == "estimated_wait_exceeded" and err["type"] == "rate_limit_error"
        wait = gw.queue.estimated_wait()[0]
        assert 40 < wait <= 51
        assert int(r.headers["retry-after"]) in (math.ceil(wait), math.ceil(wait) + 1)
        gw.fake.hold.set()
        assert (await first).status_code == 200
        assert gw.queue.active == 0


async def test_request_too_expensive(gw):
    r = await gw.generate(n=4, steps=80)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "request_too_expensive" and err["param"] == "n"
    r = await gw.generate(n=1, steps=80, size="1024x1024")
    # 1 + 80 * 0.5 = 41 s fits; make the per-step estimate expensive to hit param=steps
    assert r.status_code == 200
    set_model_estimates(gw.app, est_seconds_per_step=5.0)
    r = await gw.generate(steps=80)
    assert r.status_code == 400 and r.json()["error"]["param"] == "steps"
    assert gw.queue.active == 0


async def test_external_entry_shrinks_cap_and_grows_wait(gateway_factory):
    async with gateway_factory(queue_max_depth=K, est_external_job_s=30) as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
        await gw.queue.refresh_external()
        base_wait = gw.queue.estimated_wait()[0]
        gw.fake.add_external(client_id="comfy-ui-browser")
        await gw.queue.refresh_external()
        assert gw.queue.comfy_external_pending == 1
        assert gw.queue.estimated_wait()[0] == pytest.approx(base_wait + 30)
        assert gw.queue.snapshot()["comfy_external_pending"] == 1

        gw.fake.hold = asyncio.Event()
        tasks = await _fire(gw, K + 1)  # cap is now K: one of K+1 must be rejected
        await _first_done(tasks)
        gw.fake.external.clear()
        gw.fake.hold.set()
        codes = sorted(r.status_code for r in await asyncio.gather(*tasks))
        assert codes == [200] * K + [429]
        assert gw.queue.active == 0


async def test_stale_gateway_entry_counts_as_external(gw):
    gw.fake.add_external(client_id="openai-api-00000000-0000-4000-8000-000000000000")
    await gw.queue.refresh_external()
    assert gw.queue.comfy_external_pending == 1


async def test_owned_entries_never_external(gw):
    set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
    gw.fake.hold = asyncio.Event()
    task = asyncio.create_task(gw.generate())
    await gw.wait_until(lambda: gw.fake.prompt_count == 1)
    # the held prompt (running, gateway client_id) plus an owned pending one never count
    gw.fake.prompts["owned-pending"] = {
        "graph": {},
        "client_id": gw.app.state.client.client_id,
        "number": 99,
        "outcome": "success",
    }
    gw.fake.queue.append("owned-pending")
    await gw.queue.refresh_external()
    assert gw.queue.comfy_external_pending == 0
    gw.fake.queue.remove("owned-pending")
    gw.fake.hold.set()
    assert (await task).status_code == 200


async def test_poller_tracks_external_load(gw):
    gw.fake.add_external()
    await gw.wait_until(lambda: gw.queue.comfy_external_pending == 1)
    gw.fake.external.clear()
    await gw.wait_until(lambda: gw.queue.comfy_external_pending == 0)


async def test_poller_survives_comfy_down(gw):
    gw.fake.down = True
    await gw.queue.refresh_external()
    assert gw.queue.comfy_reachable is False and gw.queue.comfy_external_pending == 0


# ------------------------------------------------------------------ deadline / slot release

FAST = {"request_timeout_s": 1.0, "postprocess_reserve_s": 0.3, "est_job_overhead_s": 0.05}


def _fast(gw) -> None:
    set_model_estimates(gw.app, est_seconds_per_step=0.0001, cold_load_s=0)


async def test_running_job_504_interrupts_and_releases(gateway_factory):
    async with gateway_factory(queue_max_depth=K, **FAST) as gw:
        _fast(gw)
        gw.fake.hold = asyncio.Event()  # never set: job stays running
        # generous bound: the 1 s request deadline answers, not ComfyUI or a 10 s timeout
        r = await asyncio.wait_for(gw.generate(), 5)
        assert r.status_code == 504
        assert r.json()["error"]["code"] == "timeout"
        assert r.headers["x-should-retry"] == "false"
        pid = next(iter(gw.fake.prompts))
        assert gw.fake.interrupts == [pid]
        assert pid in gw.fake.history_deletes
        assert gw.queue.active == 0
        await _assert_no_leak(gw)


async def test_queued_job_504_behind_hung_submit(gateway_factory):
    async with gateway_factory(queue_max_depth=K, **FAST) as gw:
        _fast(gw)
        gw.fake.submit_gate = asyncio.Event()  # worker stuck submitting job 1
        r1, r2 = await asyncio.gather(gw.generate(), gw.generate())
        assert (r1.status_code, r2.status_code) == (504, 504)
        assert gw.queue.active == 0
        # the hung /prompt is bounded by job 1's deadline; the worker moves on
        await gw.wait_until(lambda: gw.queue.running is None and not gw.queue._pending)
        gw.fake.submit_gate.set()
        # job 2 shares job 1's deadline (+ epsilon): it is either skipped as expired or its
        # own /prompt is bounded the same way; nothing is ever accepted upstream
        assert 1 <= gw.fake.prompt_count <= 2
        assert gw.fake.prompts == {}
        assert gw.fake.interrupts == []
        assert gw.queue.snapshot()["queued"] == 0
        await _assert_no_leak(gw)


async def test_slow_postprocess_504(gateway_factory, monkeypatch):
    async with gateway_factory(**FAST) as gw:
        _fast(gw)
        gw.fake.view_gate = asyncio.Event()  # /view never answers within the budget
        r = await asyncio.wait_for(gw.generate(), 5)
        assert r.status_code == 504
        assert gw.fake.view_calls == 1  # the deadline fired while /view was in flight
        assert gw.fake.interrupts == []  # finished upstream: nothing to cancel
        assert len(gw.fake.history_deletes) == 1
        assert list(gw.fake.output_dir.iterdir()) == []
        assert gw.queue.active == 0


async def test_slow_encode_504(gateway_factory, monkeypatch):
    import threading

    import app.routes.images as images

    real = images.reencode
    entered, gate = threading.Event(), threading.Event()

    def slow(*a, **kw):
        entered.set()
        gate.wait(10)  # released only after the 504 was delivered
        return real(*a, **kw)

    monkeypatch.setattr(images, "reencode", slow)
    async with gateway_factory(**FAST) as gw:
        _fast(gw)
        try:
            r = await asyncio.wait_for(gw.generate(), 5)
            assert entered.is_set()
            assert r.status_code == 504  # answered while the encode thread was still busy
            assert gw.queue.active == 0
        finally:
            gate.set()


async def test_task_cancelled_mid_encode(gateway_factory, monkeypatch):
    import threading

    import app.routes.images as images

    entered, gate = threading.Event(), threading.Event()
    real = images.reencode

    def blocking(*a, **kw):
        entered.set()
        gate.wait(5)
        return real(*a, **kw)

    monkeypatch.setattr(images, "reencode", blocking)
    async with gateway_factory(queue_max_depth=K) as gw:
        _fast(gw)
        task = asyncio.create_task(gw.generate())
        await gw.wait_until(entered.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        gate.set()
        pid = next(iter(gw.fake.prompts))
        assert gw.fake.history_deletes == [pid]
        assert gw.fake.interrupts == []
        assert gw.queue.active == 0
        await _assert_no_leak(gw)


async def test_task_cancelled_while_running(gateway_factory):
    async with gateway_factory(queue_max_depth=K) as gw:
        _fast(gw)
        gw.fake.hold = asyncio.Event()
        task = asyncio.create_task(gw.generate())
        await gw.wait_until(lambda: gw.fake.prompt_count == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        pid = next(iter(gw.fake.prompts))
        assert gw.fake.interrupts == [pid]
        assert gw.fake.history_deletes == [pid]
        assert gw.queue.active == 0
        gw.fake.hold = None
        await _assert_no_leak(gw)


async def _assert_no_leak(gw) -> None:
    """k+1 concurrent requests are all admitted; the k+2-th would be rejected."""
    depth = gw.app.state.settings.queue_max_depth
    gw.fake.hold = asyncio.Event()
    gw.fake.submit_gate = None
    tasks = [asyncio.create_task(gw.generate()) for _ in range(depth + 1)]
    await gw.wait_until(lambda: gw.queue.active == depth + 1)
    extra = await gw.generate()
    assert extra.status_code == 429 and extra.json()["error"]["code"] == "queue_full"
    gw.fake.hold.set()
    assert all(r.status_code == 200 for r in await asyncio.gather(*tasks))
    assert gw.queue.active == 0


# ------------------------------------------------------------------ unit level


class _Settings:
    queue_max_depth = 1
    est_job_overhead_s = 3.0
    est_external_job_s = 30.0
    generation_budget_s = 80.0
    comfy_queue_poll_s = 0.01


def _params(model, **kw):
    return resolve_params(model, GenerationParams(prompt="x", width=1024, height=1024, **kw))


@pytest.fixture
def registry():
    import dataclasses

    from app.main import default_workflows_dir

    reg = Registry.load(default_workflows_dir())
    # Pin estimate parameters so production calibration in registry.json cannot break the math.
    reg._models["sdxl"] = dataclasses.replace(
        reg._models["sdxl"], est_seconds_per_step=0.5, cold_load_s=15
    )
    return reg


async def test_release_is_idempotent(registry):
    q = JobQueue(_Settings(), client=None)  # type: ignore[arg-type]
    m = registry.get("sdxl")
    p = _params(m, quality="low")
    job = q.admit(m, p, build_prompt(m, p), time.monotonic() + 80)
    assert q.active == 1
    assert q.release(job) is True
    assert q.release(job) is False
    assert q.active == 0


async def test_estimates_and_cold_load(registry):
    q = JobQueue(_Settings(), client=None)  # type: ignore[arg-type]
    m = registry.get("sdxl")
    p = _params(m, quality="low")
    j1 = q.admit(m, p, {}, time.monotonic() + 80)
    j2 = q.admit(m, p, {}, time.monotonic() + 80)
    assert j1.cold_load_applied and not j2.cold_load_applied
    assert j1.est_s == pytest.approx(3 + 15 * 0.5 + 15)
    assert j2.est_s == pytest.approx(3 + 15 * 0.5)
    assert j2.est_wait_s == pytest.approx(j1.est_s)
    with pytest.raises(OpenAIError) as exc:
        q.admit(m, p, {}, time.monotonic() + 80)
    assert exc.value.code == "queue_full"
    assert exc.value.headers["Retry-After"] == str(math.ceil(j1.est_s + j2.est_s))
    q.comfy_external_pending = 1
    assert q.estimated_wait()[0] == pytest.approx(j1.est_s + j2.est_s + 30)
    q.abandon(j2)
    assert q.snapshot()["queued"] == 1


async def test_admission_charges_elapsed_time(registry):
    q = JobQueue(_Settings(), client=None)  # type: ignore[arg-type]
    q.settings.queue_max_depth = 8
    m = registry.get("sdxl")
    p = _params(m, quality="low")  # warm 3 + 7.5 = 10.5, cold + 15 = 25.5
    first = q.admit(m, p, {}, time.monotonic() + 80)
    assert first.est_s == pytest.approx(25.5)
    # 25.5 + 10.5 fits the full 80 s budget but not the 20 s actually left for this request
    with pytest.raises(OpenAIError) as exc:
        q.admit(m, p, {}, time.monotonic() + 20)
    assert exc.value.code == "estimated_wait_exceeded"
    assert q.admit(m, p, {}, time.monotonic() + 79).est_s == pytest.approx(10.5)


async def test_cold_start_on_idle_queue_admitted(registry):
    q = JobQueue(_Settings(), client=None)  # type: ignore[arg-type]
    m = registry.get("sdxl")
    p = _params(m, quality="low")  # warm 10.5 fits 20 s; cold 25.5 does not
    job = q.admit(m, p, {}, time.monotonic() + 20)
    assert job.cold_load_applied and job.est_wait_s == 0
    q2 = JobQueue(_Settings(), client=None)  # type: ignore[arg-type]
    with pytest.raises(OpenAIError) as exc:  # not even the warm estimate fits
        q2.admit(m, p, {}, time.monotonic() + 5)
    assert exc.value.code == "estimated_wait_exceeded"


async def test_cold_start_heavy_request_admitted(gw):
    # n=4, steps=35: warm 1 + 35*0.5*4 = 71 s fits the 80 s budget; cold (+15) does not.
    # On an empty queue this must not be a retryable 429 loop (Retry-After 0).
    set_model_estimates(gw.app, est_seconds_per_step=0.5, cold_load_s=15)
    assert gw.queue.last_checkpoint is None
    r = await gw.generate(n=4, steps=35, size="1024x1024")
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 4
    assert gw.queue.active == 0


async def test_readyz_snapshot(gw):
    r = await gw.client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert set(body["queue"]) == {
        "active",
        "queued",
        "comfy_external_pending",
        "estimated_wait_s",
        "max_depth",
    }  # no running_job_id on the unauthenticated endpoint
    gw.fake.down = True
    gw.app.state.readyz_cache = None  # skip the cache window
    r = await gw.client.get("/readyz")
    assert r.status_code == 503 and r.json()["status"] == "not_ready"


async def test_readyz_comfy_check_cached(gw, monkeypatch):
    import app.routes.health as health

    def stats_calls() -> int:
        return sum(1 for _, p in gw.fake.requests if p == "/system_stats")

    responses = await asyncio.gather(*(gw.client.get("/readyz") for _ in range(5)))
    assert all(r.status_code == 200 for r in responses)
    assert stats_calls() == 1
    gw.fake.down = True  # within the window the cached result is served
    assert (await gw.client.get("/readyz")).status_code == 200
    assert stats_calls() == 1
    monkeypatch.setattr(health, "READY_CACHE_S", 0.0)  # window over: re-probe
    assert (await gw.client.get("/readyz")).status_code == 503


async def test_healthz_no_comfy_call(gw):
    gw.fake.down = True
    n = len(gw.fake.requests)
    r = await gw.client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    assert all(p != "/system_stats" for _, p in gw.fake.requests[n:])


async def test_public_routes_need_no_key(gw):
    assert (await gw.client.get("/images/" + "a" * 32 + ".png")).status_code == 404
    assert (await gw.client.get("/images/../etc/passwd")).status_code == 404
    assert (await gw.client.get("/images/notanid.png")).status_code == 404
    assert (await gw.client.get("/v1/models", headers=AUTH)).status_code == 200


async def test_timeout_mid_encode_leaves_no_hosted_files(gateway_factory, monkeypatch):
    """504 while the encode thread is still working: no hosted file survives it."""
    import threading

    import app.routes.images as images

    real_reencode, real_encode_all = images.reencode, images._encode_all
    second, gate, thread_done = threading.Event(), threading.Event(), threading.Event()
    calls = 0

    def reencode(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:  # first image already stored; hold the second past the deadline
            second.set()
            gate.wait(5)
        return real_reencode(*a, **kw)

    def encode_all(*a, **kw):
        try:
            return real_encode_all(*a, **kw)
        finally:
            thread_done.set()

    monkeypatch.setattr(images, "reencode", reencode)
    monkeypatch.setattr(images, "_encode_all", encode_all)
    async with gateway_factory(**FAST) as gw:
        _fast(gw)
        store = gw.app.state.store
        puts: list[str] = []
        real_put = store.put
        store.put = lambda data, ext: puts.append(real_put(data, ext)) or puts[-1]
        r = await gw.generate(n=2, response_format="url")
        assert r.status_code == 504
        assert second.is_set() and len(puts) == 1  # one file was stored before the 504
        gate.set()
        await gw.wait_until(thread_done.is_set)
        assert len(puts) == 1  # the thread saw the abort and stored nothing more
        assert [p for p in store.dir.iterdir() if not p.name.startswith(".")] == []
        assert gw.queue.active == 0


async def test_cleanup_forgets_prompt_bookkeeping(gw):
    from app.routes import images

    _fast(gw)
    assert (await gw.generate()).status_code == 200
    gw.fake.hold = asyncio.Event()
    task = asyncio.create_task(gw.generate())
    await gw.wait_until(lambda: gw.fake.prompt_count == 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await gw.wait_until(lambda: not images._CLEANUP_TASKS)
    assert gw.app.state.client._submitted_at == {}
    assert gw.queue.active == 0
