"""Client disconnect handling, driving the ASGI app directly (Step 1.5, AC5)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from tests.helpers import KEY, set_model_estimates


def _scope(body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/images/generations",
        "raw_path": b"/v1/images/generations",
        "root_path": "",
        "query_string": b"",
        "headers": [
            (b"host", b"gateway.test"),
            (b"authorization", f"Bearer {KEY}".encode()),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("gateway.test", 80),
        "state": {},
    }


async def drive(app, trigger: Callable[[], Awaitable[None]]) -> list[dict]:
    """Send one generation request; report http.disconnect once ``trigger`` completes."""
    body = json.dumps({"prompt": "a red fox", "quality": "low"}).encode()
    sent: list[dict] = []
    first = True

    async def receive() -> dict:
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        await trigger()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await app(_scope(body), receive, send)
    return sent


async def test_disconnect_after_submit_cancels_upstream(gw):
    set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
    gw.fake.hold = asyncio.Event()  # job stays running in ComfyUI
    sent = await asyncio.wait_for(drive(gw.app, gw.fake.submitted.wait), 5)
    pid = next(iter(gw.fake.prompts))
    assert gw.fake.interrupts == [pid]  # targeted interrupt
    assert pid in gw.fake.history_deletes
    assert gw.queue.active == 0
    assert gw.fake.view_calls == 0
    assert not any(m.get("status") == 200 for m in sent)


async def test_disconnect_while_queued_never_submitted(gw):
    set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
    gw.fake.hold = asyncio.Event()
    first = asyncio.create_task(gw.generate())
    await gw.wait_until(lambda: gw.fake.prompt_count == 1)
    pid1 = next(iter(gw.fake.prompts))

    async def queued() -> None:
        while gw.queue.active < 2:
            await asyncio.sleep(0.005)

    released = []
    real_release = gw.queue.release
    gw.queue.release = lambda job: released.append(real_release(job)) or released[-1]

    await asyncio.wait_for(drive(gw.app, queued), 5)
    assert gw.queue.active == 1  # only job 1 remains
    assert released == [True]
    gw.fake.hold.set()
    r = await first
    assert r.status_code == 200
    await gw.wait_until(lambda: gw.queue.running is None and not gw.queue._pending)
    assert gw.fake.prompt_count == 1  # worker skipped the disconnected job
    assert gw.fake.interrupts == [] and gw.fake.queue_deletes == []
    assert gw.fake.history_deletes == [pid1]
    assert released == [True, True]  # one effective release per job
    assert gw.queue.active == 0


async def test_disconnect_before_submit_returns(gateway_factory):
    """Disconnect while /prompt is in flight: the worker cleans up once the id arrives."""
    async with gateway_factory() as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
        gw.fake.submit_gate = asyncio.Event()
        gw.fake.hold = asyncio.Event()

        async def in_flight() -> None:
            while gw.fake.prompt_count < 1:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(drive(gw.app, in_flight), 5)
        assert gw.queue.active == 0
        gw.fake.submit_gate.set()
        await gw.wait_until(lambda: len(gw.fake.history_deletes) == 1)
        pid = next(iter(gw.fake.prompts))
        assert gw.fake.interrupts == [pid]


async def test_hung_submit_then_disconnect_next_job_still_runs(gateway_factory):
    """/prompt accepted upstream but its reply never arrives; the client then leaves.

    The worker's submit is bounded by the job deadline, the accepted-but-unacknowledged
    prompt is cancelled as an orphan, and the next queued job runs normally.
    """
    async with gateway_factory(
        request_timeout_s=2.0, postprocess_reserve_s=0.5, est_job_overhead_s=0.05
    ) as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.0001, cold_load_s=0)
        gw.fake.prompt_reply_gate = asyncio.Event()  # never set: reply hangs forever
        await asyncio.wait_for(drive(gw.app, gw.fake.submitted.wait), 5)
        assert gw.queue.active == 0  # slot released on disconnect, worker still stuck
        orphan = next(iter(gw.fake.prompts))
        # setup margin, not an assertion bound: admit the next job well after the first so
        # its own (later) deadline leaves headroom once job 1's hung submit is given up
        await asyncio.sleep(0.7)
        r = await asyncio.wait_for(gw.generate(), 5)
        assert r.status_code == 200, r.text
        assert gw.fake.interrupts == [orphan]  # orphan cancelled (targeted interrupt)
        assert orphan in gw.fake.history_deletes
        assert gw.fake.prompt_count == 2
        assert gw.queue.active == 0
        assert gw.app.state.client._submitted_at == {}


async def test_submit_transport_error_cancels_accepted_orphan(gw):
    set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
    gw.fake.hold = asyncio.Event()  # the orphan would otherwise run forever
    gw.fake.prompt_reply_drop = True
    r = await gw.generate()
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "upstream_unavailable"
    orphan = next(iter(gw.fake.prompts))
    await gw.wait_until(lambda: orphan in gw.fake.history_deletes)
    assert gw.fake.interrupts == [orphan]
    assert gw.fake.queue == []
    assert gw.queue.active == 0
    gw.fake.hold.set()
    assert (await gw.generate()).status_code == 200  # the queue keeps working
