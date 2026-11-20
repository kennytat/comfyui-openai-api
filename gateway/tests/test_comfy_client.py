"""ComfyClient tests against a respx-mocked ComfyUI (Step 1.2)."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from app.comfy.client import (
    CLIENT_ID_PREFIX,
    ComfyClient,
    UpstreamError,
    UpstreamInterrupted,
    UpstreamOutputMissing,
    UpstreamUnavailable,
    new_client_id,
)

BASE = "http://comfy.test"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE_PID = "11111111-1111-4111-8111-111111111111"
PID = "aaaaaaaa-0000-4000-8000-000000000001"
CID = "openai-api-11111111-2222-4333-8444-555555555555"
OTHER_CID = "openai-api-99999999-8888-4777-8666-555555555555"
GRAPH = {"1": {"class_type": "X", "inputs": {}}}


def history(name: str, pid: str = PID) -> dict:
    raw = (FIXTURES / name).read_text().replace(FIXTURE_PID, pid)
    return json.loads(raw)


def qentry(pid: str, client_id: str | None, number: int = 0) -> list:
    extra = {"client_id": client_id} if client_id is not None else {}
    return [number, pid, {}, extra, ["9"]]


def queue_body(running=(), pending=()) -> dict:
    return {"queue_running": list(running), "queue_pending": list(pending)}


@pytest.fixture
def router():
    return respx.Router(base_url=BASE, assert_all_called=False)


@pytest.fixture
async def client(router):
    c = ComfyClient(
        BASE, client_id=CID, transport=httpx.MockTransport(router.handler), poll_interval_s=0
    )
    yield c
    await c.aclose()


def body(call) -> dict:
    return json.loads(call.request.content)


def posted(route) -> list[dict]:
    return [body(c) for c in route.calls]


# ------------------------------------------------------------------ identity / submit


def test_new_client_id_format():
    cid = new_client_id()
    assert cid.startswith(CLIENT_ID_PREFIX)
    assert len(cid) == len(CLIENT_ID_PREFIX) + 36
    assert new_client_id() != cid


async def test_default_client_id_generated(router):
    c = ComfyClient(BASE, transport=httpx.MockTransport(router.handler))
    assert c.client_id.startswith(CLIENT_ID_PREFIX)
    await c.aclose()


async def test_submit_carries_boot_client_id(client, router):
    route = router.post("/prompt").respond(
        200, json={"prompt_id": PID, "number": 3, "node_errors": {}}
    )
    assert await client.submit(GRAPH) == PID
    assert posted(route) == [{"prompt": GRAPH, "client_id": CID}]


async def test_submit_node_errors_is_non_retryable_upstream_error(client, router):
    payload = json.loads((FIXTURES / "prompt_node_errors.json").read_text())
    router.post("/prompt").respond(400, json=payload)
    with pytest.raises(UpstreamError) as ei:
        await client.submit(GRAPH)
    err = ei.value
    assert type(err) is UpstreamError
    assert err.code == "upstream_error" and err.http_status == 502
    assert err.non_retryable is True and err.oom is False
    assert "4" in err.details["node_errors"]
    assert "Prompt outputs failed validation" in err.message


async def test_submit_connection_refused(client, router):
    router.post("/prompt").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(UpstreamUnavailable) as ei:
        await client.submit(GRAPH)
    assert ei.value.http_status == 503 and ei.value.code == "upstream_unavailable"


# ------------------------------------------------------------------ wait_for


async def test_success_flow_fetch_and_delete_history(client, router):
    router.post("/prompt").respond(200, json={"prompt_id": PID, "number": 1, "node_errors": {}})
    hist = iter([{}, {}, history("history_success.json")])
    hist_route = router.get(f"/history/{PID}").mock(
        side_effect=lambda req: httpx.Response(200, json=next(hist))
    )
    queues = iter([queue_body(pending=[qentry(PID, CID)]), queue_body(running=[qentry(PID, CID)])])
    router.get("/queue").mock(side_effect=lambda req: httpx.Response(200, json=next(queues)))
    view = router.get("/view").mock(
        side_effect=lambda req: httpx.Response(200, content=req.url.params["filename"].encode())
    )
    delete = router.post("/history").respond(200)

    pid = await client.submit(GRAPH)
    result = await client.wait_for(pid, deadline_monotonic=time.monotonic() + 5)
    assert hist_route.call_count == 3
    assert result.prompt_id == PID
    assert result.submitted_at is not None and result.started_at is not None
    assert result.submitted_at <= result.started_at <= result.completed_at
    assert result.comfy_wait_s is not None and result.gpu_s is not None
    assert result.execution_s == pytest.approx(12.5)

    images = client.output_images(result.history, "9")
    assert [i.filename for i in images] == ["job_00001_.png", "job_00002_.png"]
    blobs = await client.fetch_outputs(result.history, "9")
    assert blobs == [b"job_00001_.png", b"job_00002_.png"]
    params = dict(view.calls[0].request.url.params)
    assert params == {"filename": "job_00001_.png", "subfolder": "openai-api", "type": "output"}

    await client.delete_history(pid)
    assert posted(delete) == [{"delete": [PID]}]


async def test_started_at_none_when_never_seen_running(client, router):
    router.get(f"/history/{PID}").respond(200, json=history("history_success.json"))
    result = await client.wait_for(PID)
    assert result.started_at is None and result.gpu_s is None and result.comfy_wait_s is None


async def test_execution_error(client, router):
    router.get(f"/history/{PID}").respond(200, json=history("history_error.json"))
    delete = router.post("/history").respond(200)
    with pytest.raises(UpstreamError) as ei:
        await client.wait_for(PID)
    err = ei.value
    assert type(err) is UpstreamError
    assert err.code == "upstream_error" and not err.oom and not err.non_retryable
    assert "KSampler" in err.message and "RuntimeError" in err.message
    assert err.prompt_id == PID
    # Caller deletes the history entry after a failure too.
    await client.delete_history(PID)
    assert posted(delete) == [{"delete": [PID]}]


async def test_oom(client, router):
    router.get(f"/history/{PID}").respond(200, json=history("history_oom.json"))
    with pytest.raises(UpstreamError) as ei:
        await client.wait_for(PID)
    err = ei.value
    assert err.code == "upstream_oom" and err.oom is True and err.non_retryable is True
    assert err.http_status == 502


async def test_interrupted_externally(client, router):
    router.get(f"/history/{PID}").respond(200, json=history("history_interrupted.json"))
    with pytest.raises(UpstreamInterrupted) as ei:
        await client.wait_for(PID)
    err = ei.value
    assert err.code == "upstream_interrupted" and err.non_retryable is True


async def test_interrupted_by_gateway_cancel(client, router):
    router.get("/queue").respond(200, json=queue_body(running=[qentry(PID, CID)]))
    interrupt = router.post("/interrupt").respond(200)
    assert await client.cancel(PID) == "interrupted"
    assert posted(interrupt) == [{"prompt_id": PID}]
    router.get(f"/history/{PID}").respond(200, json=history("history_interrupted.json"))
    with pytest.raises(UpstreamInterrupted):
        await client.wait_for(PID)


async def test_lost_prompt_fails_fast_within_two_polls(client, router):
    hist = router.get(f"/history/{PID}").respond(200, json={})
    queue = router.get("/queue").respond(200, json=queue_body())
    with pytest.raises(UpstreamError) as ei:
        await client.wait_for(PID, deadline_monotonic=time.monotonic() + 30)
    assert type(ei.value) is UpstreamError
    assert ei.value.code == "upstream_error" and "neither" in ei.value.message
    assert hist.call_count == 2 and queue.call_count == 2


async def test_queue_to_history_handoff_race_is_tolerated(client, router):
    hist = iter([{}, history("history_success.json")])
    router.get(f"/history/{PID}").mock(side_effect=lambda req: httpx.Response(200, json=next(hist)))
    router.get("/queue").respond(200, json=queue_body())  # missing once, then in history
    result = await client.wait_for(PID)
    assert result.prompt_id == PID


async def test_missing_streak_resets_when_seen_again(client, router):
    hist = iter([{}, {}, {}, history("history_success.json")])
    router.get(f"/history/{PID}").mock(side_effect=lambda req: httpx.Response(200, json=next(hist)))
    queues = iter([queue_body(), queue_body(pending=[qentry(PID, CID)]), queue_body()])
    router.get("/queue").mock(side_effect=lambda req: httpx.Response(200, json=next(queues)))
    assert (await client.wait_for(PID)).prompt_id == PID


async def test_wait_for_deadline_raises_timeout(client, router):
    router.get(f"/history/{PID}").respond(200, json={})
    router.get("/queue").respond(200, json=queue_body(running=[qentry(PID, CID)]))
    client.poll_interval_s = 0.01
    with pytest.raises(TimeoutError):
        await client.wait_for(PID, deadline_monotonic=time.monotonic() + 0.05)


@pytest.mark.parametrize("hung", ["/history", "/queue"])
async def test_wait_for_hung_poll_cannot_overshoot_deadline(router, hung):
    never = asyncio.Event()

    async def hang(request):
        await never.wait()

    if hung == "/history":
        router.get(f"/history/{PID}").mock(side_effect=hang)
    else:
        router.get(f"/history/{PID}").respond(200, json={})
        router.get("/queue").mock(side_effect=hang)
    c = ComfyClient(BASE, client_id=CID, transport=httpx.MockTransport(router.async_handler))
    try:
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match="deadline exceeded"):
            await c.wait_for(PID, deadline_monotonic=time.monotonic() + 0.1)
        assert time.monotonic() - t0 < 2.0  # the 10 s per-request timeout never applied
    finally:
        await c.aclose()


async def test_forget_drops_bookkeeping(client):
    client._submitted_at[PID] = 1.0
    client.forget(PID)
    client.forget(PID)  # idempotent
    assert client._submitted_at == {}


async def test_wait_for_connection_refused(client, router):
    router.get(f"/history/{PID}").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(UpstreamUnavailable):
        await client.wait_for(PID)


# ------------------------------------------------------------------ outputs


async def test_view_404_is_output_missing(client, router):
    router.get("/view").respond(404)
    with pytest.raises(UpstreamOutputMissing) as ei:
        await client.fetch_outputs(history("history_success.json")[PID], "9")
    assert ei.value.code == "upstream_output_missing" and ei.value.http_status == 502


@pytest.mark.parametrize(
    "exc",
    [httpx.RemoteProtocolError("peer closed"), httpx.ReadError("reset"), httpx.ConnectError("x")],
)
async def test_view_connection_drop_is_unavailable(client, router, exc):
    router.get("/view").mock(side_effect=exc)
    with pytest.raises(UpstreamUnavailable):
        await client.fetch_outputs(history("history_success.json")[PID], "9")


async def test_fetch_outputs_rejects_foreign_subfolder(client, router):
    view = router.get("/view").respond(200, content=b"x")
    entry = history("history_success.json")[PID]
    entry["outputs"]["9"]["images"][0]["subfolder"] = "elsewhere"
    with pytest.raises(UpstreamError, match="outside"):
        await client.fetch_outputs(entry, "9")
    assert view.call_count == 0
    entry["outputs"]["9"]["images"][0]["subfolder"] = "openai-api-evil"
    with pytest.raises(UpstreamError, match="outside"):
        await client.fetch_outputs(entry, "9")


async def test_fetch_outputs_no_images(client):
    with pytest.raises(UpstreamOutputMissing):
        await client.fetch_outputs({"outputs": {}}, "9")


# ------------------------------------------------------------------ cancel


async def test_cancel_pending_deletes_from_queue(client, router):
    router.get("/queue").respond(200, json=queue_body(pending=[qentry(PID, CID)]))
    qdel = router.post("/queue").respond(200)
    interrupt = router.post("/interrupt").respond(200)
    assert await client.cancel(PID) == "deleted"
    assert posted(qdel) == [{"delete": [PID]}]
    assert interrupt.call_count == 0


async def test_cancel_running_rechecks_then_targeted_interrupt(client, router):
    queue = router.get("/queue").respond(200, json=queue_body(running=[qentry(PID, CID)]))
    interrupt = router.post("/interrupt").respond(200)
    qdel = router.post("/queue").respond(200)
    assert await client.cancel(PID) == "interrupted"
    assert queue.call_count == 2  # initial read + re-check immediately before /interrupt
    assert posted(interrupt) == [{"prompt_id": PID}]
    assert qdel.call_count == 0


async def test_cancel_running_finished_before_recheck(client, router):
    queues = iter([queue_body(running=[qentry(PID, CID)]), queue_body()])
    router.get("/queue").mock(side_effect=lambda req: httpx.Response(200, json=next(queues)))
    interrupt = router.post("/interrupt").respond(200)
    assert await client.cancel(PID) == "not_found"
    assert interrupt.call_count == 0


async def test_cancel_already_finished_sends_nothing(client, router):
    router.get("/queue").respond(200, json=queue_body(running=[qentry("other", "browser-ui")]))
    interrupt = router.post("/interrupt").respond(200)
    qdel = router.post("/queue").respond(200)
    assert await client.cancel(PID) == "not_found"
    assert interrupt.call_count == 0 and qdel.call_count == 0


# ------------------------------------------------------------------ queue snapshot


async def test_queue_snapshot_classification_by_client_id(client, router):
    router.get("/queue").respond(
        200,
        json=queue_body(
            running=[qentry("p-owned-run", CID, 1)],
            pending=[
                qentry("p-owned-pend", CID, 2),
                qentry("p-stale", OTHER_CID, 3),
                qentry("p-ui", "4a1c9f0e-browser", 4),
                qentry("p-noclient", None, 5),
                [6, "p-short"],
            ],
        ),
    )
    snap = await client.queue_snapshot()
    assert [e.prompt_id for e in snap.owned] == ["p-owned-run", "p-owned-pend"]
    assert [e.prompt_id for e in snap.stale_gateway] == ["p-stale"]
    assert [e.prompt_id for e in snap.external] == ["p-ui", "p-noclient", "p-short"]
    assert snap.external_count == 4
    assert snap.is_running("p-owned-run") and not snap.is_pending("p-owned-run")
    assert snap.is_pending("p-stale") and snap.contains("p-ui") and not snap.contains("zzz")
    assert snap.running[0].state == "running" and snap.running[0].client_id == CID


# ------------------------------------------------------------------ reconcile_startup


async def test_reconcile_startup_removes_only_stale_gateway_prompts(client, router):
    router.get("/queue").respond(
        200,
        json=queue_body(
            running=[qentry("stale-run", OTHER_CID, 1)],
            pending=[
                qentry("stale-pend", OTHER_CID, 2),
                qentry("ext-pend", "browser-client", 3),
                qentry("own-pend", CID, 4),
            ],
        ),
    )
    qdel = router.post("/queue").respond(200)
    interrupt = router.post("/interrupt").respond(200)
    hdel = router.post("/history").respond(200)

    report = await client.reconcile_startup()

    assert posted(qdel) == [{"delete": ["stale-pend"]}]
    assert posted(interrupt) == [{"prompt_id": "stale-run"}]
    assert posted(hdel) == [{"delete": ["stale-pend", "stale-run"]}]
    for route in (qdel, interrupt, hdel):
        for call in route.calls:
            assert b"ext-pend" not in call.request.content
            assert b"own-pend" not in call.request.content
    assert report.deleted_pending == ["stale-pend"]
    assert report.interrupted == ["stale-run"]
    assert not report.skipped and not report.errors


async def test_reconcile_startup_nothing_stale_makes_no_writes(client, router):
    router.get("/queue").respond(200, json=queue_body(pending=[qentry("ext", "ui")]))
    writes = router.post(path__regex=r".*").respond(200)
    report = await client.reconcile_startup()
    assert writes.call_count == 0 and report.deleted_pending == [] and report.interrupted == []


async def test_reconcile_startup_unreachable_completes(client, router):
    router.get("/queue").mock(side_effect=httpx.ConnectError("refused"))
    report = await client.reconcile_startup()
    assert report.skipped is True and report.errors


async def test_reconcile_startup_partial_failures_never_raise(client, router):
    router.get("/queue").respond(
        200,
        json=queue_body(running=[qentry("stale-run", OTHER_CID)], pending=[qentry("s", OTHER_CID)]),
    )
    router.post("/queue").respond(500)
    router.post("/interrupt").mock(side_effect=httpx.ConnectError("gone"))
    router.post("/history").respond(500)
    report = await client.reconcile_startup()
    assert len(report.errors) == 3 and report.deleted_pending == [] and report.interrupted == []


# ------------------------------------------------------------------ info / health


@pytest.mark.parametrize(
    "fixture", ["object_info_checkpoint_legacy.json", "object_info_checkpoint_combo.json"]
)
async def test_available_files_both_combo_formats(client, router, fixture):
    router.get("/object_info/CheckpointLoaderSimple").respond(
        200, json=json.loads((FIXTURES / fixture).read_text())
    )
    files = await client.available_files("CheckpointLoaderSimple", "ckpt_name")
    assert files == ["sd_xl_base_1.0.safetensors", "v1-5-pruned-emaonly.safetensors"]


async def test_available_files_unknown_shape_or_missing_class(client, router):
    router.get("/object_info/CheckpointLoaderSimple").respond(
        200, json={"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": ["COMBO", {}]}}}}
    )
    assert await client.available_files("CheckpointLoaderSimple", "ckpt_name") is None
    router.get("/object_info/UNETLoader").respond(200, json={})
    assert await client.available_files("UNETLoader", "unet_name") is None


async def test_available_files_unreachable(client, router):
    router.get("/object_info/CheckpointLoaderSimple").mock(side_effect=httpx.ConnectError("x"))
    with pytest.raises(UpstreamUnavailable):
        await client.available_files("CheckpointLoaderSimple", "ckpt_name")


async def test_health(client, router):
    router.get("/system_stats").respond(200, json={"system": {"os": "posix"}, "devices": []})
    assert (await client.health())["system"]["os"] == "posix"
    router.get("/system_stats").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(UpstreamUnavailable):
        await client.health()


async def test_unexpected_http_status_is_upstream_error(client, router):
    router.get("/queue").respond(500, text="boom")
    with pytest.raises(UpstreamError) as ei:
        await client.queue_snapshot()
    assert not isinstance(ei.value, UpstreamUnavailable)
