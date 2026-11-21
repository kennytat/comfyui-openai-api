"""Phase 4: stream=true (SSE), partial images from ComfyUI previews, WebSocket events."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import struct
import time

import pytest
import websockets
from PIL import Image

from app.comfy.client import ComfyClient
from app.comfy.events import ComfyEvents
from app.routes.images import _partial_due
from tests.helpers import AUTH, make_png
from tests.test_openai_sdk_compat import sdk


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        ev = next((ln[7:] for ln in lines if ln.startswith("event: ")), None)
        data = next((ln[6:] for ln in lines if ln.startswith("data: ")), None)
        if ev and data:
            out.append((ev, json.loads(data)))
    return out


def _png_size(b64: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def _jpeg(color=(10, 200, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="JPEG")
    return buf.getvalue()


def _preview_frame(data: bytes) -> bytes:
    return struct.pack(">II", 1, 1) + data


def _progress(pid: str, value: int, maximum: int) -> str:
    return json.dumps(
        {"type": "progress", "data": {"value": value, "max": maximum, "prompt_id": pid}}
    )


# ------------------------------------------------------------------ SSE route


async def test_stream_generate_completed_events(gw):
    r = await gw.generate(stream=True, n=2, size="1536x1024")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.startswith(": stream started")
    evs = _events(r.text)
    assert [e for e, _ in evs] == ["image_generation.completed"] * 2
    for _, data in evs:
        assert data["type"] == "image_generation.completed"
        assert data["size"] == "1216x832" and data["output_format"] == "png"
        assert _png_size(data["b64_json"]) == (1216, 832)
        assert data["usage"]["total_tokens"] == 0
    assert "extra_data" not in gw.fake.prompt_bodies[-1]  # no previews requested
    assert gw.queue.active == 0


async def test_stream_edit_uses_image_edit_events(gw):
    r = await gw.edit(stream="true", quality="hd")
    assert r.status_code == 200
    evs = _events(r.text)
    assert [e for e, _ in evs] == ["image_edit.completed"]
    assert evs[0][1]["quality"] == "high"


async def test_stream_errors_before_and_after_start(gw):
    r = await gw.generate(stream=True, n=99)
    assert r.status_code == 400  # validation errors are plain JSON
    gw.fake.node_errors = True
    r = await gw.generate(stream=True)
    assert r.status_code == 200  # headers already sent: the error is an SSE event
    evs = _events(r.text)
    assert evs[-1][0] == "error" and evs[-1][1]["error"]["code"] == "upstream_error"
    assert gw.queue.active == 0
    text = (await gw.client.get("/metrics", headers=AUTH)).text
    assert 'gateway_errors_total{code="upstream_error",status="502"}' in text


async def test_stream_partial_images_from_previews(gw):
    events: ComfyEvents = gw.app.state.events
    gw.fake.hold = asyncio.Event()
    task = asyncio.create_task(gw.generate(stream=True, partial_images=2, output_format="webp"))
    await gw.wait_until(lambda: gw.fake.queue)
    pid = gw.fake.queue[0]
    assert gw.fake.prompt_bodies[-1]["extra_data"] == {"preview_method": "auto"}
    for step in range(1, 11):
        events.handle_text(_progress(pid, step, 10))
        events.handle_binary(_preview_frame(_jpeg((step * 20, 0, 0))))
        await asyncio.sleep(0.02)
    gw.fake.hold.set()
    r = await task
    evs = _events(r.text)
    kinds = [e for e, _ in evs]
    assert kinds == ["image_generation.partial_image"] * 2 + ["image_generation.completed"]
    assert [d["partial_image_index"] for _, d in evs[:2]] == [0, 1]
    img = Image.open(io.BytesIO(base64.b64decode(evs[0][1]["b64_json"])))
    assert img.format == "WEBP"
    assert events.state(pid) is None  # forgotten at cleanup


async def test_stream_keepalive_and_long_budget(gateway_factory):
    async with gateway_factory(stream_keepalive_s=0.05) as gw:
        gw.fake.hold = asyncio.Event()
        task = asyncio.create_task(gw.generate(stream=True))
        await gw.wait_until(lambda: gw.fake.queue)
        await asyncio.sleep(0.3)
        gw.fake.hold.set()
        assert ": keep-alive" in (await task).text
        # 25 steps * 0.75 s * 4 + 3 = 78 s fits 80 s; steps=60 does not, but fits the
        # stream budget (300 - 10 s)
        assert (await gw.generate(n=4, steps=60)).status_code == 400
        r = await gw.generate(n=4, steps=60, stream=True)
        assert r.status_code == 200 and len(_events(r.text)) == 4


async def test_stream_url_rejected_variation_rejected(gw):
    r = await gw.generate(stream=True, response_format="url")
    assert r.status_code == 400 and r.json()["error"]["param"] == "response_format"
    r = await gw.variation(stream="true")
    assert r.status_code == 400 and r.json()["error"]["param"] == "stream"


async def test_stream_with_official_sdk(gw):
    client = sdk(gw.app)
    stream = await client.images.generate(model="sdxl", prompt="a fox", stream=True, n=1)
    seen = [event async for event in stream]
    assert [e.type for e in seen] == ["image_generation.completed"]
    assert _png_size(seen[0].b64_json) == (1024, 1024)
    stream = await client.images.edit(
        model="sdxl", prompt="x", image=("a.png", make_png(), "image/png"), stream=True
    )
    assert [e.type async for e in stream] == ["image_edit.completed"]


async def test_stream_client_disconnect_cleans_up(gw):
    gw.fake.hold = asyncio.Event()
    task = asyncio.create_task(gw.generate(stream=True))
    await gw.wait_until(lambda: gw.fake.queue)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await gw.wait_until(lambda: gw.queue.active == 0)
    assert gw.fake.interrupts or gw.fake.queue_deletes


def test_partial_schedule():
    assert [s for s in range(1, 11) if _partial_due(s, 10, 0, 2)][0] == 4
    assert [s for s in range(1, 11) if _partial_due(s, 10, 1, 2)][0] == 7
    assert _partial_due(0, 0, 0, 3)


# ------------------------------------------------------------------ ComfyEvents


class _Client:
    base_url = "http://comfy.test:8188"
    client_id = "openai-api-x"

    def __init__(self) -> None:
        self.notified: list[tuple[str, bool]] = []

    def notify(self, pid: str, started: bool = False) -> None:
        self.notified.append((pid, started))


def test_events_dispatch():
    c = _Client()
    ev = ComfyEvents(c)
    assert ev.url == "ws://comfy.test:8188/ws?clientId=openai-api-x"
    c.base_url = "https://h"
    assert ev.url.startswith("wss://h/ws")
    ev.handle_text("not json")
    ev.handle_text(json.dumps({"type": "status", "data": {"status": {}}}))
    ev.handle_text(json.dumps({"type": "execution_start", "data": {"prompt_id": "p"}}))
    ev.handle_binary(_preview_frame(b"img1"))  # before any progress: attributed via start
    ev.handle_text(_progress("p", 3, 8))
    ev.handle_binary(_preview_frame(b"img2"))
    st = ev.state("p")
    assert (st.progress, st.max, st.preview.data, st.preview.step, st.preview.seq) == (
        3,
        8,
        b"img2",
        3,
        2,
    )
    meta = json.dumps({"prompt_id": "q", "image_type": "image/png"}).encode()
    ev.handle_binary(struct.pack(">II", 4, len(meta)) + meta + b"png!")
    assert ev.state("q").preview.mime == "image/png"
    ev.handle_binary(struct.pack(">II", 3, 0) + b"x")  # unknown binary type
    ev.handle_binary(b"xx")
    ev.handle_text(json.dumps({"type": "executing", "data": {"node": None, "prompt_id": "p"}}))
    ev.handle_text(json.dumps({"type": "execution_error", "data": {"prompt_id": "q"}}))
    assert c.notified == [("p", True), ("p", False), ("q", False)]
    ev.forget("p")
    assert ev.state("p") is None


async def test_wait_changed_wakes_and_times_out():
    ev = ComfyEvents(_Client())
    t = time.monotonic()
    await ev.wait_changed(0.05)
    assert time.monotonic() - t >= 0.04
    waiter = asyncio.create_task(ev.wait_changed(5))
    await asyncio.sleep(0.01)
    ev.handle_text(_progress("p", 1, 2))
    await asyncio.wait_for(waiter, 1)


async def test_events_run_against_real_websocket_server():
    received = asyncio.Event()

    async def handler(ws):
        assert ws.request.path.endswith("clientId=openai-api-x")
        await ws.send(_progress("p", 1, 4))
        await ws.send(_preview_frame(b"jpg"))
        await ws.send(json.dumps({"type": "execution_success", "data": {"prompt_id": "p"}}))
        received.set()
        await asyncio.sleep(1)

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        c = _Client()
        c.base_url = f"http://127.0.0.1:{port}"
        ev = ComfyEvents(c, reconnect_s=0.05)
        task = asyncio.create_task(ev.run())
        await asyncio.wait_for(received.wait(), 5)
        for _ in range(100):
            if c.notified:
                break
            await asyncio.sleep(0.01)
        assert ev.connected and c.notified == [("p", False)]
        assert ev.state("p").preview.data == b"jpg"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_events_reconnect_after_failure():
    c = _Client()
    c.base_url = "http://127.0.0.1:9"  # nothing listens
    ev = ComfyEvents(c, reconnect_s=0.01)
    task = asyncio.create_task(ev.run())
    await asyncio.sleep(0.1)
    assert not ev.connected and not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_notify_wakes_history_poll(fake):
    client = ComfyClient("http://comfy.test", transport=fake.transport, poll_interval_s=5.0)
    fake.hold = asyncio.Event()
    from app.comfy.workflows import GenerationParams, Registry, build_prompt
    from app.main import default_workflows_dir

    sdxl = Registry.load(default_workflows_dir()).get("sdxl")
    graph = build_prompt(sdxl, GenerationParams(prompt="x", width=64, height=64, seed=1))
    pid = await client.submit(graph, extra_data={"a": 1})
    assert fake.prompt_bodies[-1]["extra_data"] == {"a": 1}
    t = time.monotonic()
    waiter = asyncio.create_task(client.wait_for(pid, time.monotonic() + 30))
    await asyncio.sleep(0.05)
    client.notify(pid, started=True)  # records the start; wakes one poll (still held)
    await asyncio.sleep(0.05)
    fake.hold.set()
    client.notify(pid)
    result = await asyncio.wait_for(waiter, 2)
    assert time.monotonic() - t < 2 and result.started_at is not None
    client.forget(pid)
    await client.aclose()
