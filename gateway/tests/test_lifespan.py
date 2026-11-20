"""App factory / lifespan: never blocks on ComfyUI; startup reconciliation runs in background."""

from __future__ import annotations

import asyncio

import httpx
from asgi_lifespan import LifespanManager

from app.main import create_app
from tests.helpers import AUTH, make_settings


async def test_starts_while_comfy_down(tmp_path, fake, clock):
    fake.down = True
    app = create_app(make_settings(tmp_path), comfy_transport=fake.transport, clock=clock)
    async with LifespanManager(app, startup_timeout=2):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gw.test") as c:
            assert (await c.get("/healthz")).status_code == 200
            ready = await c.get("/readyz")
            assert ready.status_code == 503
            assert ready.json()["queue"]["active"] == 0
            assert (await c.get("/v1/models")).status_code == 401
            r = await c.get("/v1/models", headers=AUTH)
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "upstream_unavailable"
            fake.down = False  # ComfyUI comes up later: no gateway restart needed
            app.state.readyz_cache = None  # skip the READY_CACHE_S window
            assert (await c.get("/readyz")).status_code == 200
            assert (await c.get("/v1/models", headers=AUTH)).json()["data"][0]["id"] == "sdxl"


async def test_startup_reconciles_stale_prompts(tmp_path, fake, clock):
    stale = fake.add_external(client_id="openai-api-00000000-0000-4000-8000-000000000000")
    foreign = fake.add_external(client_id="browser-tab")
    app = create_app(make_settings(tmp_path), comfy_transport=fake.transport, clock=clock)
    async with LifespanManager(app):
        for _ in range(200):
            if stale in fake.history_deletes:
                break
            await asyncio.sleep(0.01)
        assert fake.queue_deletes == [stale]
        assert stale in fake.history_deletes
        assert foreign not in fake.queue_deletes and foreign not in fake.history_deletes


def test_lazy_module_app(monkeypatch, tmp_path):
    import app.main as main

    monkeypatch.setenv("API_KEYS", "k1")
    monkeypatch.setenv("ENABLE_URL_FORMAT", "false")
    monkeypatch.setenv("IMAGE_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_app", None)
    first = main.app
    assert main.app is first
    assert first.state.settings.api_keys == ["k1"]
