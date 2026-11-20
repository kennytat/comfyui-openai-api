"""GET /v1/models(/{id}) and the availability catalog (Step 1.5, AC6)."""

from __future__ import annotations

import asyncio

import pytest

from tests.helpers import AUTH


@pytest.mark.parametrize("mode", ["legacy", "combo"])
async def test_list_both_combo_formats(gw, mode):
    gw.fake.object_info_mode = mode
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["sdxl"]
    m = body["data"][0]
    assert m["object"] == "model" and m["owned_by"] == "comfyui-openai-api"
    assert m["created"] == 1_763_400_000


async def test_checkpoint_absent_empty_list(gw):
    gw.fake.object_info_mode = "absent"
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200 and r.json()["data"] == []


async def test_unparseable_object_info_unavailable(gw):
    gw.fake.object_info_mode = "error"  # HTTP 500 -> treated as unavailable, not 503
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200 and r.json()["data"] == []


async def test_get_model(gw):
    r = await gw.client.get("/v1/models/sdxl", headers=AUTH)
    assert r.status_code == 200 and r.json()["id"] == "sdxl"
    r = await gw.client.get("/v1/models/nope", headers=AUTH)
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "model_not_found" and err["param"] == "model"


async def test_requires_key(gw):
    assert (await gw.client.get("/v1/models")).status_code == 401
    assert (await gw.client.get("/v1/models/sdxl")).status_code == 401


async def test_cache_ttl(gw):
    await gw.client.get("/v1/models", headers=AUTH)
    await gw.client.get("/v1/models", headers=AUTH)
    assert gw.fake.object_info_calls == 1


async def test_unreachable_at_startup_503(gw):
    gw.fake.down = True
    for path in ("/v1/models", "/v1/models/sdxl"):
        r = await gw.client.get(path, headers=AUTH)
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "upstream_unavailable"
    gw.fake.down = False  # recovers without restart
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200 and [m["id"] for m in r.json()["data"]] == ["sdxl"]


async def test_no_stale_list_after_expiry(gateway_factory):
    async with gateway_factory(model_check_ttl_s=0.0) as gw:
        assert (await gw.client.get("/v1/models", headers=AUTH)).status_code == 200
        gw.fake.down = True
        assert (await gw.client.get("/v1/models", headers=AUTH)).status_code == 503


async def test_forced_refresh_on_not_found(gw):
    gw.fake.object_info_mode = "absent"
    r = await gw.generate()
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    calls = gw.fake.object_info_calls
    # checkpoint downloaded: the next request forces a refresh despite the fresh cache...
    gw.fake.object_info_mode = "legacy"
    gw.app.state.catalog._last_forced = -1e9  # outside the rate-limit window
    r = await gw.generate()
    assert r.status_code == 200, r.text
    assert gw.fake.object_info_calls == calls + 1


async def test_forced_refresh_rate_limited(gw):
    gw.fake.object_info_mode = "absent"
    assert (await gw.generate()).status_code == 404
    first = gw.fake.object_info_calls  # initial fill + one forced refresh
    assert first == 2
    assert (await gw.generate()).status_code == 404
    assert gw.fake.object_info_calls == first  # within MODEL_REFRESH_MIN_INTERVAL_S


async def test_unknown_model_no_refresh(gw):
    r = await gw.generate(model="dall-e-3")
    assert r.status_code == 404
    assert gw.fake.object_info_calls == 0


async def test_generation_503_when_down(gw):
    gw.fake.down = True
    r = await gw.generate()
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "upstream_unavailable"
    assert gw.queue.active == 0


async def test_concurrent_expired_cache_single_sweep(gateway_factory):
    async with gateway_factory(model_check_ttl_s=0.5) as gw:
        catalog = gw.app.state.catalog
        gw.fake.object_info_gate = asyncio.Event()
        tasks = [asyncio.create_task(gw.client.get("/v1/models", headers=AUTH)) for _ in range(5)]
        await gw.wait_until(lambda: gw.fake.object_info_calls == 1)
        await gw.wait_until(lambda: len(catalog._lock._waiters or ()) == 4)
        gw.fake.object_info_gate.set()
        responses = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in responses)
        assert all([m["id"] for m in r.json()["data"]] == ["sdxl"] for r in responses)
        assert gw.fake.object_info_calls == 1  # one /object_info sweep for all five


async def test_failed_forced_refresh_clears_cache(gw):
    gw.fake.object_info_mode = "absent"
    assert (await gw.client.get("/v1/models", headers=AUTH)).json()["data"] == []
    gw.fake.down = True  # the forced refresh in require() hits an unreachable ComfyUI
    r = await gw.generate()
    assert r.status_code == 503 and r.json()["error"]["code"] == "upstream_unavailable"
    assert gw.app.state.catalog._available is None  # same as available(): no stale list
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert r.status_code == 503


async def test_get_model_single_availability_check(gw):
    assert (await gw.client.get("/v1/models/sdxl", headers=AUTH)).status_code == 200
    assert gw.fake.object_info_calls == 1
    r = await gw.client.get("/v1/models/dall-e-3", headers=AUTH)
    assert r.status_code == 404 and gw.fake.object_info_calls == 1
