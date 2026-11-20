"""The official OpenAI SDK as acceptance oracle (Step 1.5; AC1, AC4, AC5).

``AsyncOpenAI`` talks to the gateway through an in-process ASGI transport inside
``LifespanManager``. The SDK ships its own httpx fork (``httpx2``) in recent releases; the
transport is taken from whichever module the installed SDK uses.
"""

from __future__ import annotations

import asyncio
import base64
import io

import openai
import pytest
from PIL import Image

from tests.helpers import KEY, set_model_estimates

try:  # openai>=3 depends on the httpx2 fork
    import httpx2 as sdk_httpx
except ImportError:  # pragma: no cover - older SDKs use httpx
    import httpx as sdk_httpx


def sdk(app, api_key: str = KEY, max_retries: int = 0) -> openai.AsyncOpenAI:
    http_client = sdk_httpx.AsyncClient(transport=sdk_httpx.ASGITransport(app=app))
    return openai.AsyncOpenAI(
        base_url="http://gateway.test/v1",
        api_key=api_key,
        http_client=http_client,
        max_retries=max_retries,
    )


async def test_generate_b64(gw):
    client = sdk(gw.app)
    resp = await client.images.generate(
        model="sdxl", prompt="a red fox", response_format="b64_json"
    )
    assert len(resp.data) == 1
    img = Image.open(io.BytesIO(base64.b64decode(resp.data[0].b64_json)))
    assert img.size == (1024, 1024) and img.format == "PNG"
    assert not img.text  # no tEXt/iTXt metadata
    assert resp.size == "1024x1024"
    assert resp.output_format == "png"


async def test_generate_url_and_extra_body(gw):
    client = sdk(gw.app)
    resp = await client.images.generate(
        model="sdxl",
        prompt="a red fox",
        response_format="url",
        size="1024x1792",
        n=2,
        extra_body={"seed": 7, "negative_prompt": "blurry", "steps": 9},
    )
    assert len(resp.data) == 2 and all(d.url for d in resp.data)
    assert resp.size == "768x1344"
    graph = gw.fake.prompt_bodies[0]["prompt"]
    assert graph["3"]["inputs"]["seed"] == 7 and graph["3"]["inputs"]["steps"] == 9


async def test_models_list(gw):
    models = await sdk(gw.app).models.list()
    assert [m.id for m in models.data] == ["sdxl"]
    assert (await sdk(gw.app).models.retrieve("sdxl")).id == "sdxl"


async def test_authentication_error(gw):
    with pytest.raises(openai.AuthenticationError) as exc:
        await sdk(gw.app, api_key="wrong").images.generate(model="sdxl", prompt="x")
    assert exc.value.code == "invalid_api_key"


async def test_not_found_error(gw):
    with pytest.raises(openai.NotFoundError) as exc:
        await sdk(gw.app).images.generate(model="nope", prompt="x")
    assert exc.value.code == "model_not_found"


async def test_bad_request_error(gw):
    with pytest.raises(openai.BadRequestError) as exc:
        await sdk(gw.app).images.generate(model="sdxl", prompt="x", size="333x333")
    assert exc.value.param == "size"


async def test_rate_limit_error(gateway_factory):
    async with gateway_factory(queue_max_depth=0) as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.01, cold_load_s=0)
        gw.fake.hold = asyncio.Event()
        held = asyncio.create_task(gw.generate())
        await gw.wait_until(lambda: gw.fake.prompt_count == 1)
        with pytest.raises(openai.RateLimitError) as exc:
            await sdk(gw.app).images.generate(model="sdxl", prompt="x")
        assert exc.value.code == "queue_full"
        assert exc.value.response.headers["retry-after"]
        gw.fake.hold.set()
        await held


@pytest.mark.parametrize(
    "setup,status,code",
    [
        (lambda f: f.outcomes.append("error"), 502, "upstream_error"),
        (lambda f: setattr(f, "down", True), 503, "upstream_unavailable"),
    ],
)
async def test_internal_server_errors(gw, setup, status, code):
    await gw.client.get("/v1/models", headers={"Authorization": f"Bearer {KEY}"})  # warm cache
    setup(gw.fake)
    with pytest.raises(openai.InternalServerError) as exc:
        await sdk(gw.app).images.generate(model="sdxl", prompt="x")
    assert exc.value.status_code == status and exc.value.code == code


FAST = {"request_timeout_s": 1.0, "postprocess_reserve_s": 0.3, "est_job_overhead_s": 0.05}


async def test_timeout_is_internal_server_error(gateway_factory):
    async with gateway_factory(**FAST) as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.0001, cold_load_s=0)
        gw.fake.hold = asyncio.Event()
        with pytest.raises(openai.InternalServerError) as exc:
            await sdk(gw.app).images.generate(model="sdxl", prompt="x")
        assert exc.value.status_code == 504 and exc.value.code == "timeout"


async def test_504_not_retried(gateway_factory):
    async with gateway_factory(**FAST) as gw:
        set_model_estimates(gw.app, est_seconds_per_step=0.0001, cold_load_s=0)
        gw.fake.hold = asyncio.Event()
        with pytest.raises(openai.InternalServerError):
            await sdk(gw.app, max_retries=2).images.generate(model="sdxl", prompt="x")
        assert gw.fake.prompt_count == 1  # x-should-retry: false honoured
        assert gw.queue.active == 0


# ------------------------------------------------------------------ Phase 2: edits/variations


def _png(size=(64, 64), color=(30, 90, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask() -> bytes:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    img.paste((0, 0, 0, 0), (16, 16, 48, 48))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _decoded(item) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(item.b64_json)))
    img.load()
    return img


async def test_images_edit(gw):
    resp = await sdk(gw.app).images.edit(
        image=("fox.png", _png(), "image/png"),
        prompt="the same fox wearing a red scarf",
        model="sdxl",
        response_format="b64_json",
        extra_body={"seed": 3, "strength": 0.6},
    )
    assert len(resp.data) == 1 and resp.size == "1024x1024"
    assert _decoded(resp.data[0]).size == (1024, 1024)
    graph = gw.fake.prompt_bodies[0]["prompt"]
    assert graph["3"]["inputs"]["denoise"] == 0.6 and graph["3"]["inputs"]["seed"] == 3


async def test_images_edit_with_mask_and_size(gw):
    resp = await sdk(gw.app).images.edit(
        image=("fox.png", _png(), "image/png"),
        mask=("mask.png", _mask(), "image/png"),
        prompt="a small campfire",
        size="1536x1024",
        n=2,
    )
    assert len(resp.data) == 2 and resp.size == "1216x832"
    assert all(_decoded(d).size == (1216, 832) for d in resp.data)
    assert "13" in gw.fake.prompt_bodies[0]["prompt"]  # LoadImageMask node present


async def test_images_edit_image_list(gw):
    resp = await sdk(gw.app).images.edit(image=[("a.png", _png(), "image/png")], prompt="x")
    assert len(resp.data) == 1
    with pytest.raises(openai.BadRequestError) as exc:
        await sdk(gw.app).images.edit(
            image=[("a.png", _png(), "image/png"), ("b.png", _png(), "image/png")], prompt="x"
        )
    assert exc.value.param == "image" and exc.value.code == "unsupported_value"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # SDK marks dall-e-2 variations
async def test_images_create_variation(gw):
    resp = await sdk(gw.app).images.create_variation(
        image=("fox.png", _png(), "image/png"), n=2, response_format="url"
    )
    assert len(resp.data) == 2 and all(d.url for d in resp.data)
    graph = gw.fake.prompt_bodies[0]["prompt"]
    assert graph["3"]["inputs"]["denoise"] == 0.55 and graph["12"]["inputs"]["amount"] == 2


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
async def test_images_edit_errors(gw):
    with pytest.raises(openai.BadRequestError) as exc:
        await sdk(gw.app).images.edit(
            image=("fox.png", _png(), "image/png"), mask=("m.png", _png(), "image/png"), prompt="x"
        )
    assert exc.value.param == "mask"
    with pytest.raises(openai.BadRequestError) as exc:
        await sdk(gw.app).images.create_variation(image=("x.png", b"nope", "image/png"))
    assert exc.value.param == "image"
