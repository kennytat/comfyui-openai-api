"""POST /v1/images/generations end to end against the fake ComfyUI (Step 1.5)."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from pathlib import Path

import pytest
from PIL import Image

from tests.helpers import AUTH, PUBLIC, set_model_estimates


def decode(item: dict) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
    img.load()
    return img


def png_chunks(data: bytes) -> list[bytes]:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, kinds = 8, []
    while pos < len(data):
        length = int.from_bytes(data[pos : pos + 4], "big")
        kinds.append(data[pos + 4 : pos + 8])
        pos += 12 + length
    return kinds


def graph_of(fake, index: int = -1) -> dict:
    return fake.prompt_bodies[index]["prompt"]


async def test_b64_default(gw):
    r = await gw.generate()
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"created", "data", "output_format", "size", "quality"}
    assert body["size"] == "1024x1024"
    assert body["output_format"] == "png"
    assert body["created"] == 1_763_400_000
    assert len(body["data"]) == 1
    item = body["data"][0]
    assert item["revised_prompt"] is None and "url" not in item
    img = decode(item)
    assert img.size == (1024, 1024) and img.format == "PNG"
    raw = base64.b64decode(item["b64_json"])
    chunks = png_chunks(raw)
    assert b"tEXt" not in chunks and b"iTXt" not in chunks and b"zTXt" not in chunks
    assert b"workflow" not in raw and b"secret" not in raw
    assert gw.queue.active == 0


async def test_prompt_body_carries_boot_client_id(gw):
    await gw.generate()
    cid = gw.app.state.client.client_id
    assert cid.startswith("openai-api-")
    assert gw.fake.prompt_bodies[0]["client_id"] == cid


async def test_url_format_and_fetch(gw, clock):
    r = await gw.generate(response_format="url")
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    m = re.fullmatch(re.escape(PUBLIC) + r"/images/([0-9a-f]{32})\.png", url)
    assert m, url
    path = url.removeprefix(PUBLIC)
    got = await gw.client.get(path)  # no auth needed
    assert got.status_code == 200
    assert got.headers["content-type"] == "image/png"
    assert got.headers["cache-control"].startswith("private, max-age=")
    assert got.headers["x-content-type-options"] == "nosniff"
    assert b"tEXt" not in png_chunks(got.content)
    clock.advance(3601)  # expiry on read, no sweeper run
    gone = await gw.client.get(path)
    assert gone.status_code == 404
    assert gone.json()["error"]["code"] == "not_found"


async def test_n2_distinct(gw):
    r = await gw.generate(n=2)
    data = r.json()["data"]
    assert len(data) == 2
    assert data[0]["b64_json"] != data[1]["b64_json"]
    assert graph_of(gw.fake)["5"]["inputs"]["batch_size"] == 2


async def test_n_too_large(gw):
    r = await gw.generate(n=5)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "n"
    assert gw.fake.prompt_count == 0


@pytest.mark.parametrize(
    "req,actual",
    [
        ("832x1216", "832x1216"),
        ("1024x1792", "768x1344"),
        ("1792x1024", "1344x768"),
        ("auto", "1024x1024"),
    ],
)
async def test_size_and_alias(gw, req, actual):
    r = await gw.generate(size=req)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["size"] == actual
    w, h = map(int, actual.split("x"))
    assert decode(body["data"][0]).size == (w, h)
    latent = graph_of(gw.fake)["5"]["inputs"]
    assert (latent["width"], latent["height"]) == (w, h)


async def test_invalid_size(gw):
    r = await gw.generate(size="333x333")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "size" and err["code"] == "invalid_size"


@pytest.mark.parametrize("quality,steps", [("low", 15), ("standard", 25), ("hd", 35), ("high", 35)])
async def test_quality_to_steps(gw, quality, steps):
    r = await gw.generate(quality=quality)
    assert r.status_code == 200
    assert r.json()["quality"] == quality
    assert graph_of(gw.fake)["3"]["inputs"]["steps"] == steps


async def test_explicit_steps_override_quality(gw):
    await gw.generate(quality="hd", steps=7)
    assert graph_of(gw.fake)["3"]["inputs"]["steps"] == 7


async def test_extensions_passthrough(gw):
    r = await gw.generate(seed=1234, negative_prompt="blurry", cfg_scale=4.5, prompt="a cat")
    assert r.status_code == 200
    g = graph_of(gw.fake)
    assert g["3"]["inputs"]["seed"] == 1234
    assert g["3"]["inputs"]["cfg"] == 4.5
    assert g["7"]["inputs"]["text"] == "blurry"
    assert g["6"]["inputs"]["text"] == "a cat"
    assert g["9"]["inputs"]["filename_prefix"].startswith("openai-api/")


async def test_random_seed_when_absent(gw):
    await gw.generate()
    await gw.generate()
    s1, s2 = (graph_of(gw.fake, i)["3"]["inputs"]["seed"] for i in (0, 1))
    assert 0 <= s1 <= 2**53 and s1 != s2


@pytest.mark.parametrize("fmt,pil", [("jpeg", "JPEG"), ("webp", "WEBP"), ("png", "PNG")])
async def test_output_formats(gw, fmt, pil):
    r = await gw.generate(output_format=fmt, output_compression=50)
    assert r.status_code == 200
    body = r.json()
    assert body["output_format"] == fmt
    assert decode(body["data"][0]).format == pil


async def test_jpeg_url_extension(gw):
    r = await gw.generate(output_format="jpeg", response_format="url")
    url = r.json()["data"][0]["url"]
    assert url.endswith(".jpeg")
    got = await gw.client.get(url.removeprefix(PUBLIC))
    assert got.headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize(
    "extra,param",
    [
        ({"stream": True, "response_format": "url"}, "response_format"),
        ({"background": "transparent"}, "background"),
    ],
)
async def test_unsupported_options(gw, extra, param):
    r = await gw.generate(**extra)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == param and err["code"] == "unsupported_value"
    assert gw.fake.prompt_count == 0


async def test_ignored_fields(gw):
    r = await gw.generate(
        style="vivid", user="u1", moderation="low", background="opaque", unknown_field=1
    )
    assert r.status_code == 200


async def test_url_disabled(gateway_factory):
    async with gateway_factory(enable_url_format=False, public_base_url="") as gw:
        r = await gw.generate(response_format="url")
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "response_format"
        assert (await gw.generate()).status_code == 200


async def test_default_response_format_url(gateway_factory):
    async with gateway_factory(default_response_format="url") as gw:
        r = await gw.generate()
        assert "url" in r.json()["data"][0]


async def test_outputs_and_history_cleaned(gw):
    r = await gw.generate(n=2)
    assert r.status_code == 200
    pid = next(iter(gw.fake.prompts))
    assert gw.fake.history_deletes == [pid]
    assert pid not in gw.fake.history
    assert list(gw.fake.output_dir.iterdir()) == []
    assert gw.fake.interrupts == [] and gw.fake.queue_deletes == []


async def test_cleanup_disabled_keeps_files(gateway_factory):
    async with gateway_factory(cleanup_comfy_outputs=False) as gw:
        await gw.generate()
        assert len(list(Path(gw.fake.output_dir).iterdir())) == 1
        assert len(gw.fake.history_deletes) == 1


async def test_job_log_line(gw, caplog):
    caplog.set_level(logging.INFO, logger="gateway.jobs")
    await gw.generate(prompt="very secret prompt text", seed=42, quality="low")
    lines = [r.getMessage() for r in caplog.records if r.name == "gateway.jobs"]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    for key in (
        "job_id",
        "prompt_id",
        "model",
        "n",
        "size",
        "steps",
        "seed",
        "queue_wait_ms",
        "comfy_wait_ms",
        "gpu_ms",
        "postprocess_ms",
        "total_ms",
        "est_job_s",
        "est_wait_s",
        "cold_load_applied",
        "comfy_external_pending",
        "outcome",
        "http_status",
    ):
        assert key in rec, key
    assert rec["outcome"] == "ok" and rec["http_status"] == 200
    assert rec["seed"] == 42 and rec["steps"] == 15
    assert rec["cold_load_applied"] is True
    assert "secret" not in lines[0] and AUTH["Authorization"] not in lines[0]


async def test_cold_load_only_on_checkpoint_change(gw, caplog):
    set_model_estimates(gw.app, cold_load_s=15)
    caplog.set_level(logging.INFO, logger="gateway.jobs")
    await gw.generate()
    await gw.generate()
    recs = [json.loads(r.getMessage()) for r in caplog.records if r.name == "gateway.jobs"]
    assert [r["cold_load_applied"] for r in recs] == [True, False]
    assert recs[0]["est_job_s"] - recs[1]["est_job_s"] == pytest.approx(15.0)
