"""POST /v1/images/edits and /v1/images/variations against the fake ComfyUI (Phase 2)."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import logging
import struct
import threading
import time
import zlib

import httpx
import pytest
from PIL import Image, ImageOps
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartParser

from app import uploads
from tests.helpers import AUTH, KEY, PUBLIC, make_png, set_model_estimates


def decode(item: dict) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
    img.load()
    return img


def err(r: httpx.Response) -> dict:
    return r.json()["error"]


def nodes(graph: dict, class_type: str) -> list[dict]:
    return [n["inputs"] for n in graph.values() if n["class_type"] == class_type]


def graph_of(fake, index: int = -1) -> dict:
    return fake.prompt_bodies[index]["prompt"]


def encode(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_mask(w: int = 64, h: int = 64, hole: float = 0.5) -> bytes:
    """Opaque RGBA PNG with a fully transparent centered square (the area to edit)."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    hw, hh = int(w * hole / 2), int(h * hole / 2)
    img.paste((0, 0, 0, 0), (w // 2 - hw, h // 2 - hh, w // 2 + hw, h // 2 + hh))
    return encode(img)


def bomb_png(width: int, height: int) -> bytes:
    """Tiny PNG whose IHDR claims ``width x height`` (decompression-bomb header)."""
    data = bytearray(make_png(4, 4))
    ihdr = struct.pack(">II", width, height) + bytes(data[24:29])
    data[16:29] = ihdr
    data[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr))
    return bytes(data)


def input_files(fake) -> list[str]:
    return sorted(p.name for p in fake.input_dir.iterdir())


def output_files(fake) -> list[str]:
    return sorted(p.name for p in fake.output_dir.iterdir())


def uploaded(fake, index: int) -> Image.Image:
    img = Image.open(io.BytesIO(fake.uploads[index]["data"]))
    img.load()
    return img


def fast_ops(gw) -> None:
    set_model_estimates(gw.app, est_seconds_per_step=0.0001, cold_load_s=0)
    entry = gw.app.state.registry.get("sdxl")
    ops = {k: dataclasses.replace(v, est_extra_s=0.0) for k, v in entry.operations.items()}
    set_model_estimates(gw.app, operations=ops)


# ------------------------------------------------------------------ edits: happy paths


async def test_edit_b64_img2img(gw):
    r = await gw.edit(prompt="the same fox wearing a red scarf", seed=11)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"created", "data", "output_format", "size", "quality"}
    assert body["size"] == "1024x1024" and len(body["data"]) == 1
    img = decode(body["data"][0])
    assert img.size == (1024, 1024) and not img.text
    graph = graph_of(gw.fake)
    job_image = nodes(graph, "LoadImage")[0]["image"]
    assert job_image.startswith("openai-api/") and job_image.endswith("_image.png")
    assert nodes(graph, "KSampler")[0]["denoise"] == 0.7
    assert nodes(graph, "KSampler")[0]["seed"] == 11
    assert nodes(graph, "RepeatLatentBatch")[0]["amount"] == 1
    assert not nodes(graph, "EmptyLatentImage") and not nodes(graph, "LoadImageMask")
    texts = [n["text"] for n in nodes(graph, "CLIPTextEncode")]
    assert "the same fox wearing a red scarf" in texts
    (up,) = gw.fake.uploads
    assert up["subfolder"] == "openai-api" and up["type"] == "input"
    assert up["overwrite"] == "true"
    assert job_image == f"openai-api/{up['filename']}"
    src = uploaded(gw.fake, 0)
    assert src.size == (1024, 1024) and src.mode == "RGB" and not src.text
    # inputs and outputs are gone once the response is delivered
    assert input_files(gw.fake) == [] and output_files(gw.fake) == []
    assert gw.queue.active == 0


async def test_edit_url_format(gw):
    r = await gw.edit(response_format="url")
    assert r.status_code == 200, r.text
    url = r.json()["data"][0]["url"]
    assert url.startswith(f"{PUBLIC}/images/")
    got = await gw.client.get(url.removeprefix(PUBLIC))
    assert got.status_code == 200 and got.headers["content-type"] == "image/png"


async def test_edit_with_mask_graph_and_alpha(gw):
    r = await gw.edit(mask=make_mask(), prompt="a small campfire")
    assert r.status_code == 200, r.text
    graph = graph_of(gw.fake)
    (mask_node,) = nodes(graph, "LoadImageMask")
    assert mask_node["channel"] == "alpha"
    assert mask_node["image"].endswith("_mask.png")
    (noise,) = nodes(graph, "SetLatentNoiseMask")
    mask_id = next(k for k, v in graph.items() if v["class_type"] == "LoadImageMask")
    assert noise["mask"] == [mask_id, 0]
    assert nodes(graph, "KSampler")[0]["denoise"] == 0.9
    names = [u["filename"] for u in gw.fake.uploads]
    assert names[0].endswith("_image.png") and names[1].endswith("_mask.png")
    assert names[0].split("_")[0] == names[1].split("_")[0]  # same job id
    mask = uploaded(gw.fake, 1)
    assert mask.mode == "RGBA" and mask.size == (1024, 1024)
    alpha = mask.getchannel("A")
    assert alpha.getpixel((512, 512)) == 0  # transparent = edit area preserved
    assert alpha.getpixel((5, 5)) == 255 and alpha.getpixel((1018, 1018)) == 255
    assert input_files(gw.fake) == []


async def test_mask_size_mismatch_is_resized(gw):
    image = encode(Image.new("RGB", (200, 100), (10, 120, 30)))
    r = await gw.edit(image=image, mask=make_mask(20, 20), size="1536x1024")
    assert r.status_code == 200, r.text
    assert r.json()["size"] == "1216x832"
    mask = uploaded(gw.fake, 1)
    assert mask.size == (1216, 832)
    alpha = mask.getchannel("A")
    assert alpha.getpixel((608, 416)) == 0 and alpha.getpixel((2, 2)) == 255


async def test_edit_n_via_repeat_latent_batch(gw):
    r = await gw.edit(n=3)
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 3
    assert nodes(graph_of(gw.fake), "RepeatLatentBatch")[0]["amount"] == 3


async def test_edit_size_alias_and_cover_crop(gw):
    # left half red, right half blue; 4:1 input cropped to a 3:2-ish bucket keeps both
    src = Image.new("RGB", (400, 100), (255, 0, 0))
    src.paste((0, 0, 255), (200, 0, 400, 100))
    r = await gw.edit(image=encode(src), size="1536x1024")
    assert r.status_code == 200, r.text
    assert r.json()["size"] == "1216x832"
    assert decode(r.json()["data"][0]).size == (1216, 832)
    up = uploaded(gw.fake, 0)
    assert up.size == (1216, 832)
    assert up.getpixel((10, 416))[0] > 200 and up.getpixel((1200, 416))[2] > 200


async def test_edit_invalid_size(gw):
    r = await gw.edit(size="333x333")
    assert r.status_code == 400 and err(r)["param"] == "size"
    assert gw.fake.uploads == []


@pytest.mark.parametrize("fmt", ["JPEG", "WEBP"])
async def test_edit_accepts_jpeg_webp(gw, fmt):
    r = await gw.edit(image=encode(Image.new("RGB", (80, 60), (1, 2, 3)), fmt))
    assert r.status_code == 200, r.text
    assert uploaded(gw.fake, 0).format == "PNG"


async def test_edit_rgba_image_is_flattened(gw):
    r = await gw.edit(image=encode(Image.new("RGBA", (32, 32), (255, 0, 0, 0))))
    assert r.status_code == 200, r.text
    up = uploaded(gw.fake, 0)
    assert up.mode == "RGB" and up.getpixel((5, 5)) == (128, 128, 128)


async def test_edit_jpeg_output_and_strength(gw):
    r = await gw.edit(output_format="jpeg", output_compression=50, strength=0.3, steps=7)
    assert r.status_code == 200, r.text
    assert r.json()["output_format"] == "jpeg"
    assert decode(r.json()["data"][0]).format == "JPEG"
    ks = nodes(graph_of(gw.fake), "KSampler")[0]
    assert ks["denoise"] == 0.3 and ks["steps"] == 7


async def test_image_array_field_single_ok(gw):
    files = [("image[]", ("a.png", make_png(), "image/png"))]
    r = await gw.edit(files=files)
    assert r.status_code == 200, r.text


@pytest.mark.parametrize(
    "names", [("image[]", "image[]"), ("image", "image[]"), ("image", "image")]
)
async def test_more_than_one_image_rejected(gw, names):
    files = [(n, (f"{i}.png", make_png(), "image/png")) for i, n in enumerate(names)]
    r = await gw.edit(files=files)
    assert r.status_code == 400
    assert err(r)["code"] == "unsupported_value" and err(r)["param"] == "image"
    assert gw.fake.uploads == []


async def test_job_log_line(gw, caplog):
    caplog.set_level(logging.INFO, logger="gateway.jobs")
    r = await gw.edit(mask=make_mask())
    assert r.status_code == 200
    record = json.loads(caplog.records[-1].getMessage())
    assert record["operation"] == "edit_masked" and record["denoise"] == 0.9
    assert record["inputs_deleted"] == 2 and record["outcome"] == "ok"
    assert "upload_ms" in record and "prompt" not in record


# ------------------------------------------------------------------ edits: validation


@pytest.mark.parametrize(
    "image,code",
    [
        (b"not an image at all", "invalid_image_format"),
        (b"GIF89a" + b"\x00" * 64, "invalid_image_format"),
        (b"", "invalid_image"),
        (make_png()[:-30], "invalid_image"),
    ],
    ids=["text", "gif-header", "empty", "truncated"],
)
async def test_invalid_image_bytes(gw, image, code):
    r = await gw.edit(image=image)
    assert r.status_code == 400, r.text
    assert err(r)["param"] == "image" and err(r)["code"] == code
    assert gw.fake.uploads == [] and gw.fake.prompt_count == 0
    assert gw.queue.active == 0


async def test_gif_rejected(gw):
    r = await gw.edit(image=encode(Image.new("RGB", (8, 8)), "GIF"))
    assert r.status_code == 400 and err(r)["code"] == "invalid_image_format"


async def test_decompression_bomb_header(gw):
    r = await gw.edit(image=bomb_png(100_000, 100_000))
    assert r.status_code == 400
    assert err(r)["param"] == "image" and err(r)["code"] == "image_too_large"


async def test_pixel_limit(gateway_factory):
    async with gateway_factory(max_input_pixels=1000) as gw:
        r = await gw.edit(image=make_png(64, 64))
        assert r.status_code == 400 and err(r)["code"] == "image_too_large"
        r = await gw.edit(image=make_png(16, 16), mask=make_mask(64, 64))
        assert r.status_code == 400 and err(r)["param"] == "mask"


async def test_mask_without_alpha(gw):
    r = await gw.edit(mask=make_png(64, 64))
    assert r.status_code == 400
    assert err(r)["param"] == "mask" and err(r)["code"] == "invalid_value"
    assert gw.fake.uploads == [] and gw.fake.prompt_count == 0


async def test_mask_must_be_png(gw):
    mask = encode(Image.new("RGBA", (64, 64), (0, 0, 0, 0)), "WEBP")
    r = await gw.edit(mask=mask)
    assert r.status_code == 400 and err(r)["param"] == "mask"


async def test_mask_palette_transparency_ok(gw):
    img = Image.new("P", (32, 32), 0)
    img.info["transparency"] = 0
    r = await gw.edit(mask=encode(img))
    assert r.status_code == 200, r.text
    assert uploaded(gw.fake, 1).getchannel("A").getpixel((3, 3)) == 0


async def test_missing_prompt(gw):
    files = [("image", ("a.png", make_png(), "image/png"))]
    r = await gw.client.post("/v1/images/edits", files=files, headers=AUTH)
    assert r.status_code == 400
    assert err(r)["param"] == "prompt" and err(r)["code"] == "missing_required_parameter"


async def test_missing_image(gw):
    r = await gw.client.post(
        "/v1/images/edits", data={"prompt": "x"}, files={"other": b"x"}, headers=AUTH
    )
    assert r.status_code == 400
    assert err(r)["param"] == "image" and err(r)["code"] == "missing_required_parameter"


async def test_image_as_text_field(gw):
    r = await gw.client.post(
        "/v1/images/edits",
        data={"prompt": "x", "image": "https://example.com/a.png"},
        files={"other": b"x"},
        headers=AUTH,
    )
    assert r.status_code == 400 and err(r)["param"] == "image"


async def test_json_body_rejected(gw):
    r = await gw.client.post("/v1/images/edits", json={"prompt": "x"}, headers=AUTH)
    assert r.status_code == 400 and err(r)["code"] == "invalid_content_type"


async def test_malformed_multipart(gw):
    r = await gw.client.post(
        "/v1/images/edits",
        content=b"--zzz\r\ngarbage",
        headers={**AUTH, "content-type": "multipart/form-data"},  # no boundary
    )
    assert r.status_code == 400 and err(r)["code"] == "invalid_value"
    assert "multipart" in err(r)["message"]


async def test_edits_require_auth(gw):
    r = await gw.client.post("/v1/images/edits", files={"image": make_png()})
    assert r.status_code == 401


@pytest.mark.parametrize(
    "fields,param",
    [
        ({"stream": "true"}, "stream"),
        ({"partial_images": "2"}, "partial_images"),
        ({"background": "transparent"}, "background"),
    ],
)
async def test_unsupported_options(gw, fields, param):
    r = await gw.edit(**fields)
    assert r.status_code == 400
    assert err(r)["code"] == "unsupported_value" and err(r)["param"] == param


@pytest.mark.parametrize(
    "fields,param",
    [
        ({"strength": "0.01"}, "strength"),
        ({"n": "0"}, "n"),
        ({"n": "9"}, "n"),
        ({"quality": "ultra"}, "quality"),
        ({"prompt": ""}, "prompt"),
    ],
)
async def test_invalid_fields(gw, fields, param):
    r = await gw.edit(**fields)
    assert r.status_code == 400 and err(r)["param"] == param


async def test_url_disabled(gateway_factory):
    async with gateway_factory(enable_url_format=False) as gw:
        r = await gw.edit(response_format="url")
        assert r.status_code == 400 and err(r)["param"] == "response_format"


async def test_model_without_operation(gw):
    set_model_estimates(gw.app, operations={})
    r = await gw.edit()
    assert r.status_code == 400
    assert err(r)["code"] == "unsupported_value" and err(r)["param"] == "model"
    r = await gw.variation()
    assert r.status_code == 400 and err(r)["param"] == "model"
    assert (await gw.generate()).status_code == 200  # generations unaffected


async def test_unknown_model(gw):
    r = await gw.edit(model="nope")
    assert r.status_code == 404 and err(r)["code"] == "model_not_found"


# ------------------------------------------------------------------ size limits


async def test_oversize_file_413(gateway_factory):
    async with gateway_factory(max_upload_bytes=2000) as gw:
        big = encode(Image.effect_noise((64, 64), 100).convert("RGB"))
        assert len(big) > 2000
        r = await gw.edit(image=big)
        assert r.status_code == 413
        assert err(r)["code"] == "request_too_large" and err(r)["param"] == "image"
        assert gw.fake.uploads == []


async def test_oversize_mask_413(gateway_factory):
    async with gateway_factory(max_upload_bytes=3000) as gw:
        noise = Image.effect_noise((64, 64), 100).convert("L")
        mask = encode(Image.merge("RGBA", (noise, noise, noise, noise)))
        r = await gw.edit(image=make_png(8, 8), mask=mask)
        assert r.status_code == 413 and err(r)["param"] == "mask"


async def test_oversized_multipart_content_length_413(gateway_factory):
    async with gateway_factory(max_upload_bytes=4000, max_multipart_bytes=6000) as gw:
        r = await gw.edit(image=make_png(8, 8), prompt="x" * 7000)
        assert r.status_code == 413 and err(r)["code"] == "request_too_large"
        assert gw.fake.uploads == []


async def test_oversized_multipart_streamed_413(gateway_factory):
    async with gateway_factory(max_upload_bytes=4000, max_multipart_bytes=6000) as gw:
        req = httpx.Request(
            "POST",
            "http://gateway.test/v1/images/edits",
            data={"prompt": "x" * 7000},
            files={"image": ("a.png", make_png(8, 8), "image/png")},
        )
        payload = req.read()

        async def chunks():
            for i in range(0, len(payload), 1000):
                yield payload[i : i + 1000]

        r = await gw.client.post(
            "/v1/images/edits",
            content=chunks(),
            headers={**AUTH, "content-type": req.headers["content-type"]},
        )
        assert "content-length" not in {k.lower() for k in r.request.headers}
        assert r.status_code == 413 and err(r)["code"] == "request_too_large"
        assert gw.fake.uploads == []


async def test_multipart_limit_does_not_apply_to_json(gateway_factory):
    async with gateway_factory(max_body_bytes=1000) as gw:
        r = await gw.edit(prompt="x" * 3000)  # multipart route: allowed
        assert r.status_code == 200, r.text
        r = await gw.generate(prompt="x" * 3000)  # JSON route: still 1000 bytes
        assert r.status_code == 413


# ------------------------------------------------------------------ variations


async def test_variation(gw):
    r = await gw.variation(n=2, seed=5)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["data"]) == 2 and body["size"] == "1024x1024"
    imgs = [decode(d) for d in body["data"]]
    assert all(i.size == (1024, 1024) for i in imgs)
    graph = graph_of(gw.fake)
    ks = nodes(graph, "KSampler")[0]
    assert ks["denoise"] == 0.55 and ks["seed"] == 5
    assert nodes(graph, "RepeatLatentBatch")[0]["amount"] == 2
    assert all(n["text"] == "" for n in nodes(graph, "CLIPTextEncode"))
    assert not nodes(graph, "LoadImageMask")
    assert input_files(gw.fake) == [] and gw.queue.active == 0


async def test_variation_random_seed_and_strength(gw):
    r1 = await gw.variation(strength=0.4)
    r2 = await gw.variation()
    assert r1.status_code == r2.status_code == 200
    s1, s2 = (nodes(graph_of(gw.fake, i), "KSampler")[0] for i in (0, 1))
    assert s1["denoise"] == 0.4 and s1["seed"] != s2["seed"]


async def test_variation_ignores_mask_field(gw):
    files = [
        ("image", ("a.png", make_png(), "image/png")),
        ("mask", ("m.png", make_mask(), "image/png")),
    ]
    r = await gw.client.post("/v1/images/variations", files=files, headers=AUTH)
    assert r.status_code == 200, r.text
    assert len(gw.fake.uploads) == 1


# ------------------------------------------------------------------ cleanup on failure paths


async def test_inputs_deleted_after_upstream_failure(gw):
    gw.fake.outcomes = ["error"]
    r = await gw.edit(mask=make_mask())
    assert r.status_code == 502
    assert len(gw.fake.uploads) == 2 and input_files(gw.fake) == []
    assert gw.queue.active == 0


async def test_upload_failure_502_never_submitted(gw):
    gw.fake.upload_mode = "500"
    r = await gw.edit()
    assert r.status_code == 502 and err(r)["code"] == "upstream_error"
    assert gw.fake.prompt_count == 0 and gw.queue.active == 0
    assert (await gw.generate()).status_code == 200  # worker not stuck


async def test_upload_renamed_502(gw):
    gw.fake.upload_mode = "rename"
    r = await gw.edit()
    assert r.status_code == 502
    assert gw.fake.prompt_count == 0 and gw.queue.active == 0
    assert input_files(gw.fake) == []  # the renamed file ComfyUI created is deleted too


async def test_comfy_down_during_upload_503(gw):
    entry = gw.app.state.catalog
    await entry.refresh()  # model cached as available
    gw.fake.down = True
    r = await gw.edit()
    assert r.status_code == 503 and gw.queue.active == 0


FAST = {"request_timeout_s": 1.0, "postprocess_reserve_s": 0.3, "est_job_overhead_s": 0.05}


async def test_inputs_deleted_after_timeout(gateway_factory):
    async with gateway_factory(**FAST) as gw:
        fast_ops(gw)
        gw.fake.hold = asyncio.Event()  # job stays running
        r = await asyncio.wait_for(gw.edit(mask=make_mask()), 5)
        assert r.status_code == 504
        pid = next(iter(gw.fake.prompts))
        assert gw.fake.interrupts == [pid] and pid in gw.fake.history_deletes
        assert len(gw.fake.uploads) == 2 and input_files(gw.fake) == []
        assert gw.queue.active == 0


async def test_timeout_during_upload_does_not_block_worker(gateway_factory):
    async with gateway_factory(**FAST) as gw:
        fast_ops(gw)
        gw.fake.upload_gate = asyncio.Event()  # upload hangs past the deadline
        r = await asyncio.wait_for(gw.edit(), 5)
        assert r.status_code == 504
        assert gw.fake.prompt_count == 0 and gw.queue.active == 0
        assert input_files(gw.fake) == []  # the cancelled upload never landed
        gw.fake.upload_gate.set()
        assert (await gw.generate()).status_code == 200


async def test_queued_edit_waits_for_its_uploads(gw):
    """The worker reaches the job before its upload finished: it waits, then submits."""
    fast_ops(gw)
    gw.fake.upload_gate = asyncio.Event()
    task = asyncio.create_task(gw.edit())
    await gw.wait_until(lambda: len(gw.fake.uploads) == 1)
    await gw.wait_until(
        lambda: gw.queue.running is not None and gw.queue.running.state == "awaiting_inputs"
    )
    assert gw.fake.prompt_count == 0  # not submitted before the input exists
    gw.fake.upload_gate.set()
    r = await asyncio.wait_for(task, 5)
    assert r.status_code == 200 and gw.fake.prompt_count == 1


def _scope(path: str, headers: list[tuple[bytes, bytes]]) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("gateway.test", 80),
        "state": {},
    }


async def test_inputs_deleted_after_disconnect(gw):
    fast_ops(gw)
    gw.fake.hold = asyncio.Event()
    req = httpx.Request(
        "POST",
        "http://gateway.test/v1/images/edits",
        data={"prompt": "a fox"},
        files={"image": ("a.png", make_png(), "image/png"), "mask": ("m.png", make_mask())},
    )
    payload = req.read()
    headers = [
        (b"host", b"gateway.test"),
        (b"authorization", f"Bearer {KEY}".encode()),
        (b"content-type", req.headers["content-type"].encode()),
        (b"content-length", str(len(payload)).encode()),
    ]
    sent: list[dict] = []
    first = True

    async def receive() -> dict:
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": payload, "more_body": False}
        await gw.fake.submitted.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    await asyncio.wait_for(gw.app(_scope("/v1/images/edits", headers), receive, send), 5)
    pid = next(iter(gw.fake.prompts))
    assert gw.fake.interrupts == [pid] and pid in gw.fake.history_deletes
    assert len(gw.fake.uploads) == 2 and input_files(gw.fake) == []
    assert gw.queue.active == 0
    assert not any(m.get("status") == 200 for m in sent)


async def test_edit_admission_429(gw):
    for _ in range(3):
        gw.fake.add_external()  # 3 * EST_EXTERNAL_JOB_S (30 s) > the 80 s budget
    await gw.wait_until(lambda: gw.queue.comfy_external_pending == 3)
    r = await gw.edit()
    assert r.status_code == 429 and err(r)["code"] == "estimated_wait_exceeded"
    assert "retry-after" in r.headers and gw.fake.uploads == []
    assert gw.queue.active == 0


# ------------------------------------------------------------------ preprocessing DoS guards


async def _garbage_multipart(gw) -> httpx.Response:
    headers = {**AUTH, "content-type": "multipart/form-data; boundary=x"}
    return await gw.client.post("/v1/images/edits", content=b"not multipart", headers=headers)


async def test_capacity_precheck_before_body_is_parsed(gateway_factory):
    async with gateway_factory(queue_max_depth=0) as gw:
        gw.fake.add_external()  # active + external >= depth + 1 (same threshold as admit)
        await gw.wait_until(lambda: gw.queue.comfy_external_pending == 1)
        r = await _garbage_multipart(gw)  # would be 400 if the body were parsed
        assert r.status_code == 429 and err(r)["code"] == "queue_full"
        assert "retry-after" in r.headers


async def test_wait_precheck_before_body_is_parsed(gw):
    for _ in range(3):
        gw.fake.add_external()  # current estimated wait (90 s) > the 80 s budget
    await gw.wait_until(lambda: gw.queue.comfy_external_pending == 3)
    r = await _garbage_multipart(gw)
    assert r.status_code == 429 and err(r)["code"] == "estimated_wait_exceeded"
    assert int(r.headers["retry-after"]) >= 90


async def test_preprocess_semaphore_fails_fast(gateway_factory, monkeypatch):
    from app.routes import images

    entered, release = threading.Event(), threading.Event()
    real = images.prepare_inputs

    def slow(*args, **kwargs):
        entered.set()
        release.wait(5)
        return real(*args, **kwargs)

    monkeypatch.setattr(images, "prepare_inputs", slow)
    async with gateway_factory(max_concurrent_preprocess=1) as gw:
        first = asyncio.create_task(gw.edit())
        await gw.wait_until(entered.is_set)
        r = await gw.edit()
        assert r.status_code == 429 and err(r)["code"] == "queue_full"
        assert "preprocessing" in err(r)["message"] and r.headers["retry-after"] == "1"
        release.set()
        assert (await asyncio.wait_for(first, 5)).status_code == 200
        assert gw.queue.active == 0
        assert (await gw.edit()).status_code == 200  # slot returned


def _exif_png(img: Image.Image, orientation: int) -> bytes:
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif)
    return buf.getvalue()


async def test_mask_exif_orientation_applied(gw):
    # 32x64 mask, top half transparent; orientation 6 (rotate 90 CW) -> right half.
    raw = Image.new("RGBA", (32, 64), (0, 0, 0, 255))
    raw.paste((0, 0, 0, 0), (0, 0, 32, 32))
    r = await gw.edit(image=make_png(64, 32), mask=_exif_png(raw, 6))
    assert r.status_code == 200, r.text
    alpha = uploaded(gw.fake, 1).getchannel("A")
    w, h = alpha.size
    assert alpha.getpixel((w - 5, h // 2)) == 0  # right: area to edit
    assert alpha.getpixel((5, h // 2)) == 255
    assert alpha.getpixel((w // 4, 5)) == alpha.getpixel((w // 4, h - 5)) == 255


@pytest.mark.parametrize(("mode", "param"), [("RGB", "image"), ("L", "mask")])
async def test_processing_failure_is_400(gw, monkeypatch, mode, param):
    real_fit = ImageOps.fit

    def fit(img, *args, **kwargs):
        if img.mode == mode:
            raise ValueError("boom")
        return real_fit(img, *args, **kwargs)

    monkeypatch.setattr(ImageOps, "fit", fit)
    r = await gw.edit(mask=make_mask())
    assert r.status_code == 400
    assert err(r)["param"] == param and err(r)["code"] == "invalid_image"
    assert gw.fake.uploads == [] and gw.queue.active == 0


def test_exif_transpose_failure_is_400(monkeypatch):
    def broken(img, *args, **kwargs):
        raise OSError("bad exif")

    monkeypatch.setattr(ImageOps, "exif_transpose", broken)
    with pytest.raises(Exception) as info:
        uploads.prepare_inputs(make_png(), None, 64, 64, 10_000)
    assert info.value.status == 400 and info.value.param == "image"


def test_jpeg_draft_decode():
    data = encode(Image.new("RGB", (1000, 800), (10, 20, 30)), "JPEG")
    img = uploads._open(data, "image", 10**7, uploads.ALLOWED_FORMATS, draft=(100, 100))
    assert img.size == (125, 100)  # DCT scale 1/8, never below the requested size
    png = uploads._open(make_png(64, 64), "image", 10**7, uploads.ALLOWED_FORMATS, (8, 8))
    assert png.size == (64, 64)
    image_png, _ = uploads.prepare_inputs(data, None, 64, 32, 10**7)
    assert Image.open(io.BytesIO(image_png)).size == (64, 32)


def test_configure_pillow_sets_global_limit():
    old = Image.MAX_IMAGE_PIXELS
    try:
        uploads.configure_pillow(1234)
        assert Image.MAX_IMAGE_PIXELS == 1234
    finally:
        Image.MAX_IMAGE_PIXELS = old


def test_pillow_limit_never_below_largest_output(tmp_path, fake):
    from app.main import create_app
    from tests.helpers import make_settings

    create_app(make_settings(tmp_path, max_input_pixels=1000), comfy_transport=fake.transport)
    assert Image.MAX_IMAGE_PIXELS == 1024 * 1024  # largest registered output size
    create_app(make_settings(tmp_path), comfy_transport=fake.transport)
    assert Image.MAX_IMAGE_PIXELS == 16_777_216


def test_starlette_multipart_internals_exist():
    """uploads._LimitedMultiPartParser hooks private starlette internals: fail loudly."""

    async def stream():
        yield b""

    parser = MultiPartParser(Headers({"content-type": "multipart/form-data; boundary=x"}), stream())
    assert isinstance(parser.spool_max_size, int)
    assert callable(MultiPartParser.on_part_begin) and callable(MultiPartParser.on_part_data)
    parser.on_part_begin()
    part = parser._current_part
    assert hasattr(part, "file") and hasattr(part, "field_name")


# ------------------------------------------------------------------ slow uploads / cleanup


async def test_slow_upload_holds_worker_at_most_upload_wait(gateway_factory):
    async with gateway_factory(upload_wait_s=0.3) as gw:
        fast_ops(gw)
        gw.fake.upload_gate = asyncio.Event()  # the edit's upload never finishes
        t0 = time.monotonic()
        edit = asyncio.create_task(gw.edit())
        await gw.wait_until(lambda: len(gw.fake.uploads) == 1)
        r = await asyncio.wait_for(gw.generate(), 5)  # queued behind the stuck edit
        assert r.status_code == 200
        r = await asyncio.wait_for(edit, 5)
        assert r.status_code == 503 and err(r)["code"] == "upstream_unavailable"
        assert time.monotonic() - t0 < 3  # nowhere near the 80 s generation budget
        assert gw.fake.prompt_count == 1 and gw.queue.active == 0
        gw.fake.upload_gate.set()


async def test_inputs_deleted_again_after_late_landing(gw):
    gw.app.state.input_retry_delay_s = 0.05
    r = await gw.edit()
    assert r.status_code == 200 and input_files(gw.fake) == []
    late = gw.fake.input_dir / gw.fake.uploads[0]["filename"]
    late.write_bytes(b"landed after the first delete")
    await gw.wait_until(lambda: not late.exists())
    await gw.wait_until(lambda: not gw.app.state.background_tasks)


async def test_background_retry_cancelled_on_shutdown(gateway_factory):
    async with gateway_factory() as gw:
        gw.app.state.input_retry_delay_s = 60
        assert (await gw.edit()).status_code == 200
        tasks = set(gw.app.state.background_tasks)
        assert len(tasks) == 1
    assert all(t.done() for t in tasks)


# ------------------------------------------------------------------ body limit path matching


@pytest.mark.parametrize(
    ("path", "root_path"),
    [
        ("/v1/images/edits/", ""),
        ("/api/v1/images/edits", "/api"),
        ("/api/v1/images/edits/", "/api/"),
    ],
)
async def test_multipart_limit_path_normalized(gateway_factory, path, root_path):
    from app.main import BodySizeLimitMiddleware

    seen: list[int] = []

    async def app(scope, receive, send):
        while True:
            message = await receive()
            seen.append(len(message.get("body", b"")))
            if not message.get("more_body"):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    mw = BodySizeLimitMiddleware(app, 10, {"/v1/images/edits": 1000})
    scope = _scope(path, [(b"content-length", b"500")])
    scope["root_path"] = root_path
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": b"x" * 500, "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await mw(scope, receive, send)
    assert sent[0]["status"] == 204 and seen == [500]
    scope = _scope("/v1/images/generations/", [(b"content-length", b"500")])
    sent.clear()
    await mw(scope, receive, send)
    assert sent[0]["status"] == 413
