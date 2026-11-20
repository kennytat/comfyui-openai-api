"""Every row of the plan's 3.3 error table, produced through the real app (Step 1.4/1.5).

Rows covered elsewhere through the real app: 429 queue_full / estimated_wait_exceeded and
400 request_too_expensive (test_queue.py), 504 timeout (test_queue.py, SDK compat).
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.helpers import AUTH, KEY


def err(r: httpx.Response) -> dict:
    body = r.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"message", "type", "param", "code"}
    return body["error"]


async def test_malformed_json(gw):
    r = await gw.client.post(
        "/v1/images/generations",
        content=b"{not json",
        headers={**AUTH, "content-type": "application/json"},
    )
    assert r.status_code == 400
    e = err(r)
    assert e["type"] == "invalid_request_error" and e["code"] == "invalid_json"


async def test_non_object_json(gw):
    r = await gw.client.post("/v1/images/generations", json=[1, 2], headers=AUTH)
    assert r.status_code == 400 and err(r)["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    "body,param,code",
    [
        ({}, "prompt", "missing_required_parameter"),
        ({"prompt": ""}, "prompt", "invalid_value"),
        ({"prompt": "x", "n": "two"}, "n", "invalid_value"),
        ({"prompt": "x", "n": 0}, "n", "invalid_value"),
        ({"prompt": "x", "response_format": "gif"}, "response_format", "invalid_value"),
        ({"prompt": "x", "output_format": "bmp"}, "output_format", "invalid_value"),
        ({"prompt": "x", "quality": "ultra"}, "quality", "invalid_value"),
        ({"prompt": "x", "steps": 81}, "steps", "invalid_value"),
        ({"prompt": "x" * 4001}, "prompt", "invalid_value"),
        ({"prompt": "x", "negative_prompt": "y" * 4001}, "negative_prompt", "invalid_value"),
    ],
)
async def test_schema_violations(gw, body, param, code):
    r = await gw.client.post("/v1/images/generations", json=body, headers=AUTH)
    assert r.status_code == 400
    e = err(r)
    assert (e["param"], e["code"], e["type"]) == (param, code, "invalid_request_error")
    assert gw.fake.prompt_count == 0


@pytest.mark.parametrize(
    "body,param",
    [
        ({"stream": True}, "stream"),
        ({"partial_images": 1}, "partial_images"),
        ({"background": "transparent"}, "background"),
    ],
)
async def test_unsupported(gw, body, param):
    r = await gw.generate(**body)
    assert r.status_code == 400
    assert (err(r)["code"], err(r)["param"]) == ("unsupported_value", param)


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer nope"}, {"Authorization": f"Basic {KEY}"}],
)
async def test_auth(gw, headers):
    for method, path in (("POST", "/v1/images/generations"), ("GET", "/v1/models")):
        r = await gw.client.request(method, path, json={"prompt": "x"}, headers=headers)
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        e = err(r)
        assert (e["type"], e["code"]) == ("invalid_request_error", "invalid_api_key")


async def test_model_not_found(gw):
    r = await gw.generate(model="dall-e-3")
    assert r.status_code == 404
    assert (err(r)["code"], err(r)["param"]) == ("model_not_found", "model")


async def test_route_not_found_and_method(gw):
    r = await gw.client.get("/v1/nope", headers=AUTH)
    assert r.status_code == 404 and err(r)["code"] == "not_found"
    r = await gw.client.get("/v1/images/generations", headers=AUTH)
    assert r.status_code == 405 and err(r)["code"] == "method_not_allowed"


async def test_body_too_large_content_length(gateway_factory):
    async with gateway_factory(max_body_bytes=1000) as gw:
        r = await gw.generate(prompt="x" * 2000)
        assert r.status_code == 413
        assert err(r)["code"] == "request_too_large"


async def test_body_too_large_streamed(gateway_factory):
    async with gateway_factory(max_body_bytes=1000) as gw:

        async def chunks():
            yield b'{"prompt": "'
            for _ in range(10):
                yield b"x" * 200
            yield b'"}'

        r = await gw.client.post(
            "/v1/images/generations",
            content=chunks(),
            headers={**AUTH, "content-type": "application/json"},
        )
        assert "content-length" not in {k.lower() for k in r.request.headers}
        assert r.status_code == 413
        assert err(r)["code"] == "request_too_large"
        assert gw.fake.prompt_count == 0


async def test_body_streamed_small_ok(gateway_factory):
    async with gateway_factory(max_body_bytes=1000) as gw:

        async def chunks():
            yield b'{"prompt": '
            yield b'"a fox"}'

        r = await gw.client.post(
            "/v1/images/generations",
            content=chunks(),
            headers={**AUTH, "content-type": "application/json"},
        )
        assert r.status_code == 200


async def test_node_errors_502_no_retry(gw):
    gw.fake.node_errors = True
    r = await gw.generate()
    assert r.status_code == 502
    assert (err(r)["type"], err(r)["code"]) == ("server_error", "upstream_error")
    assert r.headers["x-should-retry"] == "false"
    assert gw.queue.active == 0


async def test_execution_error_502(gw):
    gw.fake.outcomes.append("error")
    r = await gw.generate()
    assert r.status_code == 502 and err(r)["code"] == "upstream_error"
    assert "x-should-retry" not in r.headers
    pid = next(iter(gw.fake.prompts))
    assert pid in gw.fake.history_deletes and gw.fake.interrupts == []


async def test_oom_502_no_retry(gw):
    gw.fake.outcomes.append("oom")
    r = await gw.generate()
    assert r.status_code == 502 and err(r)["code"] == "upstream_oom"
    assert r.headers["x-should-retry"] == "false"


async def test_interrupted_externally_502(gw):
    gw.fake.outcomes.append("interrupted")
    r = await gw.generate()
    assert r.status_code == 502 and err(r)["code"] == "upstream_interrupted"
    assert r.headers["x-should-retry"] == "false"


async def test_lost_prompt_502_fast(gw):
    gw.fake.outcomes.append("lost")
    r = await asyncio.wait_for(gw.generate(), 2)
    assert r.status_code == 502 and err(r)["code"] == "upstream_error"


async def test_view_404_502(gw):
    gw.fake.view_mode = "404"
    r = await gw.generate()
    assert r.status_code == 502 and err(r)["code"] == "upstream_output_missing"
    assert list(gw.fake.output_dir.iterdir()) == []  # outputs still deleted


async def test_view_drop_503(gw):
    gw.fake.view_mode = "drop"
    r = await gw.generate()
    assert r.status_code == 503
    assert (err(r)["type"], err(r)["code"]) == ("server_error", "upstream_unavailable")


async def test_comfy_down_503(gw):
    await gw.client.get("/v1/models", headers=AUTH)  # model cache warm
    gw.fake.down = True
    r = await gw.generate()
    assert r.status_code == 503 and err(r)["code"] == "upstream_unavailable"
    assert gw.queue.active == 0


async def test_unexpected_exception_500(gateway_factory, monkeypatch):
    import app.routes.images as images

    def boom(*a, **kw):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(images, "build_prompt", boom)
    async with gateway_factory(raise_app_exceptions=False) as gw:
        r = await gw.generate()
        assert r.status_code == 500
        e = err(r)
        assert (e["type"], e["code"]) == ("server_error", "internal_error")
        assert "secret" not in r.text and "Traceback" not in r.text


async def test_unexpected_exception_after_admission_releases(gateway_factory, monkeypatch):
    import app.routes.images as images

    def boom(*a, **kw):
        raise RuntimeError("encode crashed")

    monkeypatch.setattr(images, "reencode", boom)
    async with gateway_factory(raise_app_exceptions=False) as gw:
        r = await gw.generate()
        assert r.status_code == 500 and err(r)["code"] == "internal_error"
        assert gw.queue.active == 0
        assert len(gw.fake.history_deletes) == 1
