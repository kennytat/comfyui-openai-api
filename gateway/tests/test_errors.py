import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from app import errors as E

CASES = [
    (lambda: E.invalid_value("bad", "n"), 400, "invalid_request_error", "invalid_value", "n", {}),
    (
        lambda: E.unsupported_value("no", "stream"),
        400,
        "invalid_request_error",
        "unsupported_value",
        "stream",
        {},
    ),
    (
        lambda: E.request_too_expensive("big", "steps"),
        400,
        "invalid_request_error",
        "request_too_expensive",
        "steps",
        {},
    ),
    (
        E.invalid_api_key,
        401,
        "invalid_request_error",
        "invalid_api_key",
        None,
        {"www-authenticate": "Bearer"},
    ),
    (lambda: E.model_not_found("x"), 404, "invalid_request_error", "model_not_found", "model", {}),
    (E.not_found, 404, "invalid_request_error", "not_found", None, {}),
    (E.method_not_allowed, 405, "invalid_request_error", "method_not_allowed", None, {}),
    (lambda: E.request_too_large(10), 413, "invalid_request_error", "request_too_large", None, {}),
    (lambda: E.queue_full(4.2), 429, "rate_limit_error", "queue_full", None, {"retry-after": "5"}),
    (
        lambda: E.estimated_wait_exceeded(31),
        429,
        "rate_limit_error",
        "estimated_wait_exceeded",
        None,
        {"retry-after": "31"},
    ),
    (
        lambda: E.upstream_error("x", retryable=False),
        502,
        "server_error",
        "upstream_error",
        None,
        {"x-should-retry": "false"},
    ),
    (E.upstream_error, 502, "server_error", "upstream_error", None, {}),
    (E.upstream_oom, 502, "server_error", "upstream_oom", None, {"x-should-retry": "false"}),
    (
        E.upstream_interrupted,
        502,
        "server_error",
        "upstream_interrupted",
        None,
        {"x-should-retry": "false"},
    ),
    (E.upstream_output_missing, 502, "server_error", "upstream_output_missing", None, {}),
    (E.upstream_unavailable, 503, "server_error", "upstream_unavailable", None, {}),
    (E.timeout, 504, "server_error", "timeout", None, {"x-should-retry": "false"}),
    (E.internal_error, 500, "server_error", "internal_error", None, {}),
]


class Body(BaseModel):
    n: int = Field(ge=1)


def make_app():
    app = FastAPI()
    E.install_error_handlers(app)
    for i, case in enumerate(CASES):
        factory = case[0]

        def route(factory=factory):
            raise factory()

        app.add_api_route(f"/err/{i}", route, methods=["GET"])

    @app.post("/body")
    def body(b: Body):
        return {}

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret stack detail")

    return app


@pytest.fixture(scope="module")
def client():
    return TestClient(make_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("i", range(len(CASES)))
def test_each_error(client, i):
    _, status, etype, code, param, headers = CASES[i]
    r = client.get(f"/err/{i}")
    assert r.status_code == status
    err = r.json()["error"]
    assert err["type"] == etype and err["code"] == code and err["param"] == param
    assert err["message"]
    for k, v in headers.items():
        assert r.headers[k] == v
    if "x-should-retry" not in headers:
        assert "x-should-retry" not in r.headers


def test_validation_param_from_loc(client):
    r = client.post("/body", json={"n": 0})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "n" and err["type"] == "invalid_request_error"


def test_missing_param(client):
    r = client.post("/body", json={})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "missing_required_parameter"
    assert r.json()["error"]["param"] == "n"


def test_malformed_json(client):
    r = client.post("/body", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_json"


def test_404_and_405(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    r = client.get("/body")
    assert r.status_code == 405 and r.json()["error"]["code"] == "method_not_allowed"
    assert "allow" in r.headers


def test_catch_all_no_leak(client):
    r = client.get("/boom")
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "internal_error"
    assert "secret" not in r.text
