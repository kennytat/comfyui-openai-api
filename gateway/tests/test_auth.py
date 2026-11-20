from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.auth import require_api_key
from app.errors import install_error_handlers


@pytest.fixture
def client():
    app = FastAPI()
    install_error_handlers(app)
    app.state.settings = SimpleNamespace(api_keys=["key-one", "key-two"])

    @app.get("/v1/x", dependencies=[Depends(require_api_key)])
    def x():
        return {"ok": True}

    return TestClient(app)


def test_valid_keys(client):
    for k in ("key-one", "key-two"):
        assert client.get("/v1/x", headers={"Authorization": f"Bearer {k}"}).status_code == 200


def test_case_insensitive_scheme(client):
    assert client.get("/v1/x", headers={"Authorization": "bearer key-one"}).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic key-one"},
        {"Authorization": "Bearer"},
        {"Authorization": "key-one"},
    ],
)
def test_rejected(client, headers):
    r = client.get("/v1/x", headers=headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert r.json()["error"]["code"] == "invalid_api_key"
