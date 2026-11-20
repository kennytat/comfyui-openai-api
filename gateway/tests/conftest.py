"""Pytest fixtures; the fake ComfyUI and other shared helpers live in ``tests.helpers``."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager

from app.main import create_app
from tests.helpers import FakeClock, FakeComfy, Gateway, make_png, make_settings


@pytest.fixture
def png_fixture() -> bytes:
    return make_png()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def fake(tmp_path: Path) -> FakeComfy:
    out = tmp_path / "comfy-output"
    out.mkdir()
    return FakeComfy(out)


@pytest.fixture
def gateway_factory(tmp_path: Path, fake: FakeComfy, clock: FakeClock):
    @contextlib.asynccontextmanager
    async def factory(
        raise_app_exceptions: bool = True, **overrides: Any
    ) -> AsyncIterator[Gateway]:
        settings = make_settings(tmp_path, **overrides)
        app = create_app(settings, comfy_transport=fake.transport, clock=clock)
        app.state.disconnect_poll_s = 0.02
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
            async with httpx.AsyncClient(transport=transport, base_url="http://gateway.test") as c:
                yield Gateway(app, c, fake)

    return factory


@pytest.fixture
async def gw(gateway_factory) -> AsyncIterator[Gateway]:
    async with gateway_factory() as g:
        yield g
