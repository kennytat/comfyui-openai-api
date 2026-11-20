import asyncio
import os

import pytest

from app.storage import (
    ImageStore,
    delete_comfy_outputs,
    run_orphan_sweeper,
    run_ttl_sweeper,
    sweep_orphans,
)


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return ImageStore(tmp_path / "images", ttl_s=100, clock=clock)


def test_put_get(store):
    i = store.put(b"data", "png")
    assert len(i) == 32
    got = store.get(i, "png")
    assert got.path.read_bytes() == b"data" and got.remaining_s == 100


def test_get_validation(store):
    i = store.put(b"d", "png")
    assert store.get(i, "gif") is None
    assert store.get("../etc/passwd", "png") is None
    assert store.get(i.upper(), "png") is None
    assert store.get("0" * 32, "png") is None


def test_expiry_on_read_before_sweeper(store, clock):
    i = store.put(b"d", "png")
    clock.t += 99
    assert store.get(i, "png").remaining_s == pytest.approx(1)
    clock.t += 1
    assert store.get(i, "png") is None
    assert not (store.dir / f"{i}.png").exists()  # lazily deleted


def test_sweep_expired(store, clock):
    a = store.put(b"a", "png")
    clock.t += 60
    b = store.put(b"b", "webp")
    clock.t += 50
    assert store.sweep_expired() == 1
    assert store.get(a, "png") is None and store.get(b, "webp") is not None


def test_delete_comfy_outputs(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    (root / "x_00001_.png").write_bytes(b"1")
    (root / "sub").mkdir()
    (root / "sub" / "y.png").write_bytes(b"1")
    got = delete_comfy_outputs(["openai-api/x_00001_.png", ("openai-api", "sub/y.png")], root)
    assert sorted(got) == ["sub/y.png", "x_00001_.png"]
    assert not any(p.is_file() for p in root.rglob("*"))


@pytest.mark.parametrize(
    "bad",
    [
        "openai-api/../secret.txt",
        "../secret.txt",
        "/etc/passwd",
        "openai-api/../../secret.txt",
        "openai-api/sub/../../secret.txt",
    ],
)
def test_traversal_refused(tmp_path, bad):
    root = tmp_path / "out"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    assert delete_comfy_outputs([bad], root) == []
    assert secret.exists()


def test_symlink_not_followed_on_delete(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("s")
    (root / "link.png").symlink_to(secret)
    (root / "dirlink").symlink_to(tmp_path)
    assert (
        delete_comfy_outputs(["openai-api/link.png", "openai-api/dirlink/secret.txt"], root) == []
    )
    assert secret.exists() and (root / "link.png").is_symlink()


def test_orphan_sweeper_keeps_fresh_removes_old(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    old, fresh = root / "old.png", root / "fresh.png"
    old.write_bytes(b"1")
    fresh.write_bytes(b"1")
    now = 1_000_000.0
    os.utime(old, (now - 7200, now - 7200))
    os.utime(fresh, (now - 10, now - 10))
    assert sweep_orphans(root, 3600, now) == 1
    assert not old.exists() and fresh.exists()


def test_orphan_sweeper_never_follows_symlinks(tmp_path):
    root = tmp_path / "out"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.png"
    victim.write_bytes(b"1")
    os.utime(victim, (0, 0))
    (root / "dirlink").symlink_to(outside)
    (root / "filelink.png").symlink_to(victim)
    assert sweep_orphans(root, 1, 1_000_000.0) == 0
    assert victim.exists()


async def test_sweeper_loops(tmp_path, store, clock):
    i = store.put(b"d", "png")
    clock.t += 1000
    root = tmp_path / "out"
    root.mkdir()
    old = root / "o.png"
    old.write_bytes(b"1")
    os.utime(old, (0, 0))
    tasks = [
        asyncio.create_task(run_ttl_sweeper(store, 0.01)),
        asyncio.create_task(run_orphan_sweeper(root, 10, 0.01, clock=clock)),
    ]
    await asyncio.sleep(0.2)
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert not (store.dir / f"{i}.png").exists()
    assert not old.exists()
