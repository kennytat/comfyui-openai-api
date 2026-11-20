"""Hosted-image store (TTL, expiry on read) and ComfyUI output cleanup."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("gateway.storage")

ID_RE = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_EXTS = ("png", "jpeg", "webp")

Clock = Callable[[], float]


@dataclass(frozen=True)
class StoredImage:
    path: Path
    ext: str
    remaining_s: float


class ImageStore:
    def __init__(self, directory: str | Path, ttl_s: float, clock: Clock = time.time) -> None:
        self.dir = Path(directory)
        self.ttl_s = float(ttl_s)
        self.clock = clock

    def put(self, data: bytes, ext: str) -> str:
        if ext not in ALLOWED_EXTS:
            raise ValueError(f"unsupported extension: {ext}")
        self.dir.mkdir(parents=True, exist_ok=True)
        image_id = uuid.uuid4().hex
        path = self.dir / f"{image_id}.{ext}"
        tmp = self.dir / f".{image_id}.tmp"
        tmp.write_bytes(data)
        now = self.clock()
        os.utime(tmp, (now, now))  # mtime follows the injected clock
        os.replace(tmp, path)
        return image_id

    def delete(self, image_id: str, ext: str) -> None:
        if ID_RE.match(image_id) and ext in ALLOWED_EXTS:
            (self.dir / f"{image_id}.{ext}").unlink(missing_ok=True)

    def get(self, image_id: str, ext: str, now: float | None = None) -> StoredImage | None:
        if not ID_RE.match(image_id) or ext not in ALLOWED_EXTS:
            return None
        path = self.dir / f"{image_id}.{ext}"
        try:
            st = path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(st.st_mode):
            return None
        now = self.clock() if now is None else now
        remaining = st.st_mtime + self.ttl_s - now
        if remaining <= 0:
            path.unlink(missing_ok=True)  # lazy delete
            return None
        return StoredImage(path=path, ext=ext, remaining_s=remaining)

    def sweep_expired(self, now: float | None = None) -> int:
        now = self.clock() if now is None else now
        return _sweep_old_files(self.dir, self.ttl_s, now)


def _sweep_old_files(root: Path, max_age_s: float, now: float) -> int:
    """Delete regular files under root older than max_age_s. Never follows symlinks."""
    removed = 0
    if not root.is_dir():
        return 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
                if not stat.S_ISREG(st.st_mode) or st.st_mtime + max_age_s > now:
                    continue
                p.unlink()
                removed += 1
            except OSError:
                log.warning("sweep could not remove %s", p, exc_info=True)
    return removed


def delete_comfy_outputs(
    images: Iterable[str | tuple[str, str]],
    output_dir: str | Path,
    subfolder: str = "openai-api",
) -> list[str]:
    """Delete ComfyUI output files. Items are "subfolder/file" strings or (subfolder, file).

    The subfolder prefix is stripped, the remainder resolved under output_dir; anything that
    escapes it, is a symlink, or is not a regular file is refused. Returns deleted names.
    """
    root = Path(output_dir).resolve()
    prefix = subfolder.strip("/")
    deleted: list[str] = []
    for item in images:
        rel = f"{item[0]}/{item[1]}" if isinstance(item, tuple) else item
        rel = rel.replace("\\", "/").lstrip("/")
        if prefix and rel.startswith(prefix + "/"):
            rel = rel[len(prefix) + 1 :]
        if not rel or ".." in Path(rel).parts:
            log.warning("refusing to delete suspicious output path %r", item)
            continue
        candidate = root / rel
        try:
            parent = candidate.parent.resolve()
            if parent != root and root not in parent.parents:
                log.warning("refusing to delete output path outside mount: %r", item)
                continue
            target = parent / candidate.name
            st = target.lstat()
            if not stat.S_ISREG(st.st_mode):
                continue
            target.unlink()
            deleted.append(rel)
        except FileNotFoundError:
            continue
        except OSError:
            log.warning("could not delete output %r", item, exc_info=True)
    return deleted


def sweep_orphans(output_dir: str | Path, max_age_s: float, now: float | None = None) -> int:
    return _sweep_old_files(Path(output_dir), max_age_s, time.time() if now is None else now)


async def run_ttl_sweeper(store: ImageStore, interval_s: float = 60.0) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await asyncio.to_thread(store.sweep_expired)
        except Exception:
            log.exception("ttl sweeper failed")


async def run_orphan_sweeper(
    output_dir: str | Path,
    max_age_s: float,
    interval_s: float = 600.0,
    clock: Clock = time.time,
) -> None:
    while True:
        await asyncio.sleep(interval_s)
        try:
            await asyncio.to_thread(sweep_orphans, output_dir, max_age_s, clock())
        except Exception:
            log.exception("orphan sweeper failed")
