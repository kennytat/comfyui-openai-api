#!/usr/bin/env python3
"""Real-GPU smoke test with the official OpenAI SDK (plan Verification 5; not run in CI).

    GATEWAY_BASE_URL=http://localhost:8001/v1 GATEWAY_API_KEY=... \
        python tests/smoke/smoke_openai_sdk.py

Checks: b64 and url generations decode with Pillow, dimensions match (incl. a size alias),
no PNG text metadata, n=2 yields 2 distinct images, a bad key raises AuthenticationError.
Runs one job after ``POST /free`` on ComfyUI when ``COMFYUI_URL`` is set (cold checkpoint)
and a warm one, then prints client-side wall times plus the gateway's per-job log lines
(``docker compose logs gateway``; disable with ``SMOKE_NO_LOGS=1``) for calibrating
``est_seconds_per_step``, ``cold_load_s`` and ``EST_EXTERNAL_JOB_S``.
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime

import openai
from PIL import Image

BASE = os.environ.get("GATEWAY_BASE_URL", "http://localhost:8001/v1")
KEY = os.environ.get("GATEWAY_API_KEY", "")
MODEL = os.environ.get("SMOKE_MODEL", "sdxl")
COMFYUI_URL = os.environ.get("COMFYUI_URL")  # e.g. http://localhost:8188, optional


def check(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def open_png(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def no_text_metadata(img: Image.Image) -> bool:
    return not getattr(img, "text", {}) and not img.info.get("exif")


def timed(client: openai.OpenAI, label: str, **kw):
    t0 = time.monotonic()
    resp = client.images.generate(model=MODEL, **kw)
    print(f"     {label}: {time.monotonic() - t0:.1f}s wall")
    return resp


def free_comfy_models() -> None:
    if not COMFYUI_URL:
        print("     (COMFYUI_URL unset: skipping POST /free, first job may already be warm)")
        return
    req = urllib.request.Request(
        f"{COMFYUI_URL.rstrip('/')}/free",
        data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10).read()


def print_job_logs(since: str) -> None:
    if os.environ.get("SMOKE_NO_LOGS"):
        return
    try:
        out = subprocess.run(
            ["docker", "compose", "logs", "--no-color", "--since", since, "gateway"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"     (could not read gateway logs: {exc})")
        return
    print("\nper-job log lines (estimate vs actual):")
    keys = ("outcome", "n", "size", "steps", "est_job_s", "est_wait_s", "cold_load_applied")
    keys += ("queue_wait_ms", "comfy_wait_ms", "gpu_ms", "postprocess_ms", "total_ms")
    for line in out.splitlines():
        if "gateway.jobs" not in line or "{" not in line:
            continue
        try:
            rec = json.loads(line[line.index("{") :])
        except ValueError:
            continue
        print("  " + "  ".join(f"{k}={rec.get(k)}" for k in keys))


def main() -> None:
    if not KEY:
        sys.exit("set GATEWAY_API_KEY")
    since = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    client = openai.OpenAI(base_url=BASE, api_key=KEY, max_retries=0, timeout=120)

    models = [m.id for m in client.models.list().data]
    check(MODEL in models, f"models.list contains {MODEL}: {models}")

    try:
        openai.OpenAI(base_url=BASE, api_key="wrong", max_retries=0).models.list()
        check(False, "bad key rejected")
    except openai.AuthenticationError:
        check(True, "bad key -> AuthenticationError")

    free_comfy_models()
    cold = timed(client, "cold b64 1024x1024", prompt="a red fox in the snow", quality="low")
    img = open_png(base64.b64decode(cold.data[0].b64_json))
    check(img.size == (1024, 1024), f"b64 image is 1024x1024 ({img.size})")
    check(img.format == "PNG" and no_text_metadata(img), "b64 PNG has no text metadata")

    warm = timed(
        client,
        "warm b64 n=2",
        prompt="a lighthouse at dusk",
        quality="low",
        n=2,
        response_format="b64_json",
    )
    check(len(warm.data) == 2, "n=2 returns 2 images")
    check(warm.data[0].b64_json != warm.data[1].b64_json, "n=2 images are distinct")

    url_resp = timed(
        client,
        "warm url 1024x1792",
        prompt="a tall glass tower",
        quality="low",
        size="1024x1792",
        response_format="url",
    )
    check(url_resp.size == "768x1344", f"aliased size reported as 768x1344 ({url_resp.size})")
    url = url_resp.data[0].url
    check(bool(url) and "/images/" in url, f"url returned: {url}")
    with urllib.request.urlopen(url, timeout=30) as r:  # no auth on hosted images
        ctype = r.headers.get("Content-Type")
        data = r.read()
    check(ctype == "image/png", f"hosted image content-type {ctype}")
    img = open_png(data)
    check(img.size == (768, 1344), f"hosted image is 768x1344 ({img.size})")
    check(no_text_metadata(img), "hosted PNG has no text metadata")

    print_job_logs(since)
    print("\nsmoke test passed")


if __name__ == "__main__":
    main()
