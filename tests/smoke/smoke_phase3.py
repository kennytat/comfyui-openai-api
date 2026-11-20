#!/usr/bin/env python3
"""Phase 3 GPU smoke test for one model: generate (cold + warm), a size alias, n=MAX_N,
hd quality, edit, masked edit and variation through the official ``openai`` SDK.

Env: ``GATEWAY_BASE_URL`` (e.g. ``http://127.0.0.1:8001/v1``), ``GATEWAY_API_KEY``,
optional ``MAX_N`` (default 4).
Usage: ``python tests/smoke/smoke_phase3.py MODEL ALIAS_SIZE OUT_DIR``

Run it right after a job on a *different* checkpoint so the first request measures the
cold load. Writes the decoded outputs plus ``OUT_DIR/generations.json`` (request params,
wall seconds, response size, per-file pixels and PNG text metadata, which must be empty).
A request rejected up front with 400 ``request_too_expensive`` is recorded as such (it is
an accepted outcome for ``n=MAX_N``); any other error fails the run. Exit code 0 when
every returned image decodes with the reported size and carries no text metadata.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import openai
from PIL import Image


def make_mask(size: tuple[int, int]) -> bytes:
    """Opaque PNG with a transparent region (lower center) = the area to repaint."""
    w, h = size
    mask = Image.new("RGBA", size, (0, 0, 0, 255))
    mask.paste((0, 0, 0, 0), (int(w * 0.3), int(h * 0.55), int(w * 0.7), int(h * 0.95)))
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    model, alias, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    max_n = int(os.environ.get("MAX_N", "4"))
    out_dir.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(
        base_url=os.environ["GATEWAY_BASE_URL"],
        api_key=os.environ["GATEWAY_API_KEY"],
        max_retries=0,
        timeout=120,
    )
    fox = "a red fox sitting in fresh snow, golden hour, detailed fur"
    gen_runs = [
        ("generate_cold", {"prompt": fox}),
        ("generate_warm", {"prompt": fox}),
        (f"generate_alias_{alias}", {"prompt": "a lighthouse on a cliff at dusk", "size": alias}),
        (f"generate_n{max_n}", {"prompt": "a bowl of ripe peaches, still life", "n": max_n}),
        ("generate_hd", {"prompt": "a mountain lake at sunrise", "quality": "hd"}),
    ]
    log, ok, source = [], True, None

    def record(name: str, endpoint: str, params: dict, call) -> None:
        nonlocal ok, source
        t0 = time.monotonic()
        entry: dict = {"name": name, "request": {"endpoint": endpoint, **params}}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                resp = call()
        except openai.BadRequestError as exc:
            entry["seconds"] = round(time.monotonic() - t0, 1)
            body = exc.body if isinstance(exc.body, dict) else {}
            entry["error"] = {"status": exc.status_code, "code": body.get("code")}
            ok &= body.get("code") == "request_too_expensive"
            log.append(entry)
            print(f"{name}: {entry['seconds']}s -> 400 {body.get('code')}")
            return
        entry["seconds"] = round(time.monotonic() - t0, 1)
        entry["response_size"] = resp.size
        entry["files"] = []
        for i, item in enumerate(resp.data, 1):
            data = base64.b64decode(item.b64_json)
            fname = f"{name}.png" if len(resp.data) == 1 else f"{name}_{i}.png"
            (out_dir / fname).write_bytes(data)
            img = Image.open(io.BytesIO(data))
            img.load()
            text = dict(getattr(img, "text", {}) or {})
            entry["files"].append(
                {"file": fname, "pixels": list(img.size), "text_metadata": text or None}
            )
            ok &= f"{img.size[0]}x{img.size[1]}" == resp.size and not text
            if name == "generate_warm":
                source = data
        log.append(entry)
        print(f"{name}: {entry['seconds']}s size={resp.size} n={len(resp.data)}")

    for name, params in gen_runs:
        params = {"model": model, **params}
        record(
            name,
            "/v1/images/generations",
            params,
            lambda p=params: client.images.generate(response_format="b64_json", **p),
        )
    assert source is not None, "warm generation failed"
    with Image.open(io.BytesIO(source)) as src:
        mask = make_mask(src.size)
    (out_dir / "mask.png").write_bytes(mask)
    image = ("generate_warm.png", source, "image/png")
    edit_runs = [
        ("edit", {"prompt": "the same fox wearing a red knitted scarf"}, False),
        ("edit_masked", {"prompt": "a small campfire with bright flames in the snow"}, True),
    ]
    for name, params, with_mask in edit_runs:
        params = {"model": model, **params}
        files = {"image": image}
        if with_mask:
            files["mask"] = ("mask.png", mask, "image/png")
        shown = {"image": "generate_warm.png", **params}
        if with_mask:
            shown["mask"] = "mask.png (transparent lower-center box)"
        record(
            name,
            "/v1/images/edits",
            shown,
            lambda p=params, f=files: client.images.edit(response_format="b64_json", **f, **p),
        )
    params = {"model": model, "n": 2}
    record(
        "variation_n2",
        "/v1/images/variations",
        {"image": "generate_warm.png", **params},
        lambda: client.images.create_variation(image=image, response_format="b64_json", **params),
    )
    (out_dir / "generations.json").write_text(json.dumps(log, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
