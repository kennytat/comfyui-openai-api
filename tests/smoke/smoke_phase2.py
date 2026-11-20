#!/usr/bin/env python3
"""Phase 2 GPU smoke test: images.edit (img2img + masked) and images.create_variation.

Env: ``GATEWAY_BASE_URL`` (e.g. ``http://127.0.0.1:8001/v1``), ``GATEWAY_API_KEY``.
Usage: ``python tests/smoke/smoke_phase2.py SOURCE_PNG OUT_DIR``

Writes the decoded outputs plus ``OUT_DIR/generations.json`` (request params, wall
seconds, response size, per-file pixels and PNG text metadata, which must be empty) and
the generated mask. Exit code 0 when every image decodes with the reported size and
carries no text metadata.
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
    source, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(
        base_url=os.environ["GATEWAY_BASE_URL"],
        api_key=os.environ["GATEWAY_API_KEY"],
        max_retries=0,
        timeout=120,
    )
    image = source.read_bytes()
    with Image.open(source) as src:
        mask = make_mask(src.size)
    (out_dir / "mask.png").write_bytes(mask)

    runs = [
        (
            "edit_scarf",
            "edit",
            {"prompt": "the same fox wearing a red knitted scarf", "model": "sdxl"},
            False,
        ),
        (
            "edit_masked_campfire",
            "edit",
            {"prompt": "a small campfire with bright flames in the snow", "model": "sdxl"},
            True,
        ),
        ("variation_n2", "variation", {"model": "sdxl", "n": 2}, False),
    ]
    log, ok = [], True
    for name, kind, params, with_mask in runs:
        files = {"image": (source.name, image, "image/png")}
        if with_mask:
            files["mask"] = ("mask.png", mask, "image/png")
        t0 = time.monotonic()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if kind == "edit":
                resp = client.images.edit(**files, response_format="b64_json", **params)
            else:
                resp = client.images.create_variation(**files, response_format="b64_json", **params)
        seconds = round(time.monotonic() - t0, 1)
        entry = {
            "name": name,
            "request": {
                "endpoint": f"/v1/images/{'edits' if kind == 'edit' else 'variations'}",
                "image": source.name,
                "mask": "mask.png (transparent lower-center box)" if with_mask else None,
                **params,
            },
            "seconds": seconds,
            "response_size": resp.size,
            "files": [],
        }
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
        log.append(entry)
        print(f"{name}: {seconds}s size={resp.size} files={[f['file'] for f in entry['files']]}")
    (out_dir / "generations.json").write_text(json.dumps(log, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
