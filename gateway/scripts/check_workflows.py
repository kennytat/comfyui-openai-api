#!/usr/bin/env python3
"""Validate the built-in workflows against ComfyUI ``/object_info``.

Usage:
    python gateway/scripts/check_workflows.py --fixture [PATH]
    python gateway/scripts/check_workflows.py --live http://localhost:8188 \
        [--output-dir comfyui/storage-user/output/openai-api] \
        [--input-dir comfyui/storage-user/input/openai-api]

Every operation of every enabled model is checked (``generate`` plus the registry
``operations``: ``edit``, ``edit_masked``, ``variation``).

``--fixture`` (offline): load the registry, build each graph and run ``validate_graph``
against the committed fixture (default ``gateway/tests/fixtures/object_info_sdxl.json``).
Uploaded-file inputs (``LoadImage.image``, ``LoadImageMask.image``) skip the combo check.

``--live``: fetch the full ``/object_info`` and validate; for img2img/inpaint operations
upload a small generated test image (and mask) via ``/upload/image`` first; then
``POST /prompt`` with ``n=1, steps=1`` and assert no ``node_errors``; immediately cancel it,
wait until it leaves ``/queue``, delete any output files its ``/history`` lists (only inside
the ``openai-api`` output folder), delete its history entry and the uploaded inputs (only
inside the ``openai-api`` input folder). Exit code 0 == all passed.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import sys
import time
import uuid
from pathlib import Path

GATEWAY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GATEWAY_DIR.parent
sys.path.insert(0, str(GATEWAY_DIR))

from app.comfy.client import ComfyClient, ComfyError  # noqa: E402
from app.comfy.object_info import validate_graph  # noqa: E402
from app.comfy.workflows import (  # noqa: E402
    OPERATIONS,
    GenerationParams,
    ModelEntry,
    Operation,
    Registry,
    build_prompt,
)
from app.storage import delete_comfy_outputs  # noqa: E402

DEFAULT_FIXTURE = GATEWAY_DIR / "tests" / "fixtures" / "object_info_sdxl.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "comfyui" / "storage-user" / "output" / "openai-api"
DEFAULT_INPUT_DIR = REPO_ROOT / "comfyui" / "storage-user" / "input" / "openai-api"
OUTPUT_SUBFOLDER = "openai-api"
INPUT_SUBFOLDER = "openai-api"


def operations(model: ModelEntry) -> list[Operation]:
    return [op for name in OPERATIONS if (op := model.operation(name)) is not None]


def make_test_inputs(op: Operation, width: int, height: int) -> dict[str, bytes]:
    """Small PNG uploads the operation needs: ``{"image": ..., "mask": ...}``."""
    from PIL import Image

    out: dict[str, bytes] = {}
    if "image" in op.bindings:
        buf = io.BytesIO()
        Image.new("RGB", (width, height), (90, 120, 160)).save(buf, format="PNG")
        out["image"] = buf.getvalue()
    if "mask" in op.bindings:
        mask = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        mask.paste((0, 0, 0, 0), (width // 4, height // 4, 3 * width // 4, 3 * height // 4))
        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        out["mask"] = buf.getvalue()
    return out


def check_graph(
    model: ModelEntry,
    op: Operation,
    object_info: dict,
    prefix: str,
    inputs: dict[str, str] | None = None,
) -> tuple[dict, list[str]]:
    w, h = model.resolve_size(None)  # type: ignore[misc]
    inputs = inputs or {name: f"{INPUT_SUBFOLDER}/check_{name}.png" for name in ("image", "mask")}
    params = GenerationParams(
        prompt="workflow check",
        width=w,
        height=h,
        n=1,
        seed=0,
        steps=1,
        filename_prefix=prefix,
        operation=op.name,
        image=inputs.get("image") if "image" in op.bindings else None,
        mask=inputs.get("mask") if "mask" in op.bindings else None,
    )
    graph = build_prompt(model, params)
    skip = [(f.loader, f.input) for f in model.required_files] + op.upload_inputs()
    errors = validate_graph(graph, object_info, skip_combo=skip)
    return graph, errors


def delete_outputs(client: ComfyClient, entry: dict, output_node: str, output_dir: Path) -> int:
    """Delete output files listed in history, refusing anything outside output_dir."""
    root = output_dir.resolve()
    removed = 0
    for img in client.output_images(entry, output_node):
        sub = img.subfolder.strip("/")
        if sub != OUTPUT_SUBFOLDER and not sub.startswith(OUTPUT_SUBFOLDER + "/"):
            print(f"  ! refusing to delete {img.filename!r} in subfolder {img.subfolder!r}")
            continue
        rel = sub[len(OUTPUT_SUBFOLDER) :].lstrip("/")
        target = (root / rel / img.filename).resolve()
        if not target.is_relative_to(root):
            print(f"  ! refusing path outside output dir: {target}")
            continue
        if target.is_file() and not target.is_symlink():
            target.unlink()
            removed += 1
        else:
            print(f"  ! output file not found locally: {target}")
    return removed


async def run_fixture(registry: Registry, fixture: Path) -> int:
    object_info = json.loads(fixture.read_text())
    failures = 0
    for model in registry.enabled():
        for op in operations(model):
            _, errors = check_graph(model, op, object_info, f"{OUTPUT_SUBFOLDER}/check")
            status = "OK" if not errors else "FAIL"
            print(f"[{status}] {model.id}/{op.name} ({op.workflow_file}) vs {fixture.name}")
            for e in errors:
                print(f"  - {e}")
            failures += bool(errors)
    return failures


async def _wait_gone(client: ComfyClient, prompt_id: str, timeout_s: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not (await client.queue_snapshot()).contains(prompt_id):
            return True
        await asyncio.sleep(0.5)
    return False


async def _check_live_op(
    client: ComfyClient,
    model: ModelEntry,
    op: Operation,
    object_info: dict,
    output_dir: Path,
    input_dir: Path,
    file_errors: list[str],
) -> bool:
    label = f"{model.id}/{op.name}"
    tag = uuid.uuid4().hex[:12]
    prefix = f"{OUTPUT_SUBFOLDER}/check-{tag}"
    w, h = model.resolve_size(None)  # type: ignore[misc]
    uploaded: list[str] = []
    try:
        inputs: dict[str, str] = {}
        for name, data in make_test_inputs(op, w, h).items():
            ref = await client.upload_image(data, f"check-{tag}_{name}.png", INPUT_SUBFOLDER)
            uploaded.append(ref)
            inputs[name] = ref
        graph, errors = check_graph(model, op, object_info, prefix, inputs)
        errors = file_errors + errors
        if errors:
            print(f"[FAIL] {label}: /object_info validation")
            for e in errors:
                print(f"  - {e}")
            return False
        try:
            prompt_id = await client.submit(graph)
        except ComfyError as exc:
            print(f"[FAIL] {label}: /prompt rejected: {exc.message}")
            if exc.details:
                print(json.dumps(exc.details, indent=2))
            return False
        print(f"[OK] {label} ({op.workflow_file}): /prompt accepted ({prompt_id}); cancelling")
        outcome = await client.cancel(prompt_id)
        gone = await _wait_gone(client, prompt_id)
        entry = await client.get_history(prompt_id)
        removed = delete_outputs(client, entry, op.output_node, output_dir) if entry else 0
        await client.delete_history(prompt_id)
        left = (await client.queue_snapshot()).contains(prompt_id)
        print(f"  cancel={outcome} left_queue={gone} outputs_deleted={removed}")
        if left or not gone:
            print(f"[FAIL] {label}: prompt {prompt_id} still in /queue")
            return False
        return True
    finally:
        if uploaded:
            deleted = delete_comfy_outputs(uploaded, input_dir, INPUT_SUBFOLDER)
            print(f"  inputs_uploaded={len(uploaded)} inputs_deleted={len(deleted)}")


async def run_live(registry: Registry, url: str, output_dir: Path, input_dir: Path) -> int:
    failures = 0
    async with ComfyClient(url, input_subfolder=INPUT_SUBFOLDER) as client:
        try:
            object_info = await client.object_info()
        except ComfyError as exc:
            print(f"[FAIL] cannot fetch /object_info from {url}: {exc}")
            return 1
        for model in registry.enabled():
            file_errors: list[str] = []
            for f in model.required_files:
                files = await client.available_files(f.loader, f.input) or []
                if f.value not in files:
                    file_errors.append(
                        f"required file {f.value!r} not listed by {f.loader}.{f.input}"
                    )
            for op in operations(model):
                ok = await _check_live_op(
                    client, model, op, object_info, output_dir, input_dir, file_errors
                )
                failures += not ok
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", nargs="?", const=str(DEFAULT_FIXTURE), metavar="PATH")
    mode.add_argument("--live", metavar="COMFYUI_URL")
    ap.add_argument("--workflows", default=str(GATEWAY_DIR / "workflows"))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    args = ap.parse_args(argv)

    registry = Registry.load(args.workflows)
    if args.fixture:
        failures = asyncio.run(run_fixture(registry, Path(args.fixture)))
    else:
        failures = asyncio.run(
            run_live(registry, args.live, Path(args.output_dir), Path(args.input_dir))
        )
    print("all workflows passed" if not failures else f"{failures} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
