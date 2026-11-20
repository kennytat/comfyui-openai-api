#!/usr/bin/env python3
"""Validate the built-in workflows against ComfyUI ``/object_info``.

Usage:
    python gateway/scripts/check_workflows.py --fixture [PATH]
    python gateway/scripts/check_workflows.py --live http://localhost:8188 \
        [--output-dir comfyui/storage-user/output/openai-api]

``--fixture`` (offline): load the registry, build each enabled model's graph and run
``validate_graph`` against the committed fixture (default
``gateway/tests/fixtures/object_info_sdxl.json``).

``--live``: fetch the full ``/object_info``, validate, then ``POST /prompt`` with
``n=1, steps=1`` and assert no ``node_errors``; immediately cancel it, wait until it
leaves ``/queue``, delete any output files its ``/history`` lists (only inside the
``openai-api`` output folder) and delete its history entry. Exit code 0 == all passed.
"""

from __future__ import annotations

import argparse
import asyncio
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
from app.comfy.workflows import GenerationParams, ModelEntry, Registry, build_prompt  # noqa: E402

DEFAULT_FIXTURE = GATEWAY_DIR / "tests" / "fixtures" / "object_info_sdxl.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "comfyui" / "storage-user" / "output" / "openai-api"
OUTPUT_SUBFOLDER = "openai-api"


def check_graph(model: ModelEntry, object_info: dict, prefix: str) -> tuple[dict, list[str]]:
    w, h = model.resolve_size(None)  # type: ignore[misc]
    params = GenerationParams(
        prompt="workflow check", width=w, height=h, n=1, seed=0, steps=1, filename_prefix=prefix
    )
    graph = build_prompt(model, params)
    skip = [(f.loader, f.input) for f in model.required_files]
    errors = validate_graph(graph, object_info, skip_combo=skip)
    return graph, errors


def delete_outputs(client: ComfyClient, entry: dict, model: ModelEntry, output_dir: Path) -> int:
    """Delete output files listed in history, refusing anything outside output_dir."""
    root = output_dir.resolve()
    removed = 0
    for img in client.output_images(entry, model.output_node):
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
        _, errors = check_graph(model, object_info, f"{OUTPUT_SUBFOLDER}/check")
        status = "OK" if not errors else "FAIL"
        print(f"[{status}] {model.id} ({model.workflow_file}) vs {fixture.name}")
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


async def run_live(registry: Registry, url: str, output_dir: Path) -> int:
    failures = 0
    async with ComfyClient(url) as client:
        try:
            object_info = await client.object_info()
        except ComfyError as exc:
            print(f"[FAIL] cannot fetch /object_info from {url}: {exc}")
            return 1
        for model in registry.enabled():
            prefix = f"{OUTPUT_SUBFOLDER}/check-{uuid.uuid4().hex[:12]}"
            graph, errors = check_graph(model, object_info, prefix)
            for f in model.required_files:
                files = await client.available_files(f.loader, f.input) or []
                if f.value not in files:
                    errors.append(f"required file {f.value!r} not listed by {f.loader}.{f.input}")
            if errors:
                print(f"[FAIL] {model.id}: /object_info validation")
                for e in errors:
                    print(f"  - {e}")
                failures += 1
                continue
            try:
                prompt_id = await client.submit(graph)
            except ComfyError as exc:
                print(f"[FAIL] {model.id}: /prompt rejected: {exc.message}")
                if exc.details:
                    print(json.dumps(exc.details, indent=2))
                failures += 1
                continue
            print(f"[OK] {model.id}: /prompt accepted ({prompt_id}); cancelling")
            outcome = await client.cancel(prompt_id)
            gone = await _wait_gone(client, prompt_id)
            entry = await client.get_history(prompt_id)
            removed = delete_outputs(client, entry, model, output_dir) if entry else 0
            await client.delete_history(prompt_id)
            left = (await client.queue_snapshot()).contains(prompt_id)
            print(f"  cancel={outcome} left_queue={gone} outputs_deleted={removed}")
            if left or not gone:
                print(f"[FAIL] {model.id}: prompt {prompt_id} still in /queue")
                failures += 1
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
    args = ap.parse_args(argv)

    registry = Registry.load(args.workflows)
    if args.fixture:
        failures = asyncio.run(run_fixture(registry, Path(args.fixture)))
    else:
        failures = asyncio.run(run_live(registry, args.live, Path(args.output_dir)))
    print("all workflows passed" if not failures else f"{failures} model(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
