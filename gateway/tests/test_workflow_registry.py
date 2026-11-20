"""Registry loading, prompt patching and estimate tests (Step 1.1)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from app.comfy.workflows import (
    GenerationParams,
    Registry,
    WorkflowError,
    build_prompt,
    estimate_seconds,
    resolve_params,
)

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


@pytest.fixture
def registry() -> Registry:
    return Registry.load(WORKFLOWS)


@pytest.fixture
def sdxl(registry):
    return registry.get("sdxl")


def _params(**kw) -> GenerationParams:
    base = dict(prompt="a red fox", width=1024, height=1024, n=1, seed=42)
    base.update(kw)
    return GenerationParams(**base)


def _copy_workflows(tmp_path: Path) -> Path:
    dst = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, dst)
    return dst


def _edit_registry(directory: Path, mutate) -> None:
    path = directory / "registry.json"
    data = json.loads(path.read_text())
    mutate(data["models"][0])
    path.write_text(json.dumps(data))


def test_load_builtin_registry(registry, sdxl):
    assert len(registry) == 1
    assert "sdxl" in registry
    assert registry.get("nope") is None
    assert [m.id for m in registry.enabled()] == ["sdxl"]
    assert sdxl.output_node == "9"
    assert sdxl.checkpoint == "sd_xl_base_1.0.safetensors"
    assert sdxl.required_files[0].loader == "CheckpointLoaderSimple"
    assert sdxl.required_files[0].input == "ckpt_name"
    assert sdxl.bindings["prompt"] == ("6", "text")
    assert sdxl.bindings["n"] == ("5", "batch_size")


def test_build_prompt_patches_exact_node_ids(sdxl):
    graph = build_prompt(
        sdxl,
        _params(
            width=832,
            height=1216,
            n=3,
            seed=1234,
            steps=17,
            cfg_scale=7.5,
            negative_prompt="blurry",
            filename_prefix="openai-api/job-abc",
        ),
    )
    assert graph["6"]["inputs"]["text"] == "a red fox"
    assert graph["7"]["inputs"]["text"] == "blurry"
    assert graph["5"]["inputs"] == {"width": 832, "height": 1216, "batch_size": 3}
    assert graph["3"]["inputs"]["seed"] == 1234
    assert graph["3"]["inputs"]["steps"] == 17
    assert graph["3"]["inputs"]["cfg"] == 7.5
    assert graph["9"]["inputs"]["filename_prefix"] == "openai-api/job-abc"
    # Untouched literals and links preserved.
    assert graph["3"]["inputs"]["sampler_name"] == "dpmpp_2m"
    assert graph["3"]["inputs"]["scheduler"] == "karras"
    assert graph["3"]["inputs"]["model"] == ["4", 0]
    assert graph["4"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"


def test_build_prompt_does_not_mutate_template(sdxl):
    before = json.dumps(sdxl.workflow, sort_keys=True)
    build_prompt(sdxl, _params(prompt="changed", width=640, height=1536))
    assert json.dumps(sdxl.workflow, sort_keys=True) == before


def test_quality_maps_to_steps_and_explicit_steps_win(sdxl):
    assert build_prompt(sdxl, _params(quality="low"))["3"]["inputs"]["steps"] == 15
    assert build_prompt(sdxl, _params(quality="standard"))["3"]["inputs"]["steps"] == 25
    assert build_prompt(sdxl, _params(quality="hd"))["3"]["inputs"]["steps"] == 35
    assert build_prompt(sdxl, _params(quality="high"))["3"]["inputs"]["steps"] == 35
    assert build_prompt(sdxl, _params(quality="hd", steps=9))["3"]["inputs"]["steps"] == 9
    assert build_prompt(sdxl, _params())["3"]["inputs"]["steps"] == 25


def test_resolve_params_fills_defaults(sdxl):
    p = resolve_params(sdxl, _params(seed=None))
    assert 0 <= p.seed <= 2**53
    assert p.steps == 25
    assert p.cfg_scale == 6.5
    assert p.negative_prompt == ""
    # Idempotent once resolved.
    assert resolve_params(sdxl, p) == p
    assert build_prompt(sdxl, p)["3"]["inputs"]["seed"] == p.seed


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, (1024, 1024)),
        ("auto", (1024, 1024)),
        ("832x1216", (832, 1216)),
        ("1792x1024", (1344, 768)),
        ("1024x1792", (768, 1344)),
        ("1536x1024", (1216, 832)),
        ("1024x1536", (832, 1216)),
        ("333x333", None),
        ("512x512", None),
        ("garbage", None),
    ],
)
def test_resolve_size(sdxl, requested, expected):
    assert sdxl.resolve_size(requested) == expected


def test_estimate_seconds_formula(sdxl):
    sdxl = dataclasses.replace(sdxl, est_seconds_per_step=0.5)
    p = _params(width=1024, height=1024, n=2, steps=20)
    warm = estimate_seconds(sdxl, p, last_checkpoint=sdxl.checkpoint, overhead_s=3.0)
    assert warm == pytest.approx(3.0 + 20 * 0.5 * 1.0 * 2)
    cold = estimate_seconds(sdxl, p, last_checkpoint=None, overhead_s=3.0)
    assert cold == pytest.approx(warm + sdxl.cold_load_s)
    other = estimate_seconds(sdxl, p, last_checkpoint="other.safetensors", overhead_s=3.0)
    assert other == pytest.approx(cold)
    # Pixel scaling and quality-derived steps.
    small = estimate_seconds(
        sdxl, _params(width=512, height=512, quality="low"), sdxl.checkpoint, overhead_s=0
    )
    assert small == pytest.approx(15 * 0.5 * 0.25)


def test_missing_node_id_raises_at_load(tmp_path):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, lambda m: m["bindings"].__setitem__("seed", ["99", "seed"]))
    with pytest.raises(WorkflowError, match="missing node '99'"):
        Registry.load(d)


def test_missing_input_key_raises_at_load(tmp_path):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, lambda m: m["bindings"].__setitem__("seed", ["3", "noise_seed"]))
    with pytest.raises(WorkflowError, match="no input 'noise_seed'"):
        Registry.load(d)


def test_missing_required_binding_raises(tmp_path):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, lambda m: m["bindings"].pop("prompt"))
    with pytest.raises(WorkflowError, match="missing required bindings"):
        Registry.load(d)


def test_bad_output_node_and_alias_raise(tmp_path):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, lambda m: m.__setitem__("output_node", "42"))
    with pytest.raises(WorkflowError, match="output_node"):
        Registry.load(d)
    d2 = _copy_workflows(tmp_path / "b")
    _edit_registry(d2, lambda m: m["size_aliases"].__setitem__("1792x1024", "999x999"))
    with pytest.raises(WorkflowError, match="size alias"):
        Registry.load(d2)


def test_ui_format_workflow_rejected(tmp_path):
    d = _copy_workflows(tmp_path)
    (d / "sdxl.json").write_text(json.dumps({"nodes": [], "links": [], "version": 0.4}))
    with pytest.raises(WorkflowError, match="UI format"):
        Registry.load(d)


def test_non_api_node_rejected(tmp_path):
    d = _copy_workflows(tmp_path)
    graph = json.loads((d / "sdxl.json").read_text())
    del graph["8"]["class_type"]
    (d / "sdxl.json").write_text(json.dumps(graph))
    with pytest.raises(WorkflowError, match="not API format"):
        Registry.load(d)


def test_missing_registry_file(tmp_path):
    with pytest.raises(WorkflowError, match="not found"):
        Registry.load(tmp_path)
