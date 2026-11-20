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
    assert "sdxl" in registry
    assert registry.get("nope") is None
    assert [m.id for m in registry.models][0] == "sdxl"
    assert len(registry) == len(registry.models)
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


# ------------------------------------------------------------------ operations (Phase 2)


def test_operations_loaded(sdxl):
    assert set(sdxl.operations) == {"edit", "edit_masked", "variation"}
    gen = sdxl.operation("generate")
    assert gen.workflow is sdxl.workflow and gen.output_node == "9"
    edit = sdxl.operation("edit")
    assert edit.workflow_file == "sdxl_img2img.json"
    assert edit.bindings["image"] == ("10", "image")
    assert edit.bindings["n"] == ("12", "amount")
    assert edit.defaults["denoise"] == 0.7
    assert sdxl.operation("edit_masked").bindings["mask"] == ("13", "image")
    assert sdxl.operation("variation").defaults == {"denoise": 0.55, "prompt": ""}
    assert sdxl.operation("upscale") is None
    assert edit.upload_inputs() == [("LoadImage", "image")]
    assert sdxl.operation("edit_masked").upload_inputs() == [
        ("LoadImage", "image"),
        ("LoadImageMask", "image"),
    ]


def test_build_prompt_img2img(sdxl):
    p = _params(operation="edit", n=2, image="openai-api/j_image.png", denoise=0.4)
    graph = build_prompt(sdxl, p)
    assert graph["10"]["inputs"]["image"] == "openai-api/j_image.png"
    assert graph["12"]["inputs"]["amount"] == 2
    assert graph["3"]["inputs"]["denoise"] == 0.4
    assert "5" not in graph  # no EmptyLatentImage; width/height are optional here
    masked = build_prompt(
        sdxl,
        _params(operation="edit_masked", image="openai-api/a.png", mask="openai-api/m.png"),
    )
    assert masked["13"]["inputs"] == {"image": "openai-api/m.png", "channel": "alpha"}
    assert masked["3"]["inputs"]["denoise"] == 1.0  # operation default
    assert masked["14"]["inputs"] == {"samples": ["11", 0], "mask": ["13", 0]}


def test_variation_defaults(sdxl):
    p = resolve_params(sdxl, _params(prompt="", operation="variation", image="openai-api/x"))
    assert p.prompt == "" and p.denoise == 0.55
    assert build_prompt(sdxl, p)["6"]["inputs"]["text"] == ""


def test_build_prompt_requires_upload_and_operation(sdxl):
    with pytest.raises(WorkflowError, match="needs an uploaded 'image'"):
        build_prompt(sdxl, _params(operation="edit"))
    with pytest.raises(WorkflowError, match="needs an uploaded 'mask'"):
        build_prompt(sdxl, _params(operation="edit_masked", image="openai-api/a.png"))
    with pytest.raises(WorkflowError, match="does not support operation 'upscale'"):
        build_prompt(sdxl, _params(operation="upscale"))
    no_ops = dataclasses.replace(sdxl, operations={})
    with pytest.raises(WorkflowError, match="does not support"):
        resolve_params(no_ops, _params(operation="variation"))


def test_estimate_seconds_operation_factors(sdxl):
    sdxl = dataclasses.replace(sdxl, est_seconds_per_step=0.5)
    edit = dataclasses.replace(sdxl.operations["edit"], est_factor=1.2, est_extra_s=2.0)
    sdxl = dataclasses.replace(sdxl, operations={**sdxl.operations, "edit": edit})
    p = _params(operation="edit", image="openai-api/x", n=2, steps=10)
    est = estimate_seconds(sdxl, p, last_checkpoint=sdxl.checkpoint, overhead_s=3.0)
    assert est == pytest.approx(3.0 + 10 * 0.5 * 1.0 * 2 * 1.2 + 2.0)


def test_registry_without_operations_still_loads(tmp_path):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, lambda m: m.pop("operations"))
    sdxl = Registry.load(d).get("sdxl")
    assert sdxl.operations == {} and sdxl.operation("edit") is None
    assert sdxl.operation("generate") is not None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda m: m.__setitem__("operations", []), "operations must be an object"),
        (lambda m: m["operations"].__setitem__("upscale", {}), "unknown operation"),
        (lambda m: m["operations"].__setitem__("generate", {}), "unknown operation"),
        (lambda m: m["operations"].__setitem__("edit", "x.json"), "must be an object"),
        (lambda m: m["operations"]["edit"].pop("bindings"), "missing field 'bindings'"),
        (lambda m: m["operations"]["edit"].__setitem__("bindings", []), "must be an object"),
        (lambda m: m["operations"]["edit"]["bindings"].pop("image"), "missing required"),
        (lambda m: m["operations"]["edit_masked"]["bindings"].pop("mask"), "missing required"),
        (
            lambda m: m["operations"]["edit"]["bindings"].__setitem__("image", ["99", "image"]),
            "missing node '99'",
        ),
        (lambda m: m["operations"]["variation"].__setitem__("output_node", "77"), "output_node"),
        (lambda m: m["operations"]["edit"].__setitem__("workflow", "nope.json"), "not found"),
        (lambda m: m.__setitem__("defaults", []), "defaults must be an object"),
        (lambda m: m["operations"]["edit"].__setitem__("defaults", 1), "must be an object"),
        (lambda m: m["operations"]["edit"]["defaults"].__setitem__("denoise", 0), "denoise"),
        (lambda m: m["operations"]["edit"]["defaults"].__setitem__("denoise", 1.5), "denoise"),
        (lambda m: m["operations"]["edit"]["defaults"].__setitem__("denoise", "x"), "denoise"),
        (lambda m: m["operations"]["edit"]["defaults"].__setitem__("denoise", True), "denoise"),
    ],
)
def test_operation_validation_errors(tmp_path, mutate, match):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, mutate)
    with pytest.raises(WorkflowError, match=match):
        Registry.load(d)


def test_denoise_default_of_one_accepted(tmp_path):
    d = _copy_workflows(tmp_path)
    _edit_registry(d, lambda m: m["operations"]["edit"]["defaults"].__setitem__("denoise", 1))
    assert Registry.load(d).get("sdxl").operations["edit"].defaults["denoise"] == 1


def test_operation_checkpoint_must_match_required(tmp_path):
    d = _copy_workflows(tmp_path)
    path = d / "sdxl_inpaint.json"
    graph = json.loads(path.read_text())
    loader = next(n for n in graph.values() if n["class_type"] == "CheckpointLoaderSimple")
    loader["inputs"]["ckpt_name"] = "other_model.safetensors"
    path.write_text(json.dumps(graph))
    with pytest.raises(WorkflowError, match="edit_masked.*other_model.safetensors"):
        Registry.load(d)
    _edit_registry(d, lambda m: m.__setitem__("required_files", []))  # nothing to compare
    assert Registry.load(d).get("sdxl").operations["edit_masked"] is not None
