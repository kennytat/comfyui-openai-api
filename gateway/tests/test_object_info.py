"""Combo parsing and graph validation against the committed object_info fixture."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from app.comfy.object_info import combo_values, parse_combo, validate_graph
from app.comfy.workflows import GenerationParams, Registry, build_prompt

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def object_info() -> dict:
    return _load("object_info_sdxl.json")


@pytest.fixture
def sdxl_graph() -> dict:
    return json.loads((ROOT / "workflows" / "sdxl.json").read_text())


def test_parse_combo_legacy_format():
    assert parse_combo([["a.safetensors", "b.safetensors"], {"tooltip": "x"}]) == [
        "a.safetensors",
        "b.safetensors",
    ]
    assert parse_combo([["only"]]) == ["only"]
    assert parse_combo([[], {}]) == []


def test_parse_combo_new_format():
    assert parse_combo(["COMBO", {"options": ["a", "b"], "multiselect": False}]) == ["a", "b"]


def test_parse_combo_non_combo_types():
    assert parse_combo(["INT", {"default": 1}]) is None
    assert parse_combo(["MODEL"]) is None
    assert parse_combo(["STRING", {"multiline": True}]) is None


@pytest.mark.parametrize(
    "spec",
    [
        ["COMBO", {}],
        ["COMBO"],
        ["COMBO", {"options": "a,b"}],
        [[1, 2], {}],
        {"options": ["a"]},
        [],
        "COMBO",
        [42],
    ],
)
def test_parse_combo_unknown_shape_returns_none_and_warns(spec, caplog):
    with caplog.at_level(logging.WARNING, logger="app.comfy.object_info"):
        assert parse_combo(spec) is None
    assert caplog.records, "unknown shapes must log a warning"


@pytest.mark.parametrize(
    "fixture", ["object_info_checkpoint_legacy.json", "object_info_checkpoint_combo.json"]
)
def test_checkpoint_fixtures_both_formats(fixture):
    info = _load(fixture)
    files = combo_values(info, "CheckpointLoaderSimple", "ckpt_name")
    assert "sd_xl_base_1.0.safetensors" in files


def test_fixture_represents_both_combo_formats(object_info):
    ckpt = object_info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
    sampler = object_info["KSampler"]["input"]["required"]["sampler_name"]
    assert isinstance(ckpt[0], list)  # legacy
    assert sampler[0] == "COMBO"  # new
    assert combo_values(object_info, "KSampler", "scheduler") and "karras" in combo_values(
        object_info, "KSampler", "scheduler"
    )
    assert combo_values(object_info, "Nope", "x") is None
    assert combo_values(object_info, "KSampler", "nope") is None


def test_sdxl_workflow_validates_against_fixture(sdxl_graph, object_info):
    assert validate_graph(sdxl_graph, object_info) == []


def test_patched_sdxl_graph_validates(object_info):
    model = Registry.load(ROOT / "workflows").get("sdxl")
    graph = build_prompt(
        model, GenerationParams(prompt="x", width=1344, height=768, n=4, seed=2**53, steps=35)
    )
    assert validate_graph(graph, object_info) == []


def test_dropped_required_input(sdxl_graph, object_info):
    g = copy.deepcopy(sdxl_graph)
    del g["3"]["inputs"]["scheduler"]
    errors = validate_graph(g, object_info)
    assert len(errors) == 1 and "missing required input 'scheduler'" in errors[0]


def test_dropped_required_link(sdxl_graph, object_info):
    g = copy.deepcopy(sdxl_graph)
    del g["8"]["inputs"]["vae"]
    assert any("missing required input 'vae'" in e for e in validate_graph(g, object_info))


def test_illegal_sampler_name(sdxl_graph, object_info):
    g = copy.deepcopy(sdxl_graph)
    g["3"]["inputs"]["sampler_name"] = "not_a_sampler"
    errors = validate_graph(g, object_info)
    assert len(errors) == 1 and "sampler_name" in errors[0] and "not_a_sampler" in errors[0]


def test_illegal_legacy_combo_and_skip(sdxl_graph, object_info):
    g = copy.deepcopy(sdxl_graph)
    g["4"]["inputs"]["ckpt_name"] = "missing.safetensors"
    assert any("ckpt_name" in e for e in validate_graph(g, object_info))
    # File-list inputs covered by required_files can be excluded.
    assert (
        validate_graph(g, object_info, skip_combo=[("CheckpointLoaderSimple", "ckpt_name")]) == []
    )


def test_unknown_class_type_and_dangling_link(sdxl_graph, object_info):
    g = copy.deepcopy(sdxl_graph)
    g["8"]["class_type"] = "VAEDecodeTiledMissing"
    g["9"]["inputs"]["images"] = ["77", 0]
    errors = validate_graph(g, object_info)
    assert any("unknown class_type 'VAEDecodeTiledMissing'" in e for e in errors)
    assert any("link to missing node '77'" in e for e in errors)


def test_numeric_range_checks(sdxl_graph, object_info):
    g = copy.deepcopy(sdxl_graph)
    g["3"]["inputs"]["steps"] = 0
    g["5"]["inputs"]["batch_size"] = 5000
    errors = validate_graph(g, object_info)
    assert any("steps" in e and "min" in e for e in errors)
    assert any("batch_size" in e and "max" in e for e in errors)


def test_validate_non_api_graph(object_info):
    assert validate_graph({}, object_info)
    assert validate_graph({"1": {"inputs": {}}}, object_info)


@pytest.mark.parametrize("operation", ["edit", "edit_masked", "variation"])
def test_img2img_workflows_validate_against_fixture(object_info, operation):
    sdxl = Registry.load(Path(__file__).resolve().parents[1] / "workflows").get("sdxl")
    op = sdxl.operation(operation)
    graph = build_prompt(
        sdxl,
        GenerationParams(
            prompt="x",
            width=1024,
            height=1024,
            n=2,
            seed=1,
            operation=operation,
            image="openai-api/j_image.png",
            mask="openai-api/j_mask.png",
        ),
    )
    # uploaded names are not in the static LoadImage combo list
    assert any("not in allowed options" in e for e in validate_graph(graph, object_info))
    assert validate_graph(graph, object_info, skip_combo=op.upload_inputs()) == []
    broken = json.loads(json.dumps(graph))
    if operation == "edit_masked":
        broken["13"]["inputs"]["channel"] = "luma"
        errors = validate_graph(broken, object_info, skip_combo=op.upload_inputs())
        assert any("channel" in e for e in errors)
    broken["12"]["inputs"]["amount"] = 65
    errors = validate_graph(broken, object_info, skip_combo=op.upload_inputs())
    assert any("> max 64" in e for e in errors)
