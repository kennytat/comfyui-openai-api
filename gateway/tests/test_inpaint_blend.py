"""Masked-edit blending: gateway mask feathering and the inpaint graph wiring per model."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.comfy.object_info import validate_graph
from app.comfy.workflows import GenerationParams, Registry, build_prompt
from app.uploads import MASK_FEATHER_FRACTION, prepare_inputs

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "object_info_sdxl.json"
SIZES = {
    "sdxl": (1024, 1024),
    "sd15": (512, 512),
    "flux-schnell": (1024, 1024),
    "flux-dev": (1024, 1024),
    "qwen-image": (1328, 1328),
}
DENOISE = {"sdxl": 1.0, "sd15": 1.0, "flux-schnell": 0.8, "flux-dev": 0.8, "qwen-image": 1.0}


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(WORKFLOWS)


@pytest.fixture(scope="module")
def object_info() -> dict:
    return json.loads(FIXTURE.read_text())


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _box_mask(size: int) -> bytes:
    """Opaque mask with a transparent (edit) box covering the middle half."""
    m = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    m.paste((0, 0, 0, 0), (size // 4, size // 4, 3 * size // 4, 3 * size // 4))
    return _png(m)


def _alpha(mask_png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(mask_png)).getchannel("A")


@pytest.mark.parametrize("size", [512, 1024])
def test_mask_is_feathered_around_the_edge(size):
    image = _png(Image.new("RGB", (size, size), (40, 90, 140)))
    _, mask_png = prepare_inputs(image, _box_mask(size), size, size, 1 << 24)
    a = _alpha(mask_png)
    sigma = MASK_FEATHER_FRACTION * size
    edge, mid = size // 4, size // 2
    row = [a.getpixel((x, mid)) for x in range(edge - int(4 * sigma), edge + int(4 * sigma))]
    assert row == sorted(row, reverse=True)  # keep (255) -> edit (0), monotonic ramp
    assert 64 < a.getpixel((edge, mid)) < 192  # the boundary sits mid-ramp
    assert a.getpixel((edge - int(4 * sigma), mid)) == 255  # far outside: untouched
    assert a.getpixel((edge + int(4 * sigma), mid)) == 0  # far inside: full edit
    assert a.getpixel((mid, mid)) == 0 and a.getpixel((2, 2)) == 255
    ramp = sum(1 for x in range(size) if 0 < a.getpixel((x, mid)) < 255)
    assert 2 * sigma <= ramp / 2 <= 8 * sigma  # two edges, each a few sigma wide


def test_fully_transparent_and_opaque_masks_stay_uniform():
    image = _png(Image.new("RGB", (64, 64), (1, 2, 3)))
    for value in (0, 255):
        mask = _png(Image.new("RGBA", (64, 64), (0, 0, 0, value)))
        _, mask_png = prepare_inputs(image, mask, 64, 64, 1 << 24)
        assert _alpha(mask_png).getextrema() == (value, value)


def _inputs(graph: dict, node_id: str) -> dict:
    return graph[node_id]["inputs"]


def _only(graph: dict, class_type: str) -> str:
    ids = [k for k, v in graph.items() if v["class_type"] == class_type]
    assert len(ids) == 1, (class_type, ids)
    return ids[0]


@pytest.mark.parametrize("model_id", list(SIZES))
def test_inpaint_graph_composites_original_outside_mask(registry, object_info, model_id):
    model = registry.get(model_id)
    op = model.operation("edit_masked")
    w, h = SIZES[model_id]
    graph = build_prompt(
        model,
        GenerationParams(
            prompt="a small campfire",
            width=w,
            height=h,
            n=3,
            seed=1,
            operation="edit_masked",
            image="openai-api/j_image.png",
            mask="openai-api/j_mask.png",
        ),
    )
    skip = [(f.loader, f.input) for f in model.required_files] + op.upload_inputs()
    assert validate_graph(graph, object_info, skip_combo=skip) == []

    loader = "UNETLoader" if model_id == "qwen-image" else "CheckpointLoaderSimple"
    ckpt = _only(graph, loader)
    image = _only(graph, "LoadImage")
    mask = _only(graph, "LoadImageMask")
    decode = _only(graph, "VAEDecode")
    sampler = _only(graph, "KSampler")
    assert _inputs(graph, mask)["channel"] == "alpha"
    assert _inputs(graph, sampler)["denoise"] == DENOISE[model_id]
    # soft latent noise mask straight from the (gateway-feathered) mask
    noise = _only(graph, "SetLatentNoiseMask")
    assert _inputs(graph, noise)["mask"] == [mask, 0]
    repeat = _only(graph, "RepeatLatentBatch")
    assert _inputs(graph, repeat) == {"samples": [noise, 0], "amount": 3}

    # composite: decoded batch is the destination; the single original image is the
    # source (ComfyUI repeats source to the destination batch) pasted where keep == 1
    invert = _only(graph, "InvertMask")
    assert _inputs(graph, invert)["mask"] == [mask, 0]
    comp = _only(graph, "ImageCompositeMasked")
    assert _inputs(graph, comp) == {
        "destination": [decode, 0],
        "source": [image, 0],
        "x": 0,
        "y": 0,
        "resize_source": False,
        "mask": [invert, 0],
    }
    assert _inputs(graph, _only(graph, "SaveImage"))["images"] == [comp, 0]

    model_src = _inputs(graph, sampler)["model"][0]
    if model_id == "qwen-image":
        # Lightning-distilled Qwen-Image: plain noise mask at full denoise blends well
        # (the composite restores the outside); AuraFlow shift on the LoRA-patched model.
        assert not [v for v in graph.values() if v["class_type"] == "DifferentialDiffusion"]
        assert graph[model_src]["class_type"] == "ModelSamplingAuraFlow"
        lora = _inputs(graph, model_src)["model"][0]
        assert graph[lora]["class_type"] == "LoraLoaderModelOnly"
        assert _inputs(graph, lora)["model"] == [ckpt, 0]
    elif model_id.startswith("flux"):
        # distilled flux (schnell 4 steps, dev 15): no differential diffusion (it hardens
        # the core into a pasted-on patch); a higher constant shift keeps the start sigma
        # below 1 so the masked area sees its surroundings.
        assert not [v for v in graph.values() if v["class_type"] == "DifferentialDiffusion"]
        assert graph[model_src]["class_type"] == "ModelSamplingFlux"
        shift = _inputs(graph, model_src)
        assert shift["model"] == [ckpt, 0] and shift["max_shift"] == shift["base_shift"] > 1.15
    else:
        assert graph[model_src]["class_type"] == "DifferentialDiffusion"
        assert _inputs(graph, model_src)["model"] == [ckpt, 0]


@pytest.mark.parametrize("model_id", list(SIZES))
def test_other_operations_unchanged(registry, model_id):
    model = registry.get(model_id)
    for name in ("generate", "edit", "variation"):
        graph = model.operation(name).workflow
        classes = {v["class_type"] for v in graph.values()}
        assert not classes & {"DifferentialDiffusion", "ImageCompositeMasked", "InvertMask"}
