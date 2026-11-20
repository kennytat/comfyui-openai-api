"""Phase 3 models (sd15, flux-schnell): registry entries, graphs and end-to-end routing."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from app.comfy.object_info import validate_graph
from app.comfy.workflows import OPERATIONS, GenerationParams, Registry, build_prompt
from tests.helpers import AUTH, make_png

WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "object_info_sdxl.json"
SD15_CKPT = "v1-5-pruned-emaonly-fp16.safetensors"


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load(WORKFLOWS)


@pytest.fixture(scope="module")
def object_info() -> dict:
    return json.loads(FIXTURE.read_text())


def _decode(item: dict) -> Image.Image:
    img = Image.open(io.BytesIO(base64.b64decode(item["b64_json"])))
    img.load()
    return img


def _nodes(graph: dict, class_type: str) -> list[dict]:
    return [n["inputs"] for n in graph.values() if n["class_type"] == class_type]


# ------------------------------------------------------------------ registry / graphs


def test_registry_lists_phase3_models(registry):
    ids = [m.id for m in registry.models]
    assert ids[0] == "sdxl" and "sd15" in ids
    assert all(m.enabled for m in registry.models if m.id in ("sdxl", "sd15"))


def test_sd15_entry(registry):
    sd15 = registry.get("sd15")
    assert sd15.checkpoint == SD15_CKPT
    assert sd15.default_size == "512x512"
    assert sd15.resolve_size(None) == (512, 512)
    assert sd15.resolve_size("auto") == (512, 512)
    # OpenAI sizes map to the nearest native SD 1.5 bucket.
    assert sd15.resolve_size("1024x1024") == (512, 512)
    assert sd15.resolve_size("256x256") == (512, 512)
    assert sd15.resolve_size("1536x1024") == (768, 512)
    assert sd15.resolve_size("1024x1536") == (512, 768)
    assert sd15.resolve_size("1792x1024") == (896, 512)
    assert sd15.resolve_size("1024x1792") == (512, 896)
    assert sd15.resolve_size("640x640") == (640, 640)
    assert sd15.resolve_size("832x1216") is None  # SDXL bucket, not SD 1.5
    assert sd15.resolve_steps("low") == 15 and sd15.resolve_steps("hd") == 35
    assert sd15.resolve_steps(None) == 25
    assert set(sd15.operations) == {"edit", "edit_masked", "variation"}
    # Every pixel count stays in the SD 1.5 sweet spot (~0.25-0.46 MP).
    for s in sd15.sizes:
        w, h = map(int, s.split("x"))
        assert 0.25e6 <= w * h <= 0.47e6 and w % 64 == 0 and h % 64 == 0


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_sd15_graphs_validate_and_load_sd15_checkpoint(registry, object_info, operation):
    sd15 = registry.get("sd15")
    op = sd15.operation(operation)
    graph = build_prompt(
        sd15,
        GenerationParams(
            prompt="a lighthouse",
            width=768,
            height=512,
            n=2,
            seed=7,
            negative_prompt="blurry",
            operation=operation,
            image="openai-api/j_image.png",
            mask="openai-api/j_mask.png",
        ),
    )
    ckpts = {n["ckpt_name"] for n in _nodes(graph, "CheckpointLoaderSimple")}
    assert ckpts == {SD15_CKPT}
    skip = [(f.loader, f.input) for f in sd15.required_files] + op.upload_inputs()
    assert validate_graph(graph, object_info, skip_combo=skip) == []
    ks = _nodes(graph, "KSampler")[0]
    assert ks["cfg"] == 7.0 and ks["seed"] == 7 and ks["steps"] == 25
    texts = [n["text"] for n in _nodes(graph, "CLIPTextEncode")]
    assert "blurry" in texts
    if operation == "generate":
        latent = _nodes(graph, "EmptyLatentImage")[0]
        assert (latent["width"], latent["height"], latent["batch_size"]) == (768, 512, 2)


# ------------------------------------------------------------------ end to end (fake ComfyUI)


async def test_models_lists_sd15_when_checkpoint_present(gw):
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert [m["id"] for m in r.json()["data"]] == ["sdxl"]
    gw.fake.extra_checkpoints = [SD15_CKPT]
    gw.app.state.catalog._available = None  # drop the cached availability set
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert [m["id"] for m in r.json()["data"]] == ["sdxl", "sd15"]
    assert (await gw.client.get("/v1/models/sd15", headers=AUTH)).status_code == 200


async def test_sd15_unavailable_without_checkpoint_404(gw):
    r = await gw.generate(model="sd15")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


async def test_sd15_generate_alias_and_edit_and_variation(gw):
    gw.fake.extra_checkpoints = [SD15_CKPT]
    r = await gw.generate(model="sd15", size="1024x1024", n=2)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["size"] == "512x512"
    assert [_decode(d).size for d in body["data"]] == [(512, 512)] * 2
    graph = gw.fake.prompt_bodies[-1]["prompt"]
    assert _nodes(graph, "CheckpointLoaderSimple")[0]["ckpt_name"] == SD15_CKPT

    r = await gw.edit(model="sd15", size="1536x1024")
    assert r.status_code == 200, r.text
    assert r.json()["size"] == "768x512" and _decode(r.json()["data"][0]).size == (768, 512)

    r = await gw.variation(model="sd15", image=make_png(300, 300))
    assert r.status_code == 200, r.text
    assert _decode(r.json()["data"][0]).size == (512, 512)
    assert gw.queue.active == 0


# ------------------------------------------------------------------ flux-schnell

FLUX_CKPT = "flux1-schnell-fp8.safetensors"


def test_flux_entry(registry):
    flux = registry.get("flux-schnell")
    assert flux.enabled and flux.checkpoint == FLUX_CKPT
    assert flux.default_size == "1024x1024"
    assert flux.sizes == registry.get("sdxl").sizes
    assert flux.size_aliases == registry.get("sdxl").size_aliases
    assert {flux.resolve_steps(q) for q in ("low", "standard", "auto", "hd", "high")} == {4}
    assert flux.resolve_steps(None) == 4
    assert set(flux.operations) == {"edit", "edit_masked", "variation"}
    # Negative prompt and cfg are not exposed: Flux schnell runs at cfg 1.0.
    for op in (flux.operation(name) for name in OPERATIONS):
        assert "negative_prompt" not in op.bindings and "cfg_scale" not in op.bindings


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_flux_graphs_validate_and_ignore_negative_and_cfg(registry, object_info, operation):
    flux = registry.get("flux-schnell")
    op = flux.operation(operation)
    graph = build_prompt(
        flux,
        GenerationParams(
            prompt="a lighthouse",
            width=1216,
            height=832,
            n=2,
            seed=9,
            cfg_scale=7.5,
            negative_prompt="blurry",
            quality="hd",
            operation=operation,
            image="openai-api/j_image.png",
            mask="openai-api/j_mask.png",
        ),
    )
    assert {n["ckpt_name"] for n in _nodes(graph, "CheckpointLoaderSimple")} == {FLUX_CKPT}
    skip = [(f.loader, f.input) for f in flux.required_files] + op.upload_inputs()
    assert validate_graph(graph, object_info, skip_combo=skip) == []
    ks = _nodes(graph, "KSampler")[0]
    assert (ks["cfg"], ks["steps"], ks["sampler_name"], ks["scheduler"]) == (
        1.0,
        4,
        "euler",
        "simple",
    )
    assert ks["seed"] == 9
    texts = [n["text"] for n in _nodes(graph, "CLIPTextEncode")]
    assert "a lighthouse" in texts and "blurry" not in texts
    assert not _nodes(graph, "EmptyLatentImage")
    if operation == "generate":
        latent = _nodes(graph, "EmptySD3LatentImage")[0]
        assert (latent["width"], latent["height"], latent["batch_size"]) == (1216, 832, 2)


def test_flux_estimate_includes_cold_load(registry):
    from app.comfy.workflows import estimate_seconds

    flux = registry.get("flux-schnell")
    p = GenerationParams(prompt="x", width=1024, height=1024, n=1)
    warm = estimate_seconds(flux, p, FLUX_CKPT, overhead_s=3.0)
    cold = estimate_seconds(flux, p, "sd_xl_base_1.0.safetensors", overhead_s=3.0)
    assert warm == pytest.approx(3.0 + 4 * flux.est_seconds_per_step)
    assert cold == pytest.approx(warm + flux.cold_load_s)


async def test_flux_generate_edit_variation_end_to_end(gw):
    gw.fake.extra_checkpoints = [FLUX_CKPT]
    r = await gw.generate(model="flux-schnell", size="1536x1024", negative_prompt="blurry")
    assert r.status_code == 200, r.text
    assert r.json()["size"] == "1216x832"
    assert _decode(r.json()["data"][0]).size == (1216, 832)
    graph = gw.fake.prompt_bodies[-1]["prompt"]
    assert _nodes(graph, "EmptySD3LatentImage")[0]["batch_size"] == 1
    assert "blurry" not in [n["text"] for n in _nodes(graph, "CLIPTextEncode")]

    r = await gw.edit(model="flux-schnell", mask=None)
    assert r.status_code == 200, r.text
    assert _nodes(gw.fake.prompt_bodies[-1]["prompt"], "KSampler")[0]["steps"] == 4

    r = await gw.variation(model="flux-schnell", n=2)
    assert r.status_code == 200, r.text
    assert len(r.json()["data"]) == 2 and gw.queue.active == 0


async def test_flux_n_max_rejected_up_front_when_too_expensive(gw):
    """A job whose warm estimate exceeds the budget is a 400 before anything is queued."""
    gw.fake.extra_checkpoints = [FLUX_CKPT]
    r = await gw.generate(model="flux-schnell", n=4, size="1024x1024", steps=80)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "request_too_expensive"
    assert gw.fake.prompt_bodies == [] and gw.queue.active == 0
