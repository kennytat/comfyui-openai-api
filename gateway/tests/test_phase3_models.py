"""Phase 3 models (sd15, flux-schnell): registry entries, graphs and end-to-end routing."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
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


def _transparent_mask(size: int) -> bytes:
    m = Image.new("RGBA", (size, size), (0, 0, 0, 255))
    m.paste((0, 0, 0, 0), (size // 4, size // 4, 3 * size // 4, 3 * size // 4))
    buf = io.BytesIO()
    m.save(buf, format="PNG")
    return buf.getvalue()


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


# ------------------------------------------------------------------ flux-dev

FLUX_DEV_CKPT = "flux1-dev-fp8.safetensors"


def test_flux_dev_entry(registry):
    dev = registry.get("flux-dev")
    assert dev.enabled and dev.checkpoint == FLUX_DEV_CKPT
    assert dev.default_size == "1024x1024"
    assert dev.sizes == registry.get("sdxl").sizes
    assert dev.size_aliases == registry.get("sdxl").size_aliases
    assert dev.resolve_steps(None) == dev.resolve_steps("standard") == dev.defaults["steps"]
    assert dev.resolve_steps("low") < dev.resolve_steps("standard")
    assert set(dev.operations) == {"edit", "edit_masked", "variation"}
    # cfg_scale drives the distilled FluxGuidance (default 3.5); no negative prompt.
    assert dev.defaults["cfg_scale"] == 3.5
    for op in (dev.operation(name) for name in OPERATIONS):
        assert "negative_prompt" not in op.bindings
        node_id, input_name = op.bindings["cfg_scale"]
        assert op.workflow[node_id]["class_type"] == "FluxGuidance" and input_name == "guidance"


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_flux_dev_graphs_validate_and_bind_guidance(registry, object_info, operation):
    dev = registry.get("flux-dev")
    op = dev.operation(operation)
    params = GenerationParams(
        prompt="a lighthouse",
        width=1216,
        height=832,
        n=2,
        seed=11,
        negative_prompt="blurry",
        operation=operation,
        image="openai-api/j_image.png",
        mask="openai-api/j_mask.png",
    )
    graph = build_prompt(dev, params)
    assert {n["ckpt_name"] for n in _nodes(graph, "CheckpointLoaderSimple")} == {FLUX_DEV_CKPT}
    skip = [(f.loader, f.input) for f in dev.required_files] + op.upload_inputs()
    assert validate_graph(graph, object_info, skip_combo=skip) == []
    ks = _nodes(graph, "KSampler")[0]
    assert (ks["cfg"], ks["sampler_name"], ks["scheduler"]) == (1.0, "euler", "simple")
    assert ks["steps"] == dev.defaults["steps"] and ks["seed"] == 11
    # The positive conditioning goes through FluxGuidance; guidance defaults to 3.5.
    guidance_id = ks["positive"][0]
    assert graph[guidance_id]["class_type"] == "FluxGuidance"
    assert graph[guidance_id]["inputs"]["guidance"] == 3.5
    pos = graph[guidance_id]["inputs"]["conditioning"][0]
    assert graph[pos]["inputs"]["text"] == "a lighthouse"
    assert "blurry" not in [n["text"] for n in _nodes(graph, "CLIPTextEncode")]
    graph = build_prompt(dev, replace(params, cfg_scale=5.0))
    assert _nodes(graph, "FluxGuidance")[0]["guidance"] == 5.0
    if operation == "generate":
        latent = _nodes(graph, "EmptySD3LatentImage")[0]
        assert (latent["width"], latent["height"], latent["batch_size"]) == (1216, 832, 2)


def test_flux_dev_quality_steps_fit_budget_for_one_image(registry):
    """Every quality must be admissible warm at n=1 within the 80 s generation budget."""
    from app.comfy.workflows import estimate_seconds

    dev = registry.get("flux-dev")
    for quality in dev.quality_steps:
        for size in dev.sizes:
            w, h = map(int, size.split("x"))
            p = GenerationParams(prompt="x", width=w, height=h, n=1, quality=quality)
            for name in OPERATIONS:
                est = estimate_seconds(
                    dev, replace(p, operation=name), FLUX_DEV_CKPT, overhead_s=3.0
                )
                assert est <= 80, (quality, size, name, est)


async def test_flux_dev_end_to_end_and_n_max_rejected(gw):
    gw.fake.extra_checkpoints = [FLUX_DEV_CKPT]
    r = await gw.generate(model="flux-dev", size="1024x1792", cfg_scale=4.5)
    assert r.status_code == 200, r.text
    assert r.json()["size"] == "768x1344"
    graph = gw.fake.prompt_bodies[-1]["prompt"]
    assert _nodes(graph, "FluxGuidance")[0]["guidance"] == 4.5

    r = await gw.edit(model="flux-dev", mask=_transparent_mask(64))
    assert r.status_code == 200, r.text
    assert _nodes(gw.fake.prompt_bodies[-1]["prompt"], "ImageCompositeMasked")

    r = await gw.variation(model="flux-dev")
    assert r.status_code == 200, r.text

    sent = len(gw.fake.prompt_bodies)
    r = await gw.generate(model="flux-dev", n=4, size="1024x1024")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "request_too_expensive"
    assert len(gw.fake.prompt_bodies) == sent and gw.queue.active == 0


# ------------------------------------------------------------------ qwen-image

QWEN_UNET = "qwen_image_fp8_e4m3fn.safetensors"
QWEN_TE = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
QWEN_VAE = "qwen_image_vae.safetensors"
QWEN_NATIVE = {
    "1328x1328",
    "1664x928",
    "928x1664",
    "1472x1140",
    "1140x1472",
    "1584x1056",
    "1056x1584",
}


def _qwen_loader_files(model) -> dict[tuple[str, str], list[str]]:
    return {(f.loader, f.input): [f.value] for f in model.required_files}


def test_qwen_entry(registry):
    qwen = registry.get("qwen-image")
    # Cold-load tracking keys on the diffusion model, listed first.
    assert qwen.checkpoint == QWEN_UNET
    files = {(f.loader, f.input): f.value for f in qwen.required_files}
    assert files[("UNETLoader", "unet_name")] == QWEN_UNET
    assert files[("CLIPLoader", "clip_name")] == QWEN_TE
    assert files[("VAELoader", "vae_name")] == QWEN_VAE
    assert set(qwen.sizes) <= QWEN_NATIVE and qwen.default_size in qwen.sizes
    # OpenAI sizes map to the nearest native aspect ratio.
    for src, aspect in (
        ("1024x1024", 1.0),
        ("1792x1024", 1.75),
        ("1024x1792", 1 / 1.75),
        ("1536x1024", 1.5),
        ("1024x1536", 1 / 1.5),
    ):
        w, h = qwen.resolve_size(src)
        best = min(
            (abs(w2 / h2 - aspect), (w2, h2))
            for w2, h2 in (map(int, s.split("x")) for s in qwen.sizes)
        )[1]
        assert (w, h) == best, src
    assert qwen.resolve_size(None) == qwen.resolve_size("auto")
    assert qwen.resolve_size("1216x832") is None  # SDXL bucket, not Qwen
    assert set(qwen.operations) == {"edit", "edit_masked", "variation"}


@pytest.mark.parametrize("operation", list(OPERATIONS))
def test_qwen_graphs_validate_and_load_split_files(registry, object_info, operation):
    qwen = registry.get("qwen-image")
    op = qwen.operation(operation)
    w, h = qwen.resolve_size(None)
    graph = build_prompt(
        qwen,
        GenerationParams(
            prompt="a lighthouse",
            width=w,
            height=h,
            n=2,
            seed=13,
            negative_prompt="blurry",
            operation=operation,
            image="openai-api/j_image.png",
            mask="openai-api/j_mask.png",
        ),
    )
    skip = [(f.loader, f.input) for f in qwen.required_files] + op.upload_inputs()
    assert validate_graph(graph, object_info, skip_combo=skip) == []
    assert not _nodes(graph, "CheckpointLoaderSimple")
    for f in qwen.required_files:
        assert {n[f.input] for n in _nodes(graph, f.loader)} == {f.value}
    assert _nodes(graph, "CLIPLoader")[0]["type"] == "qwen_image"
    sampling = _nodes(graph, "ModelSamplingAuraFlow")
    assert len(sampling) == 1 and 2.5 <= sampling[0]["shift"] <= 3.5
    ks = _nodes(graph, "KSampler")[0]
    assert (ks["sampler_name"], ks["scheduler"], ks["seed"]) == ("euler", "simple", 13)
    assert ks["steps"] == qwen.defaults["steps"]
    lora = _nodes(graph, "LoraLoaderModelOnly")
    if lora:  # Lightning: distilled to cfg 1.0, negative prompt ignored
        assert ks["cfg"] == 1.0
        assert "blurry" not in [n["text"] for n in _nodes(graph, "CLIPTextEncode")]
    else:
        assert ks["cfg"] == qwen.defaults["cfg_scale"]
        assert "blurry" in [n["text"] for n in _nodes(graph, "CLIPTextEncode")]
    # every VAE node uses the Qwen VAE, every text node the Qwen text encoder
    vae_id = next(k for k, v in graph.items() if v["class_type"] == "VAELoader")
    for node in graph.values():
        if node["class_type"] in ("VAEDecode", "VAEEncode"):
            assert node["inputs"]["vae"] == [vae_id, 0]
    if operation == "generate":
        latent = _nodes(graph, "EmptySD3LatentImage")[0]
        assert (latent["width"], latent["height"], latent["batch_size"]) == (w, h, 2)


def test_qwen_operation_must_load_required_unet(tmp_path):
    import shutil

    from app.comfy.workflows import WorkflowError

    d = tmp_path / "wf"
    shutil.copytree(WORKFLOWS, d)
    path = d / "qwen-image_inpaint.json"
    graph = json.loads(path.read_text())
    next(n for n in graph.values() if n["class_type"] == "UNETLoader")["inputs"]["unet_name"] = (
        "other_unet.safetensors"
    )
    path.write_text(json.dumps(graph))
    with pytest.raises(WorkflowError, match="edit_masked.*UNETLoader.*other_unet"):
        Registry.load(d)


def test_qwen_estimate_cold_load_keys_on_diffusion_model(registry):
    from app.comfy.workflows import estimate_seconds

    qwen = registry.get("qwen-image")
    w, h = qwen.resolve_size(None)
    p = GenerationParams(prompt="x", width=w, height=h, n=1)
    warm = estimate_seconds(qwen, p, QWEN_UNET, overhead_s=3.0)
    assert estimate_seconds(qwen, p, QWEN_TE, overhead_s=3.0) == warm + qwen.cold_load_s
    if qwen.enabled:  # default quality must be admissible warm at n=1 (80 s budget)
        for name in OPERATIONS:
            assert estimate_seconds(qwen, replace(p, operation=name), QWEN_UNET, 3.0) <= 80


async def test_qwen_listed_only_with_all_files(gw):
    qwen = gw.app.state.registry.get("qwen-image")
    files = _qwen_loader_files(qwen)
    partial = dict(files)
    partial.pop(("VAELoader", "vae_name"))
    gw.fake.loader_files = partial
    r = await gw.client.get("/v1/models", headers=AUTH)
    assert "qwen-image" not in [m["id"] for m in r.json()["data"]]
    gw.fake.loader_files = files
    gw.app.state.catalog._available = None
    r = await gw.client.get("/v1/models", headers=AUTH)
    ids = [m["id"] for m in r.json()["data"]]
    assert ("qwen-image" in ids) == qwen.enabled


async def test_qwen_end_to_end(gw):
    qwen = gw.app.state.registry.get("qwen-image")
    if not qwen.enabled:
        pytest.skip("qwen-image is disabled in the registry")
    gw.fake.loader_files = _qwen_loader_files(qwen)
    r = await gw.generate(model="qwen-image", size="1792x1024")
    assert r.status_code == 200, r.text
    w, h = qwen.resolve_size("1792x1024")
    assert r.json()["size"] == f"{w}x{h}" and _decode(r.json()["data"][0]).size == (w, h)
    graph = gw.fake.prompt_bodies[-1]["prompt"]
    assert _nodes(graph, "UNETLoader")[0]["unet_name"] == QWEN_UNET

    r = await gw.edit(model="qwen-image", mask=_transparent_mask(64))
    assert r.status_code == 200, r.text
    assert _nodes(gw.fake.prompt_bodies[-1]["prompt"], "ImageCompositeMasked")
    r = await gw.variation(model="qwen-image")
    assert r.status_code == 200, r.text

    sent = len(gw.fake.prompt_bodies)
    r = await gw.generate(model="qwen-image", n=4)
    assert r.status_code == 400 and r.json()["error"]["code"] == "request_too_expensive"
    assert len(gw.fake.prompt_bodies) == sent and gw.queue.active == 0
