import pytest
from pydantic import ValidationError

from app.errors import OpenAIError
from app.schemas import ImageData, ImageGenerationRequest, ImagesResponse, Model, ModelList


def req(**kw):
    return ImageGenerationRequest.model_validate({"prompt": "a cat", **kw})


def test_minimal_and_extra_ignored():
    r = req(foo="bar", style="vivid", user="u", moderation="low")
    assert r.prompt == "a cat" and r.n is None and not hasattr(r, "foo")


def test_extensions():
    r = req(seed=5, negative_prompt="x", steps=80, cfg_scale=20)
    assert (r.seed, r.steps, r.cfg_scale) == (5, 80, 20)


@pytest.mark.parametrize(
    "kw",
    [
        {"prompt": ""},
        {"n": 0},
        {"n": 11},
        {"response_format": "xml"},
        {"quality": "ultra"},
        {"output_format": "gif"},
        {"output_compression": 101},
        {"output_compression": -1},
        {"steps": 0},
        {"steps": 81},
        {"cfg_scale": 0.5},
        {"cfg_scale": 21},
        {"negative_prompt": "x" * 32001},
    ],
)
def test_validation_errors(kw):
    with pytest.raises(ValidationError):
        req(**kw)


@pytest.mark.parametrize(
    "kw,param",
    [
        ({"background": "transparent"}, "background"),
        ({"stream": True}, "stream"),
        ({"partial_images": 2}, "partial_images"),
    ],
)
def test_unsupported(kw, param):
    with pytest.raises(OpenAIError) as ei:
        req(**kw)
    assert ei.value.code == "unsupported_value" and ei.value.param == param
    assert ei.value.status == 400


def test_accepted_noops():
    r = req(
        background="opaque",
        stream=False,
        partial_images=0,
        size="auto",
        quality="hd",
        output_format="webp",
        response_format="url",
    )
    assert r.size == "auto"


def test_size_kept_string():
    assert req(size="1024x1792").size == "1024x1792"


def test_check_limits():
    r = req(prompt="x" * 11, n=5)
    with pytest.raises(OpenAIError) as ei:
        r.check_limits(10, 8)
    assert ei.value.param == "prompt"
    with pytest.raises(OpenAIError) as ei:
        r.check_limits(100, 4)
    assert ei.value.param == "n"
    r.check_limits(100, 5)


def test_check_limits_negative_prompt():
    r = req(negative_prompt="x" * 11)
    with pytest.raises(OpenAIError) as ei:
        r.check_limits(10, 4)
    assert ei.value.param == "negative_prompt" and ei.value.code == "invalid_value"
    assert ei.value.status == 400
    r.check_limits(11, 4)


def test_response_models():
    resp = ImagesResponse(
        created=1,
        data=[ImageData(b64_json="x")],
        output_format="png",
        size="768x1344",
        quality="medium",
    )
    assert resp.model_dump()["data"][0]["revised_prompt"] is None
    m = ModelList(data=[Model(id="sdxl", created=1)]).model_dump()
    assert m["object"] == "list" and m["data"][0]["object"] == "model"
    assert m["data"][0]["owned_by"] == "comfyui-openai-api"
