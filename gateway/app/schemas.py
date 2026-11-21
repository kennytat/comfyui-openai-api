"""Pydantic models for the OpenAI Images / Models contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.errors import invalid_value, unsupported_value

# Hard schema caps; the configured MAX_N / MAX_PROMPT_CHARS are enforced via check_limits().
HARD_MAX_PROMPT_CHARS = 32000
HARD_MAX_N = 10


class ImageOptions(BaseModel):
    """Options shared by generations, edits and variations (no prompt)."""

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    n: int | None = Field(default=None, ge=1, le=HARD_MAX_N)
    size: str | None = None
    response_format: Literal["url", "b64_json"] | None = None
    quality: Literal["standard", "hd", "low", "medium", "high", "auto"] | None = None
    output_format: Literal["png", "jpeg", "webp"] | None = None
    output_compression: int | None = Field(default=None, ge=0, le=100)
    background: Literal["auto", "opaque", "transparent"] | None = None
    stream: bool | None = None
    partial_images: int | None = Field(default=None, ge=0, le=3)
    style: str | None = None
    user: str | None = None
    moderation: str | None = None
    # extensions (SDK extra_body)
    seed: int | None = Field(default=None, ge=0, le=2**53)
    negative_prompt: str | None = Field(default=None, max_length=HARD_MAX_PROMPT_CHARS)
    steps: int | None = Field(default=None, ge=1, le=80)
    cfg_scale: float | None = Field(default=None, ge=1, le=20)

    @field_validator("background")
    @classmethod
    def _background(cls, v: str | None) -> str | None:
        if v == "transparent":
            raise unsupported_value("Transparent backgrounds are not supported.", "background")
        return v

    def check_limits(self, max_prompt_chars: int, max_n: int) -> None:
        """Enforce the operator-configured limits (raises OpenAIError)."""
        prompt = getattr(self, "prompt", None) or ""
        if len(prompt) > max_prompt_chars:
            raise invalid_value(f"Prompt must be at most {max_prompt_chars} characters.", "prompt")
        if self.negative_prompt is not None and len(self.negative_prompt) > max_prompt_chars:
            raise invalid_value(
                f"negative_prompt must be at most {max_prompt_chars} characters.",
                "negative_prompt",
            )
        if self.n is not None and self.n > max_n:
            raise invalid_value(f"n must be between 1 and {max_n}.", "n")


class ImageGenerationRequest(ImageOptions):
    prompt: str = Field(min_length=1, max_length=HARD_MAX_PROMPT_CHARS)


class ImageEditRequest(ImageGenerationRequest):
    """Multipart text fields of ``/v1/images/edits`` (files are handled separately)."""

    # extension: denoise strength (1.0 = ignore the input image)
    strength: float | None = Field(default=None, ge=0.05, le=1.0)


class ImageVariationRequest(ImageOptions):
    """Multipart text fields of ``/v1/images/variations`` (no streaming, like OpenAI)."""

    strength: float | None = Field(default=None, ge=0.05, le=1.0)

    @field_validator("stream")
    @classmethod
    def _stream(cls, v: bool | None) -> bool | None:
        if v:
            raise unsupported_value("Variations do not support streaming.", "stream")
        return v

    @field_validator("partial_images")
    @classmethod
    def _partial(cls, v: int | None) -> int | None:
        if v:
            raise unsupported_value("Variations do not support partial_images.", "partial_images")
        return v


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImagesResponse(BaseModel):
    created: int
    data: list[ImageData]
    output_format: str
    size: str
    quality: str


class Model(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "comfyui-openai-api"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Model]
