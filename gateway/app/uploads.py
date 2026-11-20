"""Multipart parsing with per-file limits and Pillow preprocessing of uploaded images.

``parse_multipart(request, max_file_bytes) -> (fields, files)``: streams the body through
Starlette's multipart parser, raising 413 ``request_too_large`` as soon as one file part
exceeds ``max_file_bytes`` (the whole body is capped earlier by the ASGI middleware at
``MAX_MULTIPART_BYTES``). Malformed multipart -> 400. ``files`` maps field name to the list
of uploaded byte strings (``image`` and ``image[]`` stay distinct; callers merge them).

``prepare_inputs(image, mask, width, height, max_pixels) -> (image_png, mask_png | None)``
(blocking; call via ``asyncio.to_thread``): validates the image (PNG/JPEG/WEBP only, the
pixel count is checked from the header before decoding and is authoritative; JPEGs are
draft-decoded at a reduced scale no smaller than twice the target), applies EXIF
orientation (image and mask), flattens transparency onto neutral gray, converts to RGB and
cover-resizes/center-crops to ``width x height``. The optional mask must be a PNG with an
alpha channel (OpenAI convention: fully transparent = area to edit); its alpha is resized
to the image's size if they differ, then cropped with the identical transform, then
feathered with a Gaussian blur (sigma = ``MASK_FEATHER_FRACTION`` of the short side; the
soft edge straddles the original mask boundary, fully opaque areas far from it stay
untouched), and saved as an RGBA PNG whose alpha ComfyUI's
``LoadImageMask(channel="alpha")`` turns into the inpaint mask (it inverts alpha, so
transparent -> 1 -> repainted). Output PNGs carry no metadata. Any Pillow failure while
transposing/converting/resizing is a 400
``invalid_image`` for the offending parameter.

``configure_pillow(max_pixels)``: set once at startup (process-global): Pillow's own
decompression-bomb limit (``Image.MAX_IMAGE_PIXELS``; it raises above twice that) and an
ignore filter for its warning band, since the explicit header check above already rejects
anything over ``MAX_INPUT_PIXELS``.

The multipart parser hooks Starlette's private ``MultiPartParser`` internals
(``on_part_begin``, ``on_part_data``, ``_current_part``, ``spool_max_size``); starlette is
pinned to a minor version and a test asserts these exist.
"""

from __future__ import annotations

import io
import warnings
from collections.abc import AsyncIterator, Callable

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import Request

from app.errors import OpenAIError, invalid_value

ALLOWED_FORMATS = ("PNG", "JPEG", "WEBP")
MAX_FIELD_BYTES = 128 * 1024  # text fields (prompt <= 32000 chars of UTF-8)
MAX_FILES = 4
MAX_FIELDS = 64
FLATTEN_COLOR = (128, 128, 128)
# Mask feathering: Gaussian sigma as a fraction of the target's short side (~20 px at 1024,
# ~10 px at 512). The inpaint workflows use the soft mask for differential denoising and to
# composite the result back onto the original, so the seam blends instead of showing a box.
MASK_FEATHER_FRACTION = 0.02


class _FileTooLarge(Exception):
    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field


class _LimitedMultiPartParser(MultiPartParser):
    """Starlette's parser, aborting as soon as one file part exceeds ``max_file_bytes``."""

    def __init__(self, headers: Headers, stream: AsyncIterator[bytes], max_file_bytes: int) -> None:
        super().__init__(
            headers,
            stream,  # type: ignore[arg-type]
            max_files=MAX_FILES,
            max_fields=MAX_FIELDS,
            max_part_size=MAX_FIELD_BYTES,
        )
        self.max_file_bytes = max_file_bytes
        self.spool_max_size = max_file_bytes  # keep uploads in memory (bounded)
        self._file_bytes = 0

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._file_bytes = 0

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if self._current_part.file is not None:
            self._file_bytes += end - start
            if self._file_bytes > self.max_file_bytes:
                raise _FileTooLarge(self._current_part.field_name)
        super().on_part_data(data, start, end)


async def parse_multipart(
    request: Request, max_file_bytes: int
) -> tuple[dict[str, str], dict[str, list[bytes]]]:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise invalid_value(
            "This endpoint expects a multipart/form-data body.", code="invalid_content_type"
        )
    parser = _LimitedMultiPartParser(request.headers, request.stream(), max_file_bytes)
    try:
        form = await parser.parse()
    except _FileTooLarge as exc:
        raise OpenAIError(
            413,
            f"Uploaded file '{exc.field}' exceeds the {max_file_bytes} byte limit.",
            code="request_too_large",
            param=exc.field.removesuffix("[]"),
        ) from exc
    except MultiPartException as exc:
        raise invalid_value(f"Invalid multipart body: {exc.message}") from exc
    fields: dict[str, str] = {}
    files: dict[str, list[bytes]] = {}
    try:
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                files.setdefault(key, []).append(await value.read())
            else:
                fields.setdefault(key, value)  # first value wins
    finally:
        await form.close()
    return fields, files


def configure_pillow(max_pixels: int) -> None:
    """Process-global Pillow limits; call once at startup (not per request: thread-unsafe)."""
    Image.MAX_IMAGE_PIXELS = max_pixels
    warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)


def _open(
    data: bytes,
    param: str,
    max_pixels: int,
    formats: tuple[str, ...],
    draft: tuple[int, int] | None = None,
) -> Image.Image:
    if not data:
        raise invalid_value(f"'{param}' is empty.", param, code="invalid_image")
    try:
        img = Image.open(io.BytesIO(data), formats=formats)
    except Image.DecompressionBombError as exc:
        raise invalid_value(
            f"'{param}' has too many pixels (limit {max_pixels}).", param, code="image_too_large"
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        names = ", ".join(formats)
        raise invalid_value(
            f"'{param}' must be a valid image in one of: {names}.",
            param,
            code="invalid_image_format",
        ) from exc
    w, h = img.size
    if w < 1 or h < 1 or w * h > max_pixels:
        img.close()
        raise invalid_value(
            f"'{param}' is {w}x{h}; at most {max_pixels} pixels are accepted.",
            param,
            code="image_too_large",
        )
    try:
        if draft is not None and img.format == "JPEG":
            img.draft("RGB", draft)  # DCT-scaled decode; never smaller than ``draft``
        img.load()
    except Exception as exc:  # truncated/corrupt data
        img.close()
        raise invalid_value(
            f"'{param}' could not be decoded.", param, code="invalid_image"
        ) from exc
    return img


def _has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA", "RGBa", "La") or "transparency" in img.info


def _png(img: Image.Image) -> bytes:
    clean = Image.frombytes(img.mode, img.size, img.tobytes())  # drop every info chunk
    out = io.BytesIO()
    clean.save(out, format="PNG", compress_level=1)
    return out.getvalue()


def _processing(param: str, step: Callable[[], Image.Image]) -> Image.Image:
    try:
        return step()
    except OpenAIError:
        raise
    except Exception as exc:  # Pillow failures on odd-but-decodable images
        raise invalid_value(
            f"'{param}' could not be processed.", param, code="invalid_image"
        ) from exc


def _flatten(src: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(src)
    if _has_alpha(oriented):
        rgba = oriented.convert("RGBA")
        flat = Image.new("RGB", rgba.size, FLATTEN_COLOR)
        flat.paste(rgba, mask=rgba.getchannel("A"))
        return flat
    return oriented.convert("RGB")


def prepare_inputs(
    image: bytes,
    mask: bytes | None,
    width: int,
    height: int,
    max_pixels: int,
) -> tuple[bytes, bytes | None]:
    """Validate and cover-resize the image (and mask) to ``width x height``. Blocking."""
    target = (width, height)
    side = 2 * max(width, height)  # square: EXIF rotation may swap the axes
    src = _open(image, "image", max_pixels, ALLOWED_FORMATS, draft=(side, side))
    with src:
        rgb = _processing("image", lambda: _flatten(src))
    source_size = rgb.size
    image_png = _png(
        _processing("image", lambda: ImageOps.fit(rgb, target, Image.Resampling.LANCZOS))
    )

    mask_png: bytes | None = None
    if mask is not None:
        msrc = _open(mask, "mask", max_pixels, ("PNG",))
        with msrc:
            if not _has_alpha(msrc):
                raise invalid_value(
                    "'mask' must be a PNG with an alpha channel "
                    "(fully transparent pixels mark the area to edit).",
                    "mask",
                )
            alpha = _processing(
                "mask", lambda: ImageOps.exif_transpose(msrc).convert("RGBA").getchannel("A")
            )

        def _fit_alpha() -> Image.Image:
            a = alpha
            if a.size != source_size:
                a = a.resize(source_size, Image.Resampling.BILINEAR)
            return ImageOps.fit(a, target, Image.Resampling.BILINEAR)

        alpha = _processing("mask", _fit_alpha)
        sigma = MASK_FEATHER_FRACTION * min(width, height)
        alpha = _processing("mask", lambda: alpha.filter(ImageFilter.GaussianBlur(sigma)))
        black = Image.new("L", target, 0)
        mask_png = _png(Image.merge("RGBA", (black, black, black, alpha)))
    return image_png, mask_png
