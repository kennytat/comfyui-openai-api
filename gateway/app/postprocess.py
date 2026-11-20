"""Image re-encoding that strips all metadata (workflow/prompt tEXt chunks, EXIF, ICC)."""

from __future__ import annotations

import io

from PIL import Image

DEFAULT_LOSSY_QUALITY = 85


def reencode(png_bytes: bytes, output_format: str, compression: int | None = None) -> bytes:
    """Decode and re-encode without any metadata. Blocking; call via asyncio.to_thread."""
    with Image.open(io.BytesIO(png_bytes)) as src:
        src.load()
        has_alpha = src.mode in ("RGBA", "LA") or "transparency" in src.info
        img = src.convert("RGBA" if has_alpha else "RGB")
    # rebuild from raw pixels so no info dict (text chunks, icc_profile, exif) can leak
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    out = io.BytesIO()
    fmt = output_format.lower()
    quality = DEFAULT_LOSSY_QUALITY if compression is None else compression
    if fmt == "png":
        clean.save(out, format="PNG", compress_level=6)
    elif fmt in ("jpeg", "jpg"):
        if clean.mode != "RGB":
            clean = clean.convert("RGB")
        clean.save(out, format="JPEG", quality=quality)
    elif fmt == "webp":
        clean.save(out, format="WEBP", quality=quality)
    else:
        raise ValueError(f"unsupported output_format: {output_format}")
    return out.getvalue()
