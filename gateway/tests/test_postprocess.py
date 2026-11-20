import io

import pytest
from PIL import Image, PngImagePlugin

from app.postprocess import reencode


def png_with_text(mode="RGB"):
    info = PngImagePlugin.PngInfo()
    info.add_text("workflow", '{"secret": 1}')
    info.add_text("prompt", "hidden", zip=False)
    info.add_itxt("extra", "intl")
    buf = io.BytesIO()
    Image.new(
        mode, (64, 64), (10, 20, 30)[: len(mode)] if mode != "RGBA" else (10, 20, 30, 255)
    ).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def test_fixture_has_text():
    raw = png_with_text()
    assert b"tEXt" in raw and b"iTXt" in raw


def test_png_strips_metadata():
    out = reencode(png_with_text(), "png")
    assert b"tEXt" not in out and b"iTXt" not in out and b"zTXt" not in out
    assert b"workflow" not in out and b"hidden" not in out
    im = Image.open(io.BytesIO(out))
    assert im.format == "PNG" and im.size == (64, 64) and not im.text


def test_png_rgba_preserved():
    out = reencode(png_with_text("RGBA"), "png")
    assert Image.open(io.BytesIO(out)).mode == "RGBA"


@pytest.mark.parametrize("fmt,pil", [("jpeg", "JPEG"), ("webp", "WEBP")])
def test_lossy(fmt, pil):
    out = reencode(png_with_text(), fmt, 70)
    im = Image.open(io.BytesIO(out))
    assert im.format == pil and im.size == (64, 64)
    assert b"workflow" not in out


def test_jpeg_from_rgba():
    out = reencode(png_with_text("RGBA"), "jpeg")
    assert Image.open(io.BytesIO(out)).mode == "RGB"


def test_compression_affects_size():
    buf = io.BytesIO()
    import random

    random.seed(1)
    Image.frombytes(
        "RGB", (128, 128), bytes(random.randrange(256) for _ in range(128 * 128 * 3))
    ).save(buf, "PNG")
    lo = reencode(buf.getvalue(), "jpeg", 10)
    hi = reencode(buf.getvalue(), "jpeg", 95)
    assert len(lo) < len(hi)


def test_bad_format():
    with pytest.raises(ValueError):
        reencode(png_with_text(), "gif")
