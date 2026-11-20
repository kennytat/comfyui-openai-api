"""``GET /images/{id}.{ext}``: unauthenticated hosted images with expiry enforced on read."""

from __future__ import annotations

import asyncio
import math
import re

from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.errors import not_found

router = APIRouter()

_NAME_RE = re.compile(r"^([0-9a-f]{32})\.(png|jpeg|webp)$")
MEDIA_TYPES = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}


@router.get("/images/{filename}")
async def get_image(filename: str, request: Request) -> Response:
    m = _NAME_RE.match(filename)
    if not m:
        raise not_found("Image not found or expired.")
    image_id, ext = m.groups()
    store = request.app.state.store
    found = store.get(image_id, ext)
    if found is None:
        raise not_found("Image not found or expired.")
    try:
        data = await asyncio.to_thread(found.path.read_bytes)
    except FileNotFoundError:
        raise not_found("Image not found or expired.") from None
    max_age = max(0, math.floor(found.remaining_s))
    return Response(
        content=data,
        media_type=MEDIA_TYPES[ext],
        headers={
            "Cache-Control": f"private, max-age={max_age}",
            "X-Content-Type-Options": "nosniff",
        },
    )
