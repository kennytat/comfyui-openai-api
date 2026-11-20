"""``GET /metrics``: Prometheus text exposition (Bearer API key; see ``app.metrics``)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response

from app.auth import require_api_key
from app.metrics import CONTENT_TYPE

router = APIRouter(dependencies=[Depends(require_api_key)])


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    return Response(request.app.state.metrics.render(), media_type=CONTENT_TYPE)
