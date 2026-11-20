"""Bearer API-key authentication dependency."""

from __future__ import annotations

import hmac

from fastapi import Request

from app.errors import invalid_api_key


def _extract_token(header: str | None) -> str | None:
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


async def require_api_key(request: Request) -> None:
    token = _extract_token(request.headers.get("authorization"))
    if token is None:
        raise invalid_api_key("You didn't provide an API key. Use 'Authorization: Bearer <key>'.")
    candidate = token.encode()
    matched: int | None = None
    for i, key in enumerate(request.app.state.settings.api_keys):
        # no short-circuit: compare against every configured key
        if hmac.compare_digest(candidate, key.encode()) and matched is None:
            matched = i
    if matched is None:
        raise invalid_api_key()
    # Position in API_KEYS (never the key itself): rate-limit bucket and metrics label.
    request.state.api_key_id = matched
