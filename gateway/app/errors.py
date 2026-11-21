"""OpenAI-style error envelope, exception type, constructors and handlers."""

from __future__ import annotations

import logging
import math
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger("gateway.errors")

INVALID_REQUEST = "invalid_request_error"
RATE_LIMIT = "rate_limit_error"
SERVER_ERROR = "server_error"

NO_RETRY = {"x-should-retry": "false"}


class OpenAIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        type: str = INVALID_REQUEST,
        code: str | None = None,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.type = type
        self.code = code
        self.param = param
        self.headers = dict(headers or {})

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.type,
                "param": self.param,
                "code": self.code,
            }
        }

    def response(self) -> JSONResponse:
        return JSONResponse(self.body(), status_code=self.status, headers=self.headers)


# --- constructors, one per row of the plan's 3.3 table -------------------------------------


def invalid_value(
    message: str, param: str | None = None, code: str = "invalid_value"
) -> OpenAIError:
    return OpenAIError(400, message, INVALID_REQUEST, code, param)


def unsupported_value(message: str, param: str | None = None) -> OpenAIError:
    return OpenAIError(400, message, INVALID_REQUEST, "unsupported_value", param)


def request_too_expensive(message: str, param: str | None = None) -> OpenAIError:
    return OpenAIError(400, message, INVALID_REQUEST, "request_too_expensive", param)


def invalid_api_key(message: str = "Incorrect API key provided.") -> OpenAIError:
    return OpenAIError(
        401, message, INVALID_REQUEST, "invalid_api_key", headers={"WWW-Authenticate": "Bearer"}
    )


def model_not_found(model: str) -> OpenAIError:
    return OpenAIError(
        404,
        f"The model '{model}' does not exist or is not available.",
        INVALID_REQUEST,
        "model_not_found",
        "model",
    )


def not_found(message: str = "Not found.") -> OpenAIError:
    return OpenAIError(404, message, INVALID_REQUEST, "not_found")


def method_not_allowed(message: str = "Method not allowed.") -> OpenAIError:
    return OpenAIError(405, message, INVALID_REQUEST, "method_not_allowed")


def request_too_large(limit_bytes: int | None = None) -> OpenAIError:
    suffix = f" (limit {limit_bytes} bytes)" if limit_bytes is not None else ""
    return OpenAIError(
        413, f"Request body too large{suffix}.", INVALID_REQUEST, "request_too_large"
    )


def _retry_after(seconds: float) -> dict[str, str]:
    return {"Retry-After": str(max(1, math.ceil(seconds)))}


def queue_full(
    retry_after_s: float, message: str = "The generation queue is full. Try again later."
) -> OpenAIError:
    return OpenAIError(
        429,
        message,
        RATE_LIMIT,
        "queue_full",
        headers=_retry_after(retry_after_s),
    )


def rate_limit_exceeded(retry_after_s: float) -> OpenAIError:
    return OpenAIError(
        429,
        "Rate limit reached for this API key. Try again later.",
        RATE_LIMIT,
        "rate_limit_exceeded",
        headers=_retry_after(retry_after_s),
    )


def estimated_wait_exceeded(retry_after_s: float) -> OpenAIError:
    return OpenAIError(
        429,
        "Estimated wait time exceeds the request time budget. Try again later.",
        RATE_LIMIT,
        "estimated_wait_exceeded",
        headers=_retry_after(retry_after_s),
    )


def upstream_error(
    message: str = "The image backend failed.", *, retryable: bool = True
) -> OpenAIError:
    """502. Pass retryable=False for deterministic failures (ComfyUI node_errors)."""
    return OpenAIError(
        502, message, SERVER_ERROR, "upstream_error", headers=None if retryable else NO_RETRY
    )


def upstream_oom(message: str = "The image backend ran out of memory.") -> OpenAIError:
    return OpenAIError(502, message, SERVER_ERROR, "upstream_oom", headers=NO_RETRY)


def upstream_interrupted(message: str = "The generation was interrupted upstream.") -> OpenAIError:
    return OpenAIError(502, message, SERVER_ERROR, "upstream_interrupted", headers=NO_RETRY)


def upstream_output_missing(
    message: str = "The generated image could not be retrieved.",
) -> OpenAIError:
    return OpenAIError(502, message, SERVER_ERROR, "upstream_output_missing")


def upstream_unavailable(message: str = "The image backend is unavailable.") -> OpenAIError:
    return OpenAIError(503, message, SERVER_ERROR, "upstream_unavailable")


def timeout(message: str = "The request timed out before the image was ready.") -> OpenAIError:
    return OpenAIError(504, message, SERVER_ERROR, "timeout", headers=NO_RETRY)


def internal_error(
    message: str = "The server had an error while processing your request.",
) -> OpenAIError:
    return OpenAIError(500, message, SERVER_ERROR, "internal_error")


# --- handlers ------------------------------------------------------------------------------


def _param_from_loc(loc: tuple[Any, ...] | list[Any]) -> str | None:
    parts = [str(p) for p in loc if p not in ("body", "query", "path", "header", "cookie")]
    return ".".join(parts) or None


def _respond(request: Request, err: OpenAIError) -> JSONResponse:
    metrics = getattr(request.app.state, "metrics", None)
    if metrics is not None:
        metrics.error(err.status, err.code)
    return err.response()


async def _openai_error_handler(request: Request, exc: OpenAIError) -> JSONResponse:
    return _respond(request, exc)


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    first = errors[0] if errors else {}
    etype = first.get("type", "")
    param = _param_from_loc(first.get("loc", ()))
    if etype in ("json_invalid", "model_attributes_type") or not errors:
        message, code = "We could not parse the JSON body of your request.", "invalid_json"
    elif etype == "missing":
        message = f"Missing required parameter: '{param}'."
        code = "missing_required_parameter"
    else:
        msg = first.get("msg", "Invalid value")
        message = f"Invalid '{param}': {msg}." if param else f"Invalid request: {msg}."
        code = "invalid_value"
    return _respond(request, OpenAIError(400, message, INVALID_REQUEST, code, param))


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    status = exc.status_code
    headers = dict(exc.headers or {})
    detail = exc.detail if isinstance(exc.detail, str) else "Error"
    if status == 404:
        code, message = "not_found", "Invalid URL or resource not found."
    elif status == 405:
        code, message = "method_not_allowed", "Method not allowed for this endpoint."
    else:
        code, message = "http_error", detail
    etype = SERVER_ERROR if status >= 500 else INVALID_REQUEST
    return _respond(request, OpenAIError(status, message, etype, code, headers=headers))


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    return _respond(request, internal_error())


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(OpenAIError, _openai_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, _unhandled_handler)
