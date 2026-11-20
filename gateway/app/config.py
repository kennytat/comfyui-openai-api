"""Gateway configuration: environment-driven settings with fail-fast validation."""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MAX_TUNNEL_TIMEOUT_S = 95.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None, extra="ignore", case_sensitive=False, populate_by_name=True
    )

    api_keys_raw: str = Field(default="", validation_alias="API_KEYS")
    comfyui_url: str = "http://comfyui:8188"
    enable_url_format: bool = True
    public_base_url: str = ""
    default_model: str = "sdxl"
    default_response_format: str = "b64_json"
    max_n: int = 4
    max_prompt_chars: int = 4000
    # Phase 1 is JSON-only; Phase 2 multipart routes (edits/variations) raise it per-route.
    max_body_bytes: int = 1024 * 1024
    queue_max_depth: int = 8
    request_timeout_s: float = 90.0
    postprocess_reserve_s: float = 10.0
    est_job_overhead_s: float = 3.0
    est_external_job_s: float = 30.0
    poll_interval_s: float = 0.5
    comfy_queue_poll_s: float = 2.0
    image_store_dir: str = "/data/images"
    image_url_ttl_s: int = 3600
    comfy_output_dir: str = "/comfy-output"
    comfy_output_subfolder: str = "openai-api"
    cleanup_comfy_outputs: bool = True
    orphan_max_age_s: int = 3600
    model_check_ttl_s: float = 60.0
    model_refresh_min_interval_s: float = 5.0
    log_level: str = "info"

    @property
    def api_keys(self) -> list[str]:
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    @model_validator(mode="after")
    def _validate(self) -> Settings:
        if not self.api_keys:
            raise ValueError("API_KEYS must contain at least one non-empty key")
        if self.request_timeout_s > MAX_TUNNEL_TIMEOUT_S:
            raise ValueError(
                f"REQUEST_TIMEOUT_S must be <= {MAX_TUNNEL_TIMEOUT_S:g} "
                "(tunnel origin timeout margin)"
            )
        if self.request_timeout_s <= self.postprocess_reserve_s:
            raise ValueError("REQUEST_TIMEOUT_S must be greater than POSTPROCESS_RESERVE_S")
        if self.default_response_format not in ("b64_json", "url"):
            raise ValueError("DEFAULT_RESPONSE_FORMAT must be 'b64_json' or 'url'")
        self.public_base_url = self.public_base_url.strip()
        if self.enable_url_format:
            _check_public_base_url(self.public_base_url)
        return self

    @property
    def generation_budget_s(self) -> float:
        return self.request_timeout_s - self.postprocess_reserve_s


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_public_base_url(url: str) -> None:
    if not url:
        raise ValueError("PUBLIC_BASE_URL is required when ENABLE_URL_FORMAT=true")
    parsed = urlparse(url)
    if not parsed.hostname or parsed.scheme not in ("https", "http"):
        raise ValueError("PUBLIC_BASE_URL must be an absolute https:// URL with a host")
    # Plain http is only allowed for loopback hosts (local testing without a tunnel).
    if parsed.scheme == "http" and parsed.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            "PUBLIC_BASE_URL must use https:// "
            "(http:// is allowed only for localhost/127.0.0.1/::1)"
        )
    if url.endswith("/"):
        raise ValueError("PUBLIC_BASE_URL must not have a trailing slash")


def load_settings() -> Settings:
    return Settings()
