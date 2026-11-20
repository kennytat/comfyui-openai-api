import pytest
from pydantic import ValidationError

from app.config import Settings

ENV_NAMES = [
    "API_KEYS",
    "COMFYUI_URL",
    "ENABLE_URL_FORMAT",
    "PUBLIC_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_RESPONSE_FORMAT",
    "MAX_N",
    "MAX_PROMPT_CHARS",
    "MAX_BODY_BYTES",
    "MAX_UPLOAD_BYTES",
    "MAX_MULTIPART_BYTES",
    "MAX_INPUT_PIXELS",
    "COMFY_INPUT_DIR",
    "COMFY_INPUT_SUBFOLDER",
    "QUEUE_MAX_DEPTH",
    "REQUEST_TIMEOUT_S",
    "POSTPROCESS_RESERVE_S",
    "EST_JOB_OVERHEAD_S",
    "EST_EXTERNAL_JOB_S",
    "POLL_INTERVAL_S",
    "COMFY_QUEUE_POLL_S",
    "IMAGE_STORE_DIR",
    "IMAGE_URL_TTL_S",
    "COMFY_OUTPUT_DIR",
    "COMFY_OUTPUT_SUBFOLDER",
    "CLEANUP_COMFY_OUTPUTS",
    "ORPHAN_MAX_AGE_S",
    "MODEL_CHECK_TTL_S",
    "MODEL_REFRESH_MIN_INTERVAL_S",
    "LOG_LEVEL",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def make(**env) -> Settings:
    base = {"API_KEYS": "k1", "PUBLIC_BASE_URL": "https://img.example.com"}
    base.update(env)
    with pytest.MonkeyPatch.context() as mp:
        for k, v in base.items():
            mp.setenv(k, v)
        return Settings(_env_file=None)


def test_defaults():
    s = make()
    assert s.comfyui_url == "http://comfyui:8188"
    assert s.enable_url_format is True
    assert s.default_model == "sdxl"
    assert s.default_response_format == "b64_json"
    assert s.max_n == 4
    assert s.max_prompt_chars == 4000
    assert s.max_body_bytes == 1024 * 1024
    assert s.queue_max_depth == 8
    assert s.request_timeout_s == 90
    assert s.postprocess_reserve_s == 10
    assert s.est_job_overhead_s == 3
    assert s.est_external_job_s == 30
    assert s.poll_interval_s == 0.5
    assert s.comfy_queue_poll_s == 2
    assert s.image_store_dir == "/data/images"
    assert s.image_url_ttl_s == 3600
    assert s.comfy_output_dir == "/comfy-output"
    assert s.comfy_output_subfolder == "openai-api"
    assert s.cleanup_comfy_outputs is True
    assert s.orphan_max_age_s == 3600
    assert s.model_check_ttl_s == 60
    assert s.model_refresh_min_interval_s == 5
    assert s.log_level == "info"
    assert s.generation_budget_s == 80


def test_loads_from_environment(monkeypatch):
    monkeypatch.setenv("API_KEYS", "a, b ,,c")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://x.example.org")
    monkeypatch.setenv("MAX_N", "2")
    s = Settings(_env_file=None)
    assert s.api_keys == ["a", "b", "c"]
    assert s.max_n == 2


@pytest.mark.parametrize("keys", ["", "   ", " , ,"])
def test_empty_api_keys_rejected(keys):
    with pytest.raises(ValidationError, match="API_KEYS"):
        make(API_KEYS=keys)


def test_missing_api_keys_rejected(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://x.example.org")
    with pytest.raises(ValidationError, match="API_KEYS"):
        Settings(_env_file=None)


def test_timeout_above_tunnel_margin_rejected():
    with pytest.raises(ValidationError, match="REQUEST_TIMEOUT_S"):
        make(REQUEST_TIMEOUT_S="96")


def test_timeout_at_tunnel_margin_ok():
    assert make(REQUEST_TIMEOUT_S="95").request_timeout_s == 95


def test_timeout_not_above_reserve_rejected():
    with pytest.raises(ValidationError, match="POSTPROCESS_RESERVE_S"):
        make(REQUEST_TIMEOUT_S="10", POSTPROCESS_RESERVE_S="10")


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://img.example.com",
        "https://img.example.com/",
        "/images",
        "img.example.com",
        "https://",
    ],
)
def test_bad_public_base_url_rejected_when_url_format_enabled(url):
    with pytest.raises(ValidationError, match="PUBLIC_BASE_URL"):
        make(PUBLIC_BASE_URL=url)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8001", "http://localhost:8001", "http://[::1]:8001"]
)
def test_http_allowed_for_loopback_public_base_url(url):
    assert make(PUBLIC_BASE_URL=url).public_base_url == url


@pytest.mark.parametrize("url", ["http://192.168.1.10:8001", "http://127.0.0.1:8001/"])
def test_http_non_loopback_or_trailing_slash_rejected(url):
    with pytest.raises(ValidationError, match="PUBLIC_BASE_URL"):
        make(PUBLIC_BASE_URL=url)


def test_public_base_url_with_path_ok():
    assert make(PUBLIC_BASE_URL="https://x.example.com/api").public_base_url.endswith("/api")


@pytest.mark.parametrize("url", ["", "http://x", "https://x/"])
def test_public_base_url_optional_when_url_format_disabled(url):
    s = make(ENABLE_URL_FORMAT="false", PUBLIC_BASE_URL=url)
    assert s.enable_url_format is False


def test_bad_default_response_format_rejected():
    with pytest.raises(ValidationError, match="DEFAULT_RESPONSE_FORMAT"):
        make(DEFAULT_RESPONSE_FORMAT="png")


def test_upload_limits_defaults_and_derived_multipart():
    s = make()
    assert s.max_upload_bytes == 4 * 1024 * 1024
    assert s.max_multipart_bytes == 2 * 4 * 1024 * 1024 + 64 * 1024
    assert s.max_input_pixels == 16_777_216
    assert s.max_concurrent_preprocess == 2 and s.upload_wait_s == 15.0
    assert s.input_orphan_max_age_s == 300 and s.orphan_max_age_s == 3600
    assert s.comfy_input_dir == "/comfy-input" and s.comfy_input_subfolder == "openai-api"
    s = make(MAX_UPLOAD_BYTES="1000", MAX_MULTIPART_BYTES="0")
    assert s.max_multipart_bytes == 2000 + 64 * 1024
    assert make(MAX_MULTIPART_BYTES="5000000").max_multipart_bytes == 5_000_000


@pytest.mark.parametrize(
    "env",
    [
        {"MAX_UPLOAD_BYTES": "0"},
        {"MAX_INPUT_PIXELS": "-1"},
        {"MAX_UPLOAD_BYTES": "5000", "MAX_MULTIPART_BYTES": "4000"},
        {"MAX_CONCURRENT_PREPROCESS": "0"},
        {"UPLOAD_WAIT_S": "0"},
        {"INPUT_ORPHAN_MAX_AGE_S": "0"},
    ],
)
def test_upload_limit_validation(env):
    with pytest.raises(ValidationError):
        make(**env)
