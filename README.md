# comfyui-openai-api

An OpenAI-compatible Images API (`POST /v1/images/generations`, `GET /v1/models`) in front of a
local ComfyUI. The official OpenAI SDKs talk to your own GPU box through a small FastAPI gateway.

## Setup (in this order)

1. `cp .env.example .env`, then set `API_KEYS` (e.g. `openssl rand -hex 32`), `PUBLIC_BASE_URL`
   (https, no trailing slash; only needed for `response_format=url`) and `HF_TOKEN`.
2. `./scripts/download-models.sh sdxl` - **the one manual step, run it before the first
   `docker compose up`.** It creates every storage directory the stack mounts (otherwise Docker
   creates them root-owned and later script runs need `sudo`) and downloads the SDXL checkpoint.
   Re-running is a no-op. Run it without arguments to fetch every entry in
   `scripts/models.manifest` (includes the optional fp16-fix VAE). Entries with sha256 `-` are
   not hash-verified.
3. `docker compose up -d --build`. The first ComfyUI boot can take several minutes. The gateway
   starts immediately and answers `/healthz`; `/readyz` and `/v1/models` return 503 until ComfyUI
   is up. `docker compose ps` shows ComfyUI as healthy once booted (visibility only).
4. Optional tunnel: set `TUNNEL_TOKEN`, then `docker compose --profile tunnel up -d`.

Check: `curl -H "Authorization: Bearer $KEY" localhost:8001/v1/models` lists `sdxl`.

## Environment reference

Only the variables below reach the gateway container (explicit `environment:` in
`docker-compose.yml`, no `env_file`). `HF_TOKEN` is used by ComfyUI, `TUNNEL_TOKEN` only by
`cloudflared`.

| Variable | Default | Meaning |
|---|---|---|
| `API_KEYS` | required | Comma-separated Bearer keys for `/v1/*` |
| `PUBLIC_BASE_URL` | empty | Public base for image URLs. Must be absolute `https://host` (plain `http://` allowed only for `localhost`/`127.0.0.1`/`::1` for local testing), **no trailing slash**, when `ENABLE_URL_FORMAT=true`; optional otherwise |
| `ENABLE_URL_FORMAT` | `true` | Allow `response_format=url` |
| `DEFAULT_MODEL` | `sdxl` | Model used when `model` is omitted |
| `DEFAULT_RESPONSE_FORMAT` | `b64_json` | `b64_json` or `url` |
| `MAX_N` | `4` | Max images per request |
| `MAX_PROMPT_CHARS` | `4000` | Max prompt length |
| `MAX_BODY_BYTES` | `1048576` | Max request body (also enforced on chunked bodies). 1 MiB: Phase 1 is JSON-only; Phase 2 multipart edits/variations will raise it per-route |
| `QUEUE_MAX_DEPTH` | `8` | Admission cap (gateway jobs plus outside ComfyUI jobs) |
| `REQUEST_TIMEOUT_S` | `90` | End-to-end deadline; must be <= 95 and > `POSTPROCESS_RESERVE_S` |
| `POSTPROCESS_RESERVE_S` | `10` | Part of the deadline reserved for fetch, encode, base64 |
| `EST_JOB_OVERHEAD_S` | `3` | Fixed per-job overhead in wait estimates |
| `EST_EXTERNAL_JOB_S` | `30` | Assumed duration of each non-gateway ComfyUI job |
| `POLL_INTERVAL_S` | `0.5` | ComfyUI `/history` poll interval |
| `COMFY_QUEUE_POLL_S` | `2` | ComfyUI `/queue` poll interval for external load |
| `IMAGE_URL_TTL_S` | `3600` | Lifetime of hosted image URLs |
| `CLEANUP_COMFY_OUTPUTS` | `true` | Delete ComfyUI outputs/history after each job |
| `ORPHAN_MAX_AGE_S` | `3600` | Age after which stray files in the output subfolder are swept |
| `MODEL_CHECK_TTL_S` | `60` | Cache time for model availability |
| `MODEL_REFRESH_MIN_INTERVAL_S` | `5` | Min interval between forced availability refreshes |
| `LOG_LEVEL` | `info` | Log level |

Fixed by compose (not user-tunable): `COMFYUI_URL=http://comfyui:8188`, `IMAGE_STORE_DIR=/data/images`,
`COMFY_OUTPUT_DIR=/comfy-output`, `COMFY_OUTPUT_SUBFOLDER=openai-api`.

## SDK examples

Python:

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8001/v1", api_key="YOUR_KEY")
r = client.images.generate(
    model="sdxl",
    prompt="a red fox in the snow",
    size="1024x1024",
    n=1,
    response_format="b64_json",
    extra_body={"seed": 42, "negative_prompt": "blurry", "steps": 30, "cfg_scale": 6.5},
)
open("fox.png", "wb").write(base64.b64decode(r.data[0].b64_json))
```

JavaScript:

```js
import OpenAI from "openai";
import fs from "node:fs";

const client = new OpenAI({ baseURL: "http://localhost:8001/v1", apiKey: "YOUR_KEY" });
const r = await client.images.generate({
  model: "sdxl",
  prompt: "a red fox in the snow",
  size: "1024x1024",
  response_format: "b64_json",
  // extension fields are passed through as extra body properties
  seed: 42,
  negative_prompt: "blurry",
});
fs.writeFileSync("fox.png", Buffer.from(r.data[0].b64_json, "base64"));
```

Extensions (`seed`, `negative_prompt`, `steps` 1..80, `cfg_scale` 1..20) go in `extra_body` in
Python. `quality` maps to steps (`low`=15, `standard`/`medium`/`auto`=25, `hd`/`high`=35); an
explicit `steps` wins. OpenAI sizes `1792x1024`, `1024x1792`, `1536x1024`, `1024x1536` are mapped
to the nearest SDXL bucket and the response `size` reports the actual size generated.

## Errors

All errors use the OpenAI envelope `{"error": {"message", "type", "param", "code"}}`.

| Condition | HTTP | code | Retry headers |
|---|---|---|---|
| Invalid request / schema | 400 | `invalid_value` (field-specific, `param` set) | |
| `stream`, `partial_images>0`, `background=transparent`, `url` disabled | 400 | `unsupported_value` | |
| Single job too expensive for the budget | 400 | `request_too_expensive` | |
| Missing/invalid key | 401 | `invalid_api_key` | `WWW-Authenticate: Bearer` |
| Unknown/unavailable model | 404 | `model_not_found` | |
| Body too large | 413 | `request_too_large` | |
| Queue full | 429 | `queue_full` | `Retry-After` |
| Estimated wait exceeds budget | 429 | `estimated_wait_exceeded` | `Retry-After` |
| ComfyUI rejected the prompt (`node_errors`) | 502 | `upstream_error` | `x-should-retry: false` |
| ComfyUI execution error | 502 | `upstream_error` | |
| ComfyUI out of memory | 502 | `upstream_oom` | `x-should-retry: false` |
| Job interrupted upstream | 502 | `upstream_interrupted` | `x-should-retry: false` |
| Output missing in ComfyUI | 502 | `upstream_output_missing` | |
| ComfyUI unreachable | 503 | `upstream_unavailable` | |
| Deadline exceeded | 504 | `timeout` | `x-should-retry: false` |
| Unexpected error | 500 | `internal_error` | |

`x-should-retry: false` stops the SDK's automatic retries (it otherwise retries 408/409/429/5xx
twice), so one 90 s timeout does not become 270 s of GPU work. The SDK honours `Retry-After` only
up to 60 s; larger values fall back to its own backoff, so treat it as a hint.

## Limits

- One GPU job at a time (FIFO, one worker). Admission is wait-based: a request is rejected with
  429 when the estimated wait plus its own estimate exceeds `REQUEST_TIMEOUT_S - POSTPROCESS_RESERVE_S`.
  Estimates are hints; the 504 with a targeted ComfyUI cancel is the safety net.
- Hard end-to-end deadline `REQUEST_TIMEOUT_S` (default 90 s) from request receipt.
- `n` is limited by `MAX_N`; output PNGs have workflow/prompt metadata stripped.
- Hosted image URLs are unguessable and expire after `IMAGE_URL_TTL_S`.
- Not supported yet: streaming / `partial_images`, transparent backgrounds, edits, variations.

## Networking

The gateway binds `127.0.0.1:8001` (`GATEWAY_PORT`; 8000 is often taken) and ComfyUI binds `127.0.0.1:8188` and `127.0.0.1:8390`, so
ComfyUI is not reachable from the LAN. A tunnel connector must therefore run on the same host.
The bundled `cloudflared` service does (it reaches `gateway:8000` over the compose network); a
tunnel running on another machine cannot reach loopback-bound ports. Do not route ComfyUI through
the tunnel.

## Cloudflare Tunnel

1. In Cloudflare Zero Trust create a tunnel and copy its token into `TUNNEL_TOKEN` in `.env`.
2. Add a public hostname pointing to service `http://gateway:8000`.
3. Set `PUBLIC_BASE_URL=https://<that hostname>` (no trailing slash).
4. `docker compose --profile tunnel up -d`.

Cloudflare closes origin requests after about 100 s (HTTP 524), which a client sees as a non-OpenAI
error while the GPU keeps working. The gateway therefore enforces a 90 s deadline
(`REQUEST_TIMEOUT_S`, refused at startup if above 95) so it can answer with a proper OpenAI 504 and
cancel the ComfyUI job first.

## Performance notes (RTX 3060 12 GB, measured)

- ComfyUI runs with `CLI_ARGS: "--fast fp16_accumulation cublas_ops"`. Plain `--fast` also enables
  `autotune` (cuDNN benchmark), which re-benchmarks every new tensor shape: the first request at a
  new size or `n` took up to ~50 s extra (a first `n=2` request hit the 90 s timeout). Without it,
  new shapes cost nothing extra.
- Measured SDXL 1024x1024 (warm): ~11 s for 15 steps, ~19 s for 25 steps, `n=2` ~22 s, `n=4` ~41 s;
  first request after ComfyUI start adds ~18 s of model load.
- `gateway/workflows/registry.json` is calibrated from these numbers (`est_seconds_per_step: 0.75`,
  `cold_load_s: 20`), so estimates sit at or slightly above real times. With the 80 s generation
  budget, `n=4` at `quality=hd` (35 steps) is rejected up front with 400 `request_too_expensive`,
  and only one heavy `n=4` job is queued at a time (others get an immediate 429).
- Re-calibrate after changing GPU, model or ComfyUI flags: run `tests/smoke/smoke_openai_sdk.py`
  and compare `est_job_s` with `gpu_ms` in its per-job output.
- Host port: the gateway listens on `127.0.0.1:${GATEWAY_PORT:-8001}` (8000 is often taken).
  For local testing without a tunnel, `PUBLIC_BASE_URL=http://127.0.0.1:8001` is accepted
  (plain http is allowed only for loopback hosts).
