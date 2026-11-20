# comfyui-openai-api

An OpenAI-compatible Images API in front of a local ComfyUI:

| Endpoint | Body | ComfyUI workflow |
|---|---|---|
| `POST /v1/images/generations` | JSON | `<model>.json` (txt2img, e.g. `sdxl.json`) |
| `POST /v1/images/edits` | multipart: `image` (or one `image[]`), `prompt`, optional `mask` | `<model>_img2img.json`, with a mask `<model>_inpaint.json` |
| `POST /v1/images/variations` | multipart: `image` | `<model>_img2img.json` (empty prompt) |
| `GET /v1/models`, `GET /v1/models/{id}` | - | - |

 The official OpenAI SDKs talk to your own GPU box through a small FastAPI gateway.

## Models

| `model` | Operations | Native sizes (`size`) | OpenAI size aliases | Steps (`quality`) | Typical latency, RTX 3060 12 GB (warm / first request after another model) |
|---|---|---|---|---|---|
| `sdxl` (default) | generate, edit, masked edit, variation | 1024x1024, 1152x896, 896x1152, 1216x832, 832x1216, 1344x768, 768x1344, 1536x640, 640x1536 | 1792x1024 -> 1344x768, 1024x1792 -> 768x1344, 1536x1024 -> 1216x832, 1024x1536 -> 832x1216 | low 15, standard/medium/auto 25, hd/high 35 | 1024x1024 25 steps ~19 s (`n=4` ~41 s) / +~20 s |
| `sd15` | generate, edit, masked edit, variation | 512x512 (default), 640x640, 768x512, 512x768, 768x576, 576x768, 896x512, 512x896 | 1024x1024 and 256x256 -> 512x512, 1536x1024 -> 768x512, 1024x1536 -> 512x768, 1792x1024 -> 896x512, 1024x1792 -> 512x896 | low 15, standard/medium/auto 25, hd/high 35 | 512x512 25 steps ~3 s (`n=4` ~12 s), edits ~3.5 s / +~7 s |
| `flux-schnell` | generate, edit, masked edit, variation | same as `sdxl` | same as `sdxl` | always 4 | 1024x1024 ~13-14 s (`n=4` ~46-50 s), edits ~14 s / +~38 s (cold `n=4` ~80 s) |

- Files: `./scripts/download-models.sh sdxl sd15 flux-schnell` (`scripts/models.manifest`, sha256-verified).
  `sd15` is `v1-5-pruned-emaonly-fp16.safetensors` (2.1 GB); `flux-schnell` is the all-in-one
  `flux1-schnell-fp8.safetensors` checkpoint (17.2 GB: fp8 diffusion model, CLIP-L, T5-XXL fp8 and
  VAE, loaded with `CheckpointLoaderSimple`). `GET /v1/models` lists every enabled model whose files
  ComfyUI reports.
- **SD 1.5 sizes:** SD 1.5 was trained at 512 px and duplicates subjects above ~0.45 MP, so its
  buckets stay between 0.25 and 0.46 MP (multiples of 64). OpenAI's square sizes (1024x1024 and
  dall-e-2's 256x256) map to 512x512 and the wide/tall ones to the closest aspect ratio; the
  response `size` reports the real size. Upscale on the client if you need 1024 px.
- **Flux schnell:** distilled for 4 steps at cfg 1.0 (`euler`/`simple`, `EmptySD3LatentImage`), so
  `quality` does not change the step count, and `negative_prompt` and `cfg_scale` are **accepted but
  ignored** (cfg 1.0 disables the negative branch; rejecting them would break clients that always
  send them). An explicit `steps` still applies. No `FluxGuidance` node: schnell has no guidance
  embedding. The fp8 weights (~11.3 GB) plus T5 do not fit in 12 GB VRAM together; ComfyUI stages
  them with dynamic VRAM loading (peak 11.9 GB VRAM, ~19 GB host RAM for the ComfyUI process).
- **Switching models** is a full cold load each time: ComfyUI unloads the previous checkpoint itself
  (no `POST /free` needed), and the gateway adds the model's `cold_load_s` to the estimate when the
  checkpoint changes. A cold `flux-schnell` `n=4` request (~80 s measured) is only admitted on an
  idle queue because its warm estimate fits; it runs close to the 90 s deadline, so prefer `n<=2`
  right after a model switch.

## Setup (in this order)

1. `cp .env.example .env`, then set `API_KEYS` (e.g. `openssl rand -hex 32`), `PUBLIC_BASE_URL`
   (https, no trailing slash; only needed for `response_format=url`) and `HF_TOKEN`.
2. `./scripts/download-models.sh sdxl` (add `sd15 flux-schnell` for the other models) - **the one manual step, run it before the first
   `docker compose up`.** It creates every storage directory the stack mounts (otherwise Docker
   creates them root-owned and later script runs need `sudo`) and downloads the SDXL checkpoint.
   Re-running is a no-op. Run it without arguments to fetch every entry in
   `scripts/models.manifest` (includes the optional fp16-fix VAE). Every entry carries Hugging
   Face's LFS sha256 (`x-linked-etag`), so files are verified and re-downloaded on mismatch.
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
| `MAX_BODY_BYTES` | `1048576` | Max JSON request body (also enforced on chunked bodies) |
| `MAX_UPLOAD_BYTES` | `4194304` | Max size of each uploaded file (`image`, `mask`) on edits/variations; 413 while parsing |
| `MAX_MULTIPART_BYTES` | `0` | Max multipart body on edits/variations (also chunked); `0` = `2 * MAX_UPLOAD_BYTES + 64 KiB` |
| `MAX_INPUT_PIXELS` | `16777216` | Max `width * height` of an uploaded image (16 MP; decompression-bomb guard, checked from the header before decoding) |
| `MAX_CONCURRENT_PREPROCESS` | `2` | Uploads decoded/resized at once; further edits/variations get 429 `queue_full` immediately |
| `UPLOAD_WAIT_S` | `15` | Longest the GPU worker waits for a job's input upload to ComfyUI; then that job fails with 503 and the queue moves on |
| `QUEUE_MAX_DEPTH` | `8` | Admission cap (gateway jobs plus outside ComfyUI jobs) |
| `REQUEST_TIMEOUT_S` | `90` | End-to-end deadline; must be <= 95 and > `POSTPROCESS_RESERVE_S` |
| `POSTPROCESS_RESERVE_S` | `10` | Part of the deadline reserved for fetch, encode, base64 |
| `EST_JOB_OVERHEAD_S` | `3` | Fixed per-job overhead in wait estimates |
| `EST_EXTERNAL_JOB_S` | `30` | Assumed duration of each non-gateway ComfyUI job |
| `POLL_INTERVAL_S` | `0.5` | ComfyUI `/history` poll interval |
| `COMFY_QUEUE_POLL_S` | `2` | ComfyUI `/queue` poll interval for external load |
| `IMAGE_URL_TTL_S` | `3600` | Lifetime of hosted image URLs |
| `CLEANUP_COMFY_OUTPUTS` | `true` | Delete ComfyUI outputs/history after each job |
| `ORPHAN_MAX_AGE_S` | `3600` | Age after which stray files in the output `openai-api` subfolder are swept |
| `INPUT_ORPHAN_MAX_AGE_S` | `300` | Age after which stray uploaded inputs in the input `openai-api` subfolder are swept (swept at least this often) |
| `MODEL_CHECK_TTL_S` | `60` | Cache time for model availability |
| `MODEL_REFRESH_MIN_INTERVAL_S` | `5` | Min interval between forced availability refreshes |
| `LOG_LEVEL` | `info` | Log level |

Fixed by compose (not user-tunable): `COMFYUI_URL=http://comfyui:8188`, `IMAGE_STORE_DIR=/data/images`,
`COMFY_OUTPUT_DIR=/comfy-output`, `COMFY_OUTPUT_SUBFOLDER=openai-api`, `COMFY_INPUT_DIR=/comfy-input`,
`COMFY_INPUT_SUBFOLDER=openai-api`. Only `comfyui/storage-user/{output,input}/openai-api` are mounted
into the gateway, never the whole ComfyUI output/input directories.

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

### Edits and variations

```python
# img2img edit: the whole image is re-sampled with the prompt
r = client.images.edit(
    model="sdxl",
    image=open("fox.png", "rb"),
    prompt="the same fox wearing a red knitted scarf",
    response_format="b64_json",
    extra_body={"strength": 0.7, "seed": 7},   # optional extensions
)

# masked edit (inpainting): only the transparent area of the mask is repainted
r = client.images.edit(
    model="sdxl",
    image=open("fox.png", "rb"),
    mask=open("mask.png", "rb"),
    prompt="a small campfire in the snow",
)

# variations: img2img with an empty prompt, a random seed per request, distinct images per n
r = client.images.create_variation(model="sdxl", image=open("fox.png", "rb"), n=2)
```

- **Mask convention (OpenAI):** the mask is a PNG with an alpha channel; **fully transparent
  pixels mark the area to edit**, opaque pixels are kept. Example with Pillow:
  `m = Image.new("RGBA", img.size, (0, 0, 0, 255)); m.paste((0, 0, 0, 0), box); m.save("mask.png")`.
  A mask without alpha, or not a PNG, is 400 (`param=mask`). A mask of a different size is
  resized to the image first.
- **Mask feathering / blending:** the gateway Gaussian-blurs the mask's alpha (sigma = 2% of
  the short side, `MASK_FEATHER_FRACTION` in `gateway/app/uploads.py`: ~20 px at 1024, ~10 px
  at 512), so the soft edge straddles the mask boundary. The inpaint workflows denoise through
  that soft mask (SDXL/SD1.5 additionally use `DifferentialDiffusion`; flux-schnell uses a
  constant `ModelSamplingFlux` shift so its default strength 0.8 starts near full noise but
  still sees the surroundings) and then paste the decoded result onto the original through the
  same mask (`ImageCompositeMasked`). Pixels outside the feathered edge are returned
  bit-identical to the (resized) input; the repainted area fades into the scene instead of
  showing a hard box. A fully transparent mask edits everything.
- **Inputs:** `image` must be PNG, JPEG or WEBP, at most `MAX_UPLOAD_BYTES` (413 otherwise) and
  `MAX_INPUT_PIXELS`. It is EXIF-rotated, transparency is flattened onto gray, and it is
  cover-resized and center-cropped to the requested `size` (default: the model's default size,
  same whitelist and aliases as generations; the mask gets the identical transform). Only one
  image is accepted: more than one `image`/`image[]` is 400 `unsupported_value`. Without a mask
  the edit is plain img2img (the image's own alpha is not used as a mask).
- **Fields:** edits take `prompt` (required), `model`, `n`, `size`, `response_format`, `quality`,
  `output_format`, `output_compression`; variations take `model`, `n`, `size`, `response_format`.
  Extensions: `seed`, `strength` (denoise 0.05..1.0; defaults: edit 0.7, masked edit 1.0 (flux-schnell 0.8),
  variation 0.55), plus `negative_prompt`, `steps`, `cfg_scale` as for generations. `stream`,
  `partial_images > 0` and `background=transparent` are 400 like generations.
- Uploads go to ComfyUI as `input/openai-api/<job_id>_image.png` / `<job_id>_mask.png` and are
  deleted after every outcome (success, error, 504, client disconnect), and once more ~2 s later
  to catch an upload that landed after a cancelled request; the input orphan sweeper
  (`INPUT_ORPHAN_MAX_AGE_S`) is the backstop. Admission, 429/504, cancel and cleanup behave exactly like generations.
- The OpenAI SDK marks `create_variation` deprecated (it is a dall-e-2 endpoint); it still works.
- Edits/variations cost about the same as a generation of the same size and steps (ComfyUI runs all
  `steps` regardless of `strength`): SDXL 1024x1024, 25 steps ~17-22 s, `n=2` ~33-38 s on the 3060.

## Errors

All errors use the OpenAI envelope `{"error": {"message", "type", "param", "code"}}`.

| Condition | HTTP | code | Retry headers |
|---|---|---|---|
| Invalid request / schema | 400 | `invalid_value` (field-specific, `param` set) | |
| `stream`, `partial_images>0`, `background=transparent`, `url` disabled | 400 | `unsupported_value` | |
| Single job too expensive for the budget | 400 | `request_too_expensive` | |
| Missing/invalid key | 401 | `invalid_api_key` | `WWW-Authenticate: Bearer` |
| Unknown/unavailable model | 404 | `model_not_found` | |
| Body too large / uploaded file over `MAX_UPLOAD_BYTES` | 413 | `request_too_large` | |
| Invalid upload (not PNG/JPEG/WEBP, undecodable, too many pixels, mask without alpha) | 400 | `invalid_image_format` / `invalid_image` / `image_too_large` / `invalid_value` (`param` = `image` or `mask`) | |
| Model without the requested operation (edit/variation) | 400 | `unsupported_value` (`param=model`) | |
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
- Edits/variations: one image per request, `MAX_UPLOAD_BYTES` per file (default 4 MiB),
  `MAX_INPUT_PIXELS` (default 16 MP). Before the body is read, a request that cannot be admitted
  (queue full, or the current estimated wait alone exceeds the budget) gets 429 with
  `Retry-After`. At most `MAX_CONCURRENT_PREPROCESS` uploads are decoded at once (429
  `queue_full` otherwise); JPEGs are DCT-downscaled while decoding. The gateway container has
  `mem_limit: 2g`. A slow upload to ComfyUI holds the GPU queue for at most `UPLOAD_WAIT_S`
  (that request gets 503).
- Not supported yet: streaming / `partial_images`, transparent backgrounds, multi-image edits,
  image URLs/file ids instead of uploads.

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
- `sd15` (`est_seconds_per_step: 0.45`, `cold_load_s: 8`) and `flux-schnell` (`3.2`, `40`, edit
  `est_factor` 1.0) were calibrated the same way (Phase 3, `evidence/phase3/`); their warm estimates
  sit 1-20% above measured times (small `sd15` jobs more, because of the fixed 3 s job overhead).
- Re-calibrate after changing GPU, model or ComfyUI flags: run `tests/smoke/smoke_openai_sdk.py`
  (`tests/smoke/smoke_phase2.py SOURCE.png OUT_DIR` for SDXL edits/variations,
  `tests/smoke/smoke_phase3.py MODEL ALIAS_SIZE OUT_DIR` for any model) and compare
  `est_job_s` with `gpu_ms` in the gateway's per-job log lines. Edit/variation operations carry
  their own `est_factor` (0.9) and `est_extra_s` (0.5) in `registry.json` `operations`.
- Host port: the gateway listens on `127.0.0.1:${GATEWAY_PORT:-8001}` (8000 is often taken).
  For local testing without a tunnel, `PUBLIC_BASE_URL=http://127.0.0.1:8001` is accepted
  (plain http is allowed only for loopback hosts).
