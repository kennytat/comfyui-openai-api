"""Shared test helpers: a stateful fake ComfyUI (respx), fake clock, PNG builder, settings.

Imported as ``tests.helpers`` (``pythonpath = ["."]``); the pytest fixtures wrapping these
live in ``conftest.py``.

``FakeComfy`` emulates ``/prompt``, ``/history``, ``/view``, ``/queue``, ``/interrupt``,
``/upload/image`` (multipart; files land in the fake input mount, ``input_dir``),
``/object_info/CheckpointLoaderSimple`` and ``/system_stats`` behind a respx router that is
injected as the ``ComfyClient`` transport (the ASGI transport used to call the gateway is
never intercepted). It is stateful: every accepted prompt id is listed in ``/queue``
(``queue_running`` for the head, ``queue_pending`` otherwise) with ``extra_data.client_id``
set to the submitting client id until it completes, so held jobs are never "lost". The
head prompt completes on its next ``/history`` poll unless ``hold`` (an ``asyncio.Event``)
is set and not yet triggered. Outputs are 64x64-or-requested-size PNGs carrying a ``tEXt``
"workflow" chunk (like ``SaveImage``), also written to the fake output mount on disk.
Output size and count come from ``EmptyLatentImage``/``EmptySD3LatentImage`` (txt2img)
or, for img2img graphs, from the uploaded ``LoadImage`` file and ``RepeatLatentBatch.amount``.
``extra_checkpoints`` adds file names to the listed ``CheckpointLoaderSimple`` options;
``loader_files[(loader, input)]`` lists files for any other loader (``/object_info/<loader>``,
legacy combo format), e.g. ``UNETLoader.unet_name`` for split-file models.
"""

from __future__ import annotations

import asyncio
import dataclasses
import email.parser
import email.policy
import io
import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import respx
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from app.config import Settings

FIXTURES = Path(__file__).resolve().parent / "fixtures"
COMFY = "http://comfy.test"
KEY = "test-key-123"
AUTH = {"Authorization": f"Bearer {KEY}"}
PUBLIC = "https://images.example.test"
CKPT = "sd_xl_base_1.0.safetensors"


def make_png(
    width: int = 64, height: int = 64, color: tuple[int, int, int] = (200, 30, 30)
) -> bytes:
    """PNG with a tEXt 'workflow' chunk, like ComfyUI's SaveImage output."""
    info = PngInfo()
    info.add_text("workflow", json.dumps({"secret": "workflow graph"}))
    info.add_text("prompt", json.dumps({"3": {"inputs": {"text": "secret prompt"}}}))
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


class FakeClock:
    def __init__(self, start: float = 1_763_400_000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _fixture_history(name: str, pid: str) -> dict[str, Any]:
    raw = (FIXTURES / name).read_text().replace("11111111-1111-4111-8111-111111111111", pid)
    return json.loads(raw)[pid]


def _nodes(graph: dict[str, Any], class_type: str) -> list[dict[str, Any]]:
    return [n["inputs"] for n in graph.values() if n.get("class_type") == class_type]


def parse_form(request: httpx.Request) -> dict[str, tuple[str | None, bytes]]:
    """Decode a multipart/form-data request body: name -> (filename, payload)."""
    head = f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
    msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(head + request.content)
    out: dict[str, tuple[str | None, bytes]] = {}
    for part in msg.iter_parts():
        name = part.get_param("name", header="content-disposition")
        out[str(name)] = (part.get_filename(), part.get_payload(decode=True) or b"")
    return out


class FakeComfy:
    def __init__(self, output_dir: Path, input_dir: Path | None = None) -> None:
        self.output_dir = output_dir
        self.input_dir = input_dir or output_dir.parent / "comfy-input"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.uploads: list[dict[str, Any]] = []  # every /upload/image call
        self.upload_mode = "ok"  # ok | 500 | rename
        self.upload_gate: asyncio.Event | None = None
        self.router = respx.Router(base_url=COMFY, assert_all_called=False)
        self.router.route().mock(side_effect=self._handle)
        self.transport = httpx.MockTransport(self.router.async_handler)
        self.down = False
        self.hold: asyncio.Event | None = None
        self.submit_gate: asyncio.Event | None = None
        self.object_info_mode = "legacy"  # legacy | combo | absent | error
        self.view_mode = "ok"  # ok | 404 | drop
        self.view_gate: asyncio.Event | None = None  # /view blocks until set
        self.object_info_gate: asyncio.Event | None = None
        self.prompt_reply_gate: asyncio.Event | None = None  # accepted+queued, reply held
        self.prompt_reply_drop = False  # one-shot: accept+queue, then drop the connection
        self.node_errors = False
        self.outcomes: list[str] = []  # per submitted prompt: success|error|oom|interrupted|lost
        self.queue: list[str] = []
        self.prompts: dict[str, dict[str, Any]] = {}
        self.history: dict[str, dict[str, Any]] = {}
        self.files: dict[str, bytes] = {}
        self.external: list[list[Any]] = []
        self.submitted = asyncio.Event()
        self.prompt_bodies: list[dict[str, Any]] = []
        self.interrupts: list[Any] = []
        self.queue_deletes: list[str] = []
        self.history_deletes: list[str] = []
        self.view_calls = 0
        self.object_info_calls = 0
        self.extra_checkpoints: list[str] = []  # appended to the listed checkpoint files
        self.loader_files: dict[tuple[str, str], list[str]] = {}  # other loaders' files
        self.requests: list[tuple[str, str]] = []

    # --------------------------------------------------------------- helpers

    def add_external(self, client_id: str | None = "someone-else", pid: str | None = None) -> str:
        pid = pid or str(uuid.uuid4())
        extra = {"client_id": client_id} if client_id is not None else {}
        self.external.append([100 + len(self.external), pid, {}, extra, ["9"]])
        return pid

    @property
    def prompt_count(self) -> int:
        return len(self.prompt_bodies)

    def _entry(self, pid: str) -> list[Any]:
        p = self.prompts[pid]
        return [p["number"], pid, p["graph"], {"client_id": p["client_id"]}, ["9"]]

    def _complete(self, pid: str) -> None:
        p = self.prompts[pid]
        outcome = p["outcome"]
        self.queue.remove(pid)
        if outcome == "lost":
            return
        if outcome in ("error", "oom", "interrupted"):
            name = {"error": "history_error.json", "oom": "history_oom.json"}.get(
                outcome, "history_interrupted.json"
            )
            self.history[pid] = _fixture_history(name, pid)
            return
        graph = p["graph"]
        latents = _nodes(graph, "EmptyLatentImage") or _nodes(graph, "EmptySD3LatentImage")
        if latents:
            w, h, n = latents[0]["width"], latents[0]["height"], latents[0]["batch_size"]
        else:  # img2img: size of the uploaded LoadImage input, n from RepeatLatentBatch
            name = _nodes(graph, "LoadImage")[0]["image"]
            with Image.open(self.input_dir / name.rpartition("/")[2]) as src:
                w, h = src.size
            n = _nodes(graph, "RepeatLatentBatch")[0]["amount"]
        seed = _nodes(graph, "KSampler")[0]["seed"]
        prefix = _nodes(graph, "SaveImage")[0]["filename_prefix"]
        subfolder, _, base = prefix.rpartition("/")
        images = []
        for i in range(n):
            fname = f"{base}_{i + 1:05d}_.png"
            color = ((seed + 40 * i) % 256, (seed // 7 + 90 * i) % 256, 77)
            data = make_png(w, h, color)
            self.files[fname] = data
            (self.output_dir / fname).write_bytes(data)
            images.append({"filename": fname, "subfolder": subfolder, "type": "output"})
        self.history[pid] = {
            "prompt": self._entry(pid),
            "outputs": {"9": {"images": images}},
            "status": {"status_str": "success", "completed": True, "messages": []},
        }

    def _interrupt(self, pid: str) -> None:
        self.queue.remove(pid)
        self.history[pid] = _fixture_history("history_interrupted.json", pid)

    # --------------------------------------------------------------- dispatcher

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.requests.append((method, path))
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if method == "POST" and path == "/upload/image":
            return await self._upload(request)
        body = json.loads(request.content) if request.content else {}
        if method == "POST" and path == "/prompt":
            return await self._post_prompt(body)
        if method == "GET" and path.startswith("/history/"):
            return self._get_history(path.removeprefix("/history/"))
        if method == "POST" and path == "/history":
            for pid in body.get("delete", []):
                self.history_deletes.append(pid)
                self.history.pop(pid, None)
            return httpx.Response(200)
        if method == "GET" and path == "/queue":
            running = [self._entry(self.queue[0])] if self.queue else []
            pending = [self._entry(p) for p in self.queue[1:]] + list(self.external)
            return httpx.Response(200, json={"queue_running": running, "queue_pending": pending})
        if method == "POST" and path == "/queue":
            for pid in body.get("delete", []):
                self.queue_deletes.append(pid)
                if pid in self.queue[1:]:
                    self.queue.remove(pid)
                self.external = [e for e in self.external if e[1] != pid]
            return httpx.Response(200)
        if method == "POST" and path == "/interrupt":
            pid = body.get("prompt_id")
            self.interrupts.append(pid)
            if self.queue and (pid is None or self.queue[0] == pid):
                self._interrupt(self.queue[0])
            return httpx.Response(200)
        if method == "GET" and path == "/view":
            return await self._view(request)
        if method == "GET" and path == "/object_info/CheckpointLoaderSimple":
            return await self._object_info()
        if method == "GET" and path.startswith("/object_info/"):
            loader = path.removeprefix("/object_info/")
            required = {i: [list(f)] for (lo, i), f in self.loader_files.items() if lo == loader}
            if required:
                return httpx.Response(200, json={loader: {"input": {"required": required}}})
        if method == "GET" and path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "posix"}, "devices": []})
        return httpx.Response(404, json={"error": "not found"})

    async def _post_prompt(self, body: dict[str, Any]) -> httpx.Response:
        self.prompt_bodies.append(body)
        if self.submit_gate is not None:
            await self.submit_gate.wait()
        if self.node_errors:
            return httpx.Response(
                400, json=json.loads((FIXTURES / "prompt_node_errors.json").read_text())
            )
        pid = str(uuid.uuid4())
        outcome = self.outcomes.pop(0) if self.outcomes else "success"
        self.prompts[pid] = {
            "graph": body["prompt"],
            "client_id": body.get("client_id"),
            "number": len(self.prompts),
            "outcome": outcome,
        }
        self.queue.append(pid)
        self.submitted.set()
        gate, self.prompt_reply_gate = self.prompt_reply_gate, None  # one-shot
        if gate is not None:
            await gate.wait()
        if self.prompt_reply_drop:  # accepted upstream, connection lost before the reply
            self.prompt_reply_drop = False
            raise httpx.ReadError("connection dropped after accept")
        return httpx.Response(
            200, json={"prompt_id": pid, "number": len(self.prompts), "node_errors": {}}
        )

    async def _upload(self, request: httpx.Request) -> httpx.Response:
        form = parse_form(request)
        filename, data = form["image"]
        subfolder = form.get("subfolder", (None, b""))[1].decode()
        record = {
            "filename": filename,
            "subfolder": subfolder,
            "type": form.get("type", (None, b"input"))[1].decode(),
            "overwrite": form.get("overwrite", (None, b""))[1].decode(),
            "data": data,
        }
        self.uploads.append(record)
        if self.upload_gate is not None:
            await self.upload_gate.wait()
        if self.upload_mode == "500":
            return httpx.Response(500)
        name = f"renamed_{filename}" if self.upload_mode == "rename" else filename
        assert filename and "/" not in filename
        (self.input_dir / name).write_bytes(data)  # the mount is input/<subfolder>
        return httpx.Response(200, json={"name": name, "subfolder": subfolder, "type": "input"})

    def _get_history(self, pid: str) -> httpx.Response:
        if pid not in self.history and self.queue and self.queue[0] == pid:
            if self.hold is None or self.hold.is_set():
                self._complete(pid)
        if pid in self.history:
            return httpx.Response(200, json={pid: self.history[pid]})
        return httpx.Response(200, json={})

    async def _view(self, request: httpx.Request) -> httpx.Response:
        self.view_calls += 1
        if self.view_gate is not None:
            await self.view_gate.wait()
        if self.view_mode == "drop":
            raise httpx.ReadError("connection dropped", request=request)
        fname = request.url.params.get("filename", "")
        if self.view_mode == "404" or fname not in self.files:
            return httpx.Response(404)
        return httpx.Response(200, content=self.files[fname], headers={"content-type": "image/png"})

    async def _object_info(self) -> httpx.Response:
        self.object_info_calls += 1
        if self.object_info_gate is not None:
            await self.object_info_gate.wait()
        mode = self.object_info_mode
        if mode == "error":
            return httpx.Response(500)
        name = (
            "object_info_checkpoint_combo.json"
            if mode == "combo"
            else "object_info_checkpoint_legacy.json"
        )
        data = json.loads((FIXTURES / name).read_text())
        if mode == "absent":
            data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0] = [
                "v1-5-pruned-emaonly.safetensors"
            ]
        spec = data["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
        files = spec[1]["options"] if spec[0] == "COMBO" else spec[0]
        files.extend(self.extra_checkpoints)
        return httpx.Response(200, json=data)


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "api_keys_raw": KEY,
        "comfy_ws": False,  # no WebSocket server in the fake; tests opt in
        "comfyui_url": COMFY,
        "public_base_url": PUBLIC,
        "image_store_dir": str(tmp_path / "images"),
        "comfy_output_dir": str(tmp_path / "comfy-output"),
        "comfy_input_dir": str(tmp_path / "comfy-input"),
        "poll_interval_s": 0.01,
        "comfy_queue_poll_s": 0.05,
        "est_job_overhead_s": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def set_model_estimates(app: Any, model_id: str = "sdxl", **changes: Any) -> None:
    """Replace a registry entry's estimate parameters (est_seconds_per_step, cold_load_s)."""
    registry = app.state.registry
    registry._models[model_id] = dataclasses.replace(registry._models[model_id], **changes)


class Gateway:
    def __init__(self, app: Any, client: httpx.AsyncClient, fake: FakeComfy) -> None:
        self.app = app
        self.client = client
        self.fake = fake

    @property
    def queue(self) -> Any:
        return self.app.state.queue

    async def generate(self, **body: Any) -> httpx.Response:
        body.setdefault("prompt", "a red fox")
        return await self.client.post("/v1/images/generations", json=body, headers=AUTH)

    async def edit(
        self,
        image: bytes | None = None,
        mask: bytes | None = None,
        files: list[tuple[str, tuple[str, bytes, str]]] | None = None,
        **fields: Any,
    ) -> httpx.Response:
        """POST /v1/images/edits (multipart). ``files`` overrides the image/mask parts."""
        fields.setdefault("prompt", "a red fox in snow")
        if files is None:
            files = [("image", ("in.png", image if image is not None else make_png(), "image/png"))]
            if mask is not None:
                files.append(("mask", ("mask.png", mask, "image/png")))
        data = {k: str(v) for k, v in fields.items() if v is not None}
        return await self.client.post("/v1/images/edits", data=data, files=files, headers=AUTH)

    async def variation(self, image: bytes | None = None, **fields: Any) -> httpx.Response:
        files = [("image", ("in.png", image if image is not None else make_png(), "image/png"))]
        data = {k: str(v) for k, v in fields.items() if v is not None}
        return await self.client.post("/v1/images/variations", data=data, files=files, headers=AUTH)

    async def wait_until(self, predicate: Callable[[], bool], timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError("condition not reached in time")
            await asyncio.sleep(0.005)
