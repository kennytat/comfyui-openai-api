"""Async HTTP client for ComfyUI (submit, poll, fetch, cancel, cleanup, ownership).

Public API:

Exceptions (all carry ``code``, ``http_status``, ``non_retryable``, ``oom``, ``prompt_id``):

- ``ComfyError``: common base.
- ``UpstreamError`` (502, ``upstream_error``): prompt rejected (``node_errors``,
  ``non_retryable=True``), execution error, lost prompt, unexpected response.
  OOM execution errors have ``code="upstream_oom"``, ``oom=True``, ``non_retryable=True``.
- ``UpstreamInterrupted(UpstreamError)`` (502, ``upstream_interrupted``, non-retryable):
  ``execution_interrupted`` in history.
- ``UpstreamOutputMissing(UpstreamError)`` (502, ``upstream_output_missing``):
  ``/view`` returned 404 or history lists no images for the output node.
- ``UpstreamUnavailable(ComfyError)`` (503, ``upstream_unavailable``): connection
  refused/dropped/timed out.

Data:

- ``new_client_id() -> "openai-api-<uuid4>"``; ``CLIENT_ID_PREFIX``.
- ``QueueEntry(number, prompt_id, client_id, state, ownership)``; ``state`` is
  ``"running"``/``"pending"``, ``ownership`` is ``"owned"``/``"stale_gateway"``/``"external"``.
- ``QueueSnapshot(running, pending)`` with ``owned``, ``stale_gateway``, ``external``
  lists, ``external_count`` (external + stale gateway), ``is_running(id)``,
  ``is_pending(id)``, ``contains(id)``.
- ``WaitResult(prompt_id, history, submitted_at, started_at, completed_at)``:
  monotonic timestamps (``started_at`` is None if never observed running), properties
  ``comfy_wait_s`` (submit -> first seen running), ``gpu_s`` (running -> done),
  ``execution_s`` (from ComfyUI's own execution_start/success message timestamps).
- ``OutputImage(filename, subfolder, type)``; ``ReconcileReport``.

``ComfyClient(base_url, client_id=None, transport=None, poll_interval_s=0.5,
request_timeout_s=10.0, view_timeout_s=30.0, output_subfolder="openai-api",
input_subfolder="openai-api")``:

- ``await upload_image(data, filename, subfolder=None, type="input", overwrite=True)
  -> str``: ``POST /upload/image`` (multipart); returns the ComfyUI input name
  ``"<subfolder>/<name>"`` to put into ``LoadImage.image`` / ``LoadImageMask.image``.
- ``await submit(graph, client_id=None, extra_data=None) -> prompt_id`` (``extra_data``
  e.g. ``{"preview_method": "auto"}`` for latent previews over the WebSocket)
- ``notify(prompt_id, started=False)``: called by ``ComfyEvents`` (WebSocket); wakes the
  prompt's poll loop right away (``started=True`` records the WebSocket start time).
- ``await wait_for(prompt_id, deadline_monotonic=None) -> WaitResult``; raises
  ``TimeoutError`` once ``time.monotonic()`` passes the deadline, including while a
  ``/history`` or ``/queue`` poll is still in flight. A prompt missing from both ``/queue``
  and ``/history`` on two consecutive polls -> ``UpstreamError``.
- ``forget(prompt_id)``: drop per-prompt bookkeeping (call once the gateway is done with it).
- ``await get_history(prompt_id) -> dict | None`` (single ``/history`` entry)
- ``output_images(history, output_node) -> list[OutputImage]`` (no I/O)
- ``await fetch_outputs(history, output_node) -> list[bytes]``
- ``await cancel(prompt_id) -> "deleted" | "interrupted" | "not_found"``
- ``await delete_history(prompt_id)``
- ``await queue_snapshot() -> QueueSnapshot``
- ``await reconcile_startup() -> ReconcileReport`` (never raises)
- ``await available_files(loader, input) -> list[str] | None``
- ``await object_info(class_type=None) -> dict``; ``await health() -> dict``
- ``await aclose()`` / ``async with``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from app.comfy.object_info import combo_values

log = logging.getLogger(__name__)

CLIENT_ID_PREFIX = "openai-api-"

__all__ = [
    "CLIENT_ID_PREFIX",
    "ComfyClient",
    "ComfyError",
    "OutputImage",
    "QueueEntry",
    "QueueSnapshot",
    "ReconcileReport",
    "UpstreamError",
    "UpstreamInterrupted",
    "UpstreamOutputMissing",
    "UpstreamUnavailable",
    "WaitResult",
    "new_client_id",
]


def new_client_id() -> str:
    """Per-boot gateway identity carried by every submitted prompt."""
    return f"{CLIENT_ID_PREFIX}{uuid.uuid4()}"


# --------------------------------------------------------------------------- errors


class ComfyError(Exception):
    code = "upstream_error"
    http_status = 502

    def __init__(
        self,
        message: str,
        *,
        prompt_id: str | None = None,
        code: str | None = None,
        non_retryable: bool = False,
        oom: bool = False,
        details: Any = None,
    ):
        super().__init__(message)
        self.message = message
        self.prompt_id = prompt_id
        if code is not None:
            self.code = code
        self.non_retryable = non_retryable
        self.oom = oom
        self.details = details


class UpstreamError(ComfyError):
    """ComfyUI rejected or failed the job (HTTP 502)."""


class UpstreamInterrupted(UpstreamError):
    code = "upstream_interrupted"

    def __init__(self, message: str, **kw: Any):
        kw.setdefault("non_retryable", True)
        super().__init__(message, **kw)


class UpstreamOutputMissing(UpstreamError):
    code = "upstream_output_missing"


class UpstreamUnavailable(ComfyError):
    """ComfyUI unreachable or connection dropped (HTTP 503)."""

    code = "upstream_unavailable"
    http_status = 503


# --------------------------------------------------------------------------- data

Ownership = Literal["owned", "stale_gateway", "external"]


@dataclass(frozen=True)
class QueueEntry:
    number: Any
    prompt_id: str
    client_id: str | None
    state: Literal["running", "pending"]
    ownership: Ownership


@dataclass(frozen=True)
class QueueSnapshot:
    running: list[QueueEntry] = field(default_factory=list)
    pending: list[QueueEntry] = field(default_factory=list)

    @property
    def entries(self) -> list[QueueEntry]:
        return [*self.running, *self.pending]

    @property
    def owned(self) -> list[QueueEntry]:
        return [e for e in self.entries if e.ownership == "owned"]

    @property
    def stale_gateway(self) -> list[QueueEntry]:
        return [e for e in self.entries if e.ownership == "stale_gateway"]

    @property
    def external(self) -> list[QueueEntry]:
        return [e for e in self.entries if e.ownership == "external"]

    @property
    def external_count(self) -> int:
        """Entries not owned by this boot (external + stale gateway)."""
        return sum(1 for e in self.entries if e.ownership != "owned")

    def is_running(self, prompt_id: str) -> bool:
        return any(e.prompt_id == prompt_id for e in self.running)

    def is_pending(self, prompt_id: str) -> bool:
        return any(e.prompt_id == prompt_id for e in self.pending)

    def contains(self, prompt_id: str) -> bool:
        return self.is_running(prompt_id) or self.is_pending(prompt_id)


@dataclass(frozen=True)
class OutputImage:
    filename: str
    subfolder: str
    type: str


@dataclass(frozen=True)
class WaitResult:
    prompt_id: str
    history: dict[str, Any]
    submitted_at: float | None
    started_at: float | None
    completed_at: float

    @property
    def comfy_wait_s(self) -> float | None:
        if self.submitted_at is None or self.started_at is None:
            return None
        return max(0.0, self.started_at - self.submitted_at)

    @property
    def gpu_s(self) -> float | None:
        if self.started_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)

    @property
    def execution_s(self) -> float | None:
        """Execution time reported by ComfyUI's own message timestamps (ms)."""
        msgs = {name: data for name, data in _status_messages(self.history)}
        start = (msgs.get("execution_start") or {}).get("timestamp")
        end = (msgs.get("execution_success") or {}).get("timestamp")
        if isinstance(start, int | float) and isinstance(end, int | float):
            return max(0.0, (end - start) / 1000.0)
        return None


@dataclass
class ReconcileReport:
    skipped: bool = False
    deleted_pending: list[str] = field(default_factory=list)
    interrupted: list[str] = field(default_factory=list)
    history_deleted: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _status_messages(history: Mapping[str, Any]) -> list[tuple[str, dict]]:
    status = history.get("status") if isinstance(history, Mapping) else None
    msgs = status.get("messages") if isinstance(status, Mapping) else None
    out: list[tuple[str, dict]] = []
    for item in msgs or []:
        if isinstance(item, list | tuple) and len(item) >= 2 and isinstance(item[0], str):
            out.append((item[0], item[1] if isinstance(item[1], dict) else {}))
    return out


# --------------------------------------------------------------------------- client


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        client_id: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        poll_interval_s: float = 0.5,
        request_timeout_s: float = 10.0,
        view_timeout_s: float = 30.0,
        output_subfolder: str = "openai-api",
        input_subfolder: str = "openai-api",
    ):
        self.base_url = base_url.rstrip("/")
        self.input_subfolder = input_subfolder.strip("/")
        self.client_id = client_id or new_client_id()
        self.poll_interval_s = poll_interval_s
        self.output_subfolder = output_subfolder.strip("/")
        self._view_timeout = view_timeout_s
        self._http = httpx.AsyncClient(
            base_url=self.base_url, transport=transport, timeout=request_timeout_s
        )
        self._submitted_at: dict[str, float] = {}
        self._wakeups: dict[str, asyncio.Event] = {}
        self._ws_started_at: dict[str, float] = {}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> ComfyClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ transport

    async def _request(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            return await self._http.request(method, path, **kw)
        except httpx.TransportError as exc:
            raise UpstreamUnavailable(f"ComfyUI unreachable ({type(exc).__name__}): {exc}") from exc

    async def _json(self, method: str, path: str, **kw: Any) -> Any:
        resp = await self._request(method, path, **kw)
        if resp.status_code >= 400:
            raise UpstreamError(f"ComfyUI {method} {path} returned HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise UpstreamError(f"ComfyUI {method} {path} returned invalid JSON") from exc

    # ------------------------------------------------------------------ upload

    async def upload_image(
        self,
        data: bytes,
        filename: str,
        subfolder: str | None = None,
        type: str = "input",
        overwrite: bool = True,
    ) -> str:
        """Upload an input image; returns the name to reference it by in a workflow."""
        sub = (self.input_subfolder if subfolder is None else subfolder).strip("/")
        form = {"subfolder": sub, "type": type, "overwrite": "true" if overwrite else "false"}
        resp = await self._request(
            "POST",
            "/upload/image",
            files={"image": (filename, data, "image/png")},
            data=form,
        )
        if resp.status_code >= 400:
            raise UpstreamError(f"ComfyUI /upload/image returned HTTP {resp.status_code}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise UpstreamError("ComfyUI /upload/image returned invalid JSON") from exc
        name = body.get("name") if isinstance(body, dict) else None
        got_sub = str(body.get("subfolder") or "").strip("/") if isinstance(body, dict) else ""
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise UpstreamError("ComfyUI /upload/image response has no valid name")
        if got_sub != sub:
            raise UpstreamError(f"ComfyUI stored the upload in unexpected subfolder {got_sub!r}")
        return f"{sub}/{name}" if sub else name

    # ------------------------------------------------------------------ submit

    async def submit(
        self,
        graph: Mapping[str, Any],
        client_id: str | None = None,
        extra_data: Mapping[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {"prompt": graph, "client_id": client_id or self.client_id}
        if extra_data:
            body["extra_data"] = dict(extra_data)
        resp = await self._request("POST", "/prompt", json=body)
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        node_errors = data.get("node_errors") or {}
        if resp.status_code >= 400 or node_errors:
            err = data.get("error") if isinstance(data.get("error"), dict) else {}
            msg = err.get("message") or f"ComfyUI rejected the prompt (HTTP {resp.status_code})"
            if err.get("details"):
                msg = f"{msg}: {err['details']}"
            raise UpstreamError(
                msg,
                non_retryable=resp.status_code == 400 or bool(node_errors),
                details={"error": err, "node_errors": node_errors},
            )
        prompt_id = data.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise UpstreamError("ComfyUI /prompt response has no prompt_id")
        self._submitted_at[prompt_id] = time.monotonic()
        return prompt_id

    # ------------------------------------------------------------------ queue

    def _classify(self, client_id: str | None) -> Ownership:
        if client_id == self.client_id:
            return "owned"
        if isinstance(client_id, str) and client_id.startswith(CLIENT_ID_PREFIX):
            return "stale_gateway"
        return "external"

    def _parse_entries(self, raw: Any, state: Literal["running", "pending"]) -> list[QueueEntry]:
        entries: list[QueueEntry] = []
        for item in raw or []:
            if not isinstance(item, list | tuple) or len(item) < 2 or not isinstance(item[1], str):
                log.warning("comfy: unrecognised /queue entry shape")
                continue
            extra = item[3] if len(item) > 3 and isinstance(item[3], dict) else {}
            cid = extra.get("client_id")
            cid = cid if isinstance(cid, str) else None
            entries.append(QueueEntry(item[0], item[1], cid, state, self._classify(cid)))
        return entries

    async def queue_snapshot(self) -> QueueSnapshot:
        data = await self._json("GET", "/queue")
        if not isinstance(data, dict):
            raise UpstreamError("ComfyUI /queue returned an unexpected shape")
        return QueueSnapshot(
            running=self._parse_entries(data.get("queue_running"), "running"),
            pending=self._parse_entries(data.get("queue_pending"), "pending"),
        )

    # ------------------------------------------------------------------ wait

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        data = await self._json("GET", f"/history/{prompt_id}")
        if isinstance(data, dict):
            entry = data.get(prompt_id)
            if isinstance(entry, dict):
                return entry
        return None

    def _raise_for_history(self, prompt_id: str, entry: dict[str, Any]) -> None:
        status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        if status.get("status_str") != "error":
            return
        msgs = _status_messages(entry)
        for name, data in msgs:
            if name == "execution_interrupted":
                raise UpstreamInterrupted(
                    "ComfyUI job was interrupted",
                    prompt_id=prompt_id,
                    details=data,
                )
        for name, data in msgs:
            if name == "execution_error":
                etype = str(data.get("exception_type", ""))
                emsg = str(data.get("exception_message", "")).strip()
                node = data.get("node_type") or data.get("node_id")
                if "OutOfMemory" in etype:
                    raise UpstreamError(
                        f"ComfyUI ran out of GPU memory ({etype}) in node {node}",
                        prompt_id=prompt_id,
                        code="upstream_oom",
                        oom=True,
                        non_retryable=True,
                        details=data,
                    )
                raise UpstreamError(
                    f"ComfyUI execution error in node {node}: {etype}: {emsg}",
                    prompt_id=prompt_id,
                    details=data,
                )
        raise UpstreamError("ComfyUI reported an error status", prompt_id=prompt_id, details=status)

    async def wait_for(self, prompt_id: str, deadline_monotonic: float | None = None) -> WaitResult:
        if deadline_monotonic is None:
            return await self._poll_until_done(prompt_id, None)
        loop = asyncio.get_running_loop()
        try:
            # bounds in-flight polls too: a hung /history or /queue cannot overshoot
            async with asyncio.timeout_at(loop.time() + (deadline_monotonic - time.monotonic())):
                return await self._poll_until_done(prompt_id, deadline_monotonic)
        except TimeoutError as exc:
            raise TimeoutError(f"deadline exceeded waiting for prompt {prompt_id}") from exc

    async def _poll_until_done(
        self, prompt_id: str, deadline_monotonic: float | None
    ) -> WaitResult:
        started_at: float | None = None
        missing_polls = 0
        wake = self._wakeups.setdefault(prompt_id, asyncio.Event())
        while True:
            wake.clear()
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError(f"deadline exceeded waiting for prompt {prompt_id}")
            entry = await self.get_history(prompt_id)
            if started_at is None and prompt_id in self._ws_started_at:
                started_at = self._ws_started_at[prompt_id]
            if entry is not None:
                self._raise_for_history(prompt_id, entry)
                return WaitResult(
                    prompt_id=prompt_id,
                    history=entry,
                    submitted_at=self._submitted_at.get(prompt_id),
                    started_at=started_at,
                    completed_at=time.monotonic(),
                )
            snap = await self.queue_snapshot()
            if snap.is_running(prompt_id):
                missing_polls = 0
                if started_at is None:
                    started_at = time.monotonic()
            elif snap.is_pending(prompt_id):
                missing_polls = 0
            else:
                missing_polls += 1
                if missing_polls >= 2:
                    raise UpstreamError(
                        f"prompt {prompt_id} is in neither /queue nor /history (lost upstream)",
                        prompt_id=prompt_id,
                    )
            delay = self.poll_interval_s
            if deadline_monotonic is not None:
                delay = min(delay, max(0.0, deadline_monotonic - time.monotonic()))
            with contextlib.suppress(TimeoutError):  # a WebSocket event ends the wait early
                async with asyncio.timeout(delay):
                    await wake.wait()

    # ------------------------------------------------------------------ outputs

    @staticmethod
    def output_images(history: Mapping[str, Any], output_node: str) -> list[OutputImage]:
        outputs = history.get("outputs") if isinstance(history, Mapping) else None
        node_out = outputs.get(str(output_node)) if isinstance(outputs, Mapping) else None
        images = node_out.get("images") if isinstance(node_out, Mapping) else None
        result: list[OutputImage] = []
        for img in images or []:
            if isinstance(img, Mapping) and isinstance(img.get("filename"), str):
                result.append(
                    OutputImage(
                        filename=img["filename"],
                        subfolder=str(img.get("subfolder", "")),
                        type=str(img.get("type", "output")),
                    )
                )
        return result

    def _check_subfolder(self, img: OutputImage) -> None:
        sub = img.subfolder.strip("/")
        if img.type != "output" or not (
            sub == self.output_subfolder or sub.startswith(self.output_subfolder + "/")
        ):
            raise UpstreamError(
                f"output {img.filename!r} is outside the {self.output_subfolder!r} output folder"
            )

    async def fetch_outputs(self, history: Mapping[str, Any], output_node: str) -> list[bytes]:
        images = self.output_images(history, output_node)
        if not images:
            raise UpstreamOutputMissing(f"history lists no images for output node {output_node}")
        blobs: list[bytes] = []
        for img in images:
            self._check_subfolder(img)
            params = {"filename": img.filename, "subfolder": img.subfolder, "type": img.type}
            resp = await self._request("GET", "/view", params=params, timeout=self._view_timeout)
            if resp.status_code == 404:
                raise UpstreamOutputMissing(f"ComfyUI /view has no file {img.filename!r}")
            if resp.status_code >= 400:
                raise UpstreamError(f"ComfyUI /view returned HTTP {resp.status_code}")
            blobs.append(resp.content)
        return blobs

    # ------------------------------------------------------------------ cancel / cleanup

    async def _interrupt_if_running(self, prompt_id: str) -> bool:
        """Re-check ``queue_running`` right before a targeted ``/interrupt``."""
        snap = await self.queue_snapshot()
        if not snap.is_running(prompt_id):
            return False
        await self._json_ok("POST", "/interrupt", json={"prompt_id": prompt_id})
        return True

    async def _json_ok(self, method: str, path: str, **kw: Any) -> None:
        resp = await self._request(method, path, **kw)
        if resp.status_code >= 400:
            raise UpstreamError(f"ComfyUI {method} {path} returned HTTP {resp.status_code}")

    async def cancel(self, prompt_id: str) -> Literal["deleted", "interrupted", "not_found"]:
        """Targeted cancel: delete if pending, interrupt if running, else no-op."""
        snap = await self.queue_snapshot()
        if snap.is_pending(prompt_id):
            await self._json_ok("POST", "/queue", json={"delete": [prompt_id]})
            return "deleted"
        if snap.is_running(prompt_id) and await self._interrupt_if_running(prompt_id):
            return "interrupted"
        return "not_found"

    async def delete_history(self, prompt_id: str) -> None:
        await self._json_ok("POST", "/history", json={"delete": [prompt_id]})
        self.forget(prompt_id)

    def notify(self, prompt_id: str, started: bool = False) -> None:
        if started:
            self._ws_started_at.setdefault(prompt_id, time.monotonic())
        wake = self._wakeups.get(prompt_id)
        if wake is not None:
            wake.set()

    def forget(self, prompt_id: str) -> None:
        self._submitted_at.pop(prompt_id, None)
        self._wakeups.pop(prompt_id, None)
        self._ws_started_at.pop(prompt_id, None)

    async def reconcile_startup(self) -> ReconcileReport:
        """Remove prompts left by previous gateway boots. Best effort; never raises."""
        report = ReconcileReport()
        try:
            snap = await self.queue_snapshot()
        except Exception as exc:  # noqa: BLE001 - best effort
            log.warning("startup reconciliation skipped: %s", exc)
            report.skipped = True
            report.errors.append(str(exc))
            return report
        stale = snap.stale_gateway
        pending = [e.prompt_id for e in stale if e.state == "pending"]
        running = [e.prompt_id for e in stale if e.state == "running"]
        if pending:
            try:
                await self._json_ok("POST", "/queue", json={"delete": pending})
                report.deleted_pending.extend(pending)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"queue delete failed: {exc}")
        for pid in running:
            try:
                if await self._interrupt_if_running(pid):
                    report.interrupted.append(pid)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"interrupt {pid} failed: {exc}")
        ids = pending + running
        if ids:
            try:
                await self._json_ok("POST", "/history", json={"delete": ids})
                report.history_deleted.extend(ids)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"history delete failed: {exc}")
        if ids or report.errors:
            log.info(
                "startup reconciliation: deleted_pending=%s interrupted=%s errors=%s",
                report.deleted_pending,
                report.interrupted,
                report.errors,
            )
        return report

    # ------------------------------------------------------------------ info

    async def object_info(self, class_type: str | None = None) -> dict[str, Any]:
        path = f"/object_info/{class_type}" if class_type else "/object_info"
        data = await self._json("GET", path)
        if not isinstance(data, dict):
            raise UpstreamError(f"ComfyUI {path} returned an unexpected shape")
        return data

    async def available_files(self, loader: str, input_name: str) -> list[str] | None:
        """Options ComfyUI lists for ``loader.input``; ``None`` if unknown/unparseable."""
        info = await self.object_info(loader)
        values = combo_values(info, loader, input_name)
        if values is None:
            log.warning("comfy: cannot read file list for %s.%s", loader, input_name)
        return values

    async def health(self) -> dict[str, Any]:
        data = await self._json("GET", "/system_stats")
        return data if isinstance(data, dict) else {}
