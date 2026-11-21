"""ComfyUI WebSocket events: completion wake-ups, progress and latent previews.

``ComfyEvents(client, reconnect_s=2.0)`` keeps one WebSocket open to
``<comfyui>/ws?clientId=<client.client_id>`` (``run()``, reconnecting with backoff).
ComfyUI sends progress and binary preview frames only to the client id that submitted a
prompt, which is this gateway's id, so every event here belongs to a gateway job.

- Completion (``execution_success``, ``execution_error``, ``execution_interrupted``,
  ``executing`` with ``node: null``) calls ``client.notify(prompt_id)``, which wakes that
  prompt's ``/history`` poll immediately; polling stays the source of truth and the
  fallback while the socket is down. ``execution_start`` is passed on too
  (``client.notify(prompt_id, started=True)``) for an accurate ``started_at``.
- ``progress`` (``value``/``max`` per sampler step) is stored per prompt.
- Binary preview frames (event type 1 ``PREVIEW_IMAGE``: 4-byte image type then JPEG/PNG
  bytes; type 4 ``PREVIEW_IMAGE_WITH_METADATA``: 4-byte JSON length, JSON with
  ``prompt_id``, image bytes) are stored as the prompt's latest ``Preview``; type 1 frames
  carry no prompt id and are attributed to the prompt of the latest progress message (one
  GPU job runs at a time). Previews only exist for prompts submitted with
  ``extra_data.preview_method`` (streaming requests with ``partial_images``).

``state(prompt_id) -> PromptState | None`` (``progress``, ``max``, ``preview``),
``wait_changed(timeout)`` (returns after the next update or the timeout),
``forget(prompt_id)``, ``connected``. Unknown message types are ignored.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import struct
from dataclasses import dataclass
from typing import Any

import websockets

log = logging.getLogger("gateway.comfy.events")

PREVIEW_IMAGE = 1
PREVIEW_IMAGE_WITH_METADATA = 4
MAX_FRAME_BYTES = 32 * 1024 * 1024
_DONE_TYPES = ("execution_success", "execution_error", "execution_interrupted")


@dataclass
class Preview:
    data: bytes
    mime: str
    step: int  # progress value when the preview arrived (0 if unknown)
    max_steps: int
    seq: int  # increases with every preview of the prompt


@dataclass
class PromptState:
    progress: int = 0
    max: int = 0
    preview: Preview | None = None


class ComfyEvents:
    def __init__(self, client: Any, reconnect_s: float = 2.0) -> None:
        self.client = client
        self.reconnect_s = reconnect_s
        self.connected = False
        self._states: dict[str, PromptState] = {}
        self._last_pid: str | None = None
        self._changed = asyncio.Event()

    @property
    def url(self) -> str:
        base = self.client.base_url
        scheme = "wss" if base.startswith("https://") else "ws"
        return f"{scheme}://{base.split('://', 1)[1]}/ws?clientId={self.client.client_id}"

    # ------------------------------------------------------------------ state

    def state(self, prompt_id: str) -> PromptState | None:
        return self._states.get(prompt_id)

    def forget(self, prompt_id: str) -> None:
        self._states.pop(prompt_id, None)
        if self._last_pid == prompt_id:
            self._last_pid = None

    async def wait_changed(self, timeout: float) -> None:
        event = self._changed
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(timeout):
                await event.wait()

    def _bump(self) -> None:
        event, self._changed = self._changed, asyncio.Event()
        event.set()

    def _state_for(self, prompt_id: str) -> PromptState:
        st = self._states.get(prompt_id)
        if st is None:
            st = self._states[prompt_id] = PromptState()
        return st

    # ------------------------------------------------------------------ dispatch

    def handle_text(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except ValueError:
            return
        if not isinstance(msg, dict) or not isinstance(msg.get("data"), dict):
            return
        mtype, data = msg.get("type"), msg["data"]
        pid = data.get("prompt_id")
        if not isinstance(pid, str):
            return
        if mtype == "progress":
            value, maximum = data.get("value"), data.get("max")
            if isinstance(value, int) and isinstance(maximum, int):
                st = self._state_for(pid)
                st.progress, st.max = value, maximum
                self._last_pid = pid
                self._bump()
        elif mtype == "execution_start":
            self._last_pid = pid
            self.client.notify(pid, started=True)
        elif mtype in _DONE_TYPES or (mtype == "executing" and data.get("node") is None):
            self.client.notify(pid)
            self._bump()

    def handle_binary(self, frame: bytes) -> None:
        if len(frame) < 8:
            return
        (event_type,) = struct.unpack(">I", frame[:4])
        pid: str | None
        if event_type == PREVIEW_IMAGE:
            (image_type,) = struct.unpack(">I", frame[4:8])
            mime = "image/png" if image_type == 2 else "image/jpeg"
            data, pid = frame[8:], self._last_pid
        elif event_type == PREVIEW_IMAGE_WITH_METADATA:
            (meta_len,) = struct.unpack(">I", frame[4:8])
            try:
                meta = json.loads(frame[8 : 8 + meta_len])
            except ValueError:
                return
            if not isinstance(meta, dict):
                return
            pid = meta.get("prompt_id") if isinstance(meta.get("prompt_id"), str) else None
            mime = str(meta.get("image_type") or "image/jpeg")
            data = frame[8 + meta_len :]
        else:
            return
        if pid is None or not data:
            return
        st = self._state_for(pid)
        seq = st.preview.seq + 1 if st.preview else 1
        st.preview = Preview(data, mime, st.progress, st.max, seq)
        self._bump()

    # ------------------------------------------------------------------ connection

    async def run(self) -> None:
        delay = self.reconnect_s
        while True:
            try:
                async with websockets.connect(
                    self.url, max_size=MAX_FRAME_BYTES, open_timeout=10
                ) as ws:
                    self.connected = True
                    delay = self.reconnect_s
                    log.info("comfy websocket connected")
                    async for message in ws:
                        if isinstance(message, bytes):
                            self.handle_binary(message)
                        else:
                            self.handle_text(message)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any failure
                if self.connected:
                    log.warning("comfy websocket lost: %s", exc)
            finally:
                self.connected = False
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)
