"""ComfyUI integration: workflow registry (``workflows``), ``/object_info`` parsing and
graph validation (``object_info``), and the async HTTP client (``client``).

Re-exports the main names; see each module's docstring for the full API.
"""

from app.comfy.client import (
    CLIENT_ID_PREFIX,
    ComfyClient,
    ComfyError,
    OutputImage,
    QueueEntry,
    QueueSnapshot,
    ReconcileReport,
    UpstreamError,
    UpstreamInterrupted,
    UpstreamOutputMissing,
    UpstreamUnavailable,
    WaitResult,
    new_client_id,
)
from app.comfy.object_info import combo_values, parse_combo, validate_graph
from app.comfy.workflows import (
    GenerationParams,
    ModelEntry,
    Registry,
    RequiredFile,
    WorkflowError,
    build_prompt,
    estimate_seconds,
    resolve_params,
)

__all__ = [
    "CLIENT_ID_PREFIX",
    "ComfyClient",
    "ComfyError",
    "GenerationParams",
    "ModelEntry",
    "OutputImage",
    "QueueEntry",
    "QueueSnapshot",
    "ReconcileReport",
    "Registry",
    "RequiredFile",
    "UpstreamError",
    "UpstreamInterrupted",
    "UpstreamOutputMissing",
    "UpstreamUnavailable",
    "WaitResult",
    "WorkflowError",
    "build_prompt",
    "combo_values",
    "estimate_seconds",
    "new_client_id",
    "parse_combo",
    "resolve_params",
    "validate_graph",
]
