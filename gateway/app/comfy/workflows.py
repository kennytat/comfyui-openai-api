"""Built-in workflow registry, prompt patching and job-duration estimates.

Public API:

- ``WorkflowError``: raised at load time for any invalid registry/workflow (fail fast).
- ``RequiredFile(loader, input, value)``: a model file that must be listed by ComfyUI
  (``GET /object_info/<loader>`` combo ``input``) for the model to be available.
- ``ModelEntry``: one registered model. Fields: ``id``, ``workflow`` (API-format graph,
  never mutated), ``workflow_file``, ``enabled``, ``required_files``, ``bindings``
  (``param -> (node_id, input_name)``), ``output_node``, ``sizes`` (``"WxH"`` whitelist),
  ``size_aliases`` (``"WxH" -> "WxH"``), ``default_size``, ``quality_steps``,
  ``defaults``, ``est_seconds_per_step``, ``cold_load_s``.
  Helpers: ``checkpoint`` (first required file value, used for cold-load tracking; list
  the main weights first, e.g. the diffusion model of a split UNET/CLIP/VAE model),
  ``resolve_size(size) -> (w, h) | None`` (``None``/``"auto"`` -> default; aliases
  applied; ``None`` if not allowed), ``resolve_steps(quality, steps) -> int``
  (explicit ``steps`` wins over ``quality``).
- ``Registry.load(dir) -> Registry`` reads ``<dir>/registry.json`` and the referenced
  workflow files and validates them. ``registry.get(model_id) -> ModelEntry | None``,
  ``registry.models`` (all entries, registry order), ``registry.enabled()``.
- ``Operation``: one workflow of a model for one API operation (``generate``, ``edit``,
  ``edit_masked``, ``variation``): ``workflow``, ``workflow_file``, ``bindings``,
  ``output_node``, ``defaults`` (e.g. ``denoise``, ``prompt``), ``est_factor`` and
  ``est_extra_s`` (estimate tuning). ``generate`` is the model's top-level
  ``workflow``/``bindings``/``output_node``; the others come from the optional registry
  ``operations`` object (each validated like the top-level workflow, with the bindings
  that operation requires). ``model.operation(name) -> Operation | None``.
- ``GenerationParams``: per-request inputs (prompt, width, height, n, seed, steps,
  quality, cfg_scale, negative_prompt, filename_prefix, plus ``operation`` and the
  img2img inputs ``image``/``mask`` (ComfyUI input names) and ``denoise``).
- ``resolve_params(model, params) -> GenerationParams``: fills defaults (random seed in
  ``0..2**53``, steps from quality/defaults, cfg, negative prompt, operation defaults for
  ``denoise``/``prompt``). Call it before logging so the actual seed/steps are known;
  ``build_prompt`` calls it too (idempotent).
- ``build_prompt(model, params) -> dict``: deep-copied graph of ``params.operation``
  patched via its bindings.
- ``compute_seconds(model, params) -> float``: the sampling part,
  ``steps * est_seconds_per_step * (w*h / 1024**2) * n * est_factor``.
- ``estimate_seconds(model, params, last_checkpoint, overhead_s=3.0, scale=1.0) -> float``:
  ``overhead + compute_seconds * scale + est_extra_s
  + (cold_load_s if model.checkpoint != last_checkpoint else 0)``
  (``est_factor``/``est_extra_s`` of the operation; 1 and 0 for ``generate``). ComfyUI's
  KSampler runs ``steps`` steps whatever ``denoise`` is, so denoise does not scale it.
  A hint for admission.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = [
    "OPERATIONS",
    "GenerationParams",
    "ModelEntry",
    "Operation",
    "Registry",
    "RequiredFile",
    "WorkflowError",
    "build_prompt",
    "compute_seconds",
    "estimate_seconds",
    "parse_size",
    "resolve_params",
]

REGISTRY_FILE = "registry.json"
SEED_MAX = 2**53
REQUIRED_BINDINGS = ("prompt", "width", "height", "n", "seed", "steps", "filename_prefix")
_IMG2IMG_BINDINGS = ("prompt", "n", "seed", "steps", "filename_prefix", "image", "denoise")
# Bindings each operation's workflow must provide (others are optional).
OPERATIONS: dict[str, tuple[str, ...]] = {
    "generate": REQUIRED_BINDINGS,
    "edit": _IMG2IMG_BINDINGS,
    "edit_masked": (*_IMG2IMG_BINDINGS, "mask"),
    "variation": _IMG2IMG_BINDINGS,
}
# Bindings that name an uploaded ComfyUI input file (not in the static combo list).
UPLOAD_BINDINGS = ("image", "mask")
_SIZE_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")


class WorkflowError(ValueError):
    """Invalid registry entry or workflow file."""


def _parse_defaults(raw: Any, label: str) -> dict[str, Any]:
    """``defaults`` must be an object; a ``denoise`` default must satisfy 0 < d <= 1."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise WorkflowError(f"{label}: defaults must be an object")
    if "denoise" in raw:
        d = raw["denoise"]
        if isinstance(d, bool) or not isinstance(d, int | float) or not 0 < d <= 1:
            raise WorkflowError(f"{label}: defaults.denoise must be a number in (0, 1]")
    return dict(raw)


def parse_size(size: str) -> tuple[int, int] | None:
    """Parse ``"WxH"`` into ``(w, h)``; ``None`` if malformed."""
    m = _SIZE_RE.match(size) if isinstance(size, str) else None
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


@dataclass(frozen=True)
class RequiredFile:
    loader: str
    input: str
    value: str


@dataclass(frozen=True)
class Operation:
    name: str
    workflow: dict[str, Any]
    workflow_file: str
    bindings: dict[str, tuple[str, str]]
    output_node: str
    defaults: dict[str, Any] = field(default_factory=dict)
    est_factor: float = 1.0
    est_extra_s: float = 0.0

    def upload_inputs(self) -> list[tuple[str, str]]:
        """``(class_type, input)`` of inputs that receive uploaded files (combo check skip)."""
        out = []
        for name in UPLOAD_BINDINGS:
            if name in self.bindings:
                node_id, input_name = self.bindings[name]
                out.append((self.workflow[node_id]["class_type"], input_name))
        return out


@dataclass(frozen=True)
class ModelEntry:
    id: str
    workflow: dict[str, Any]
    workflow_file: str
    enabled: bool
    required_files: tuple[RequiredFile, ...]
    bindings: dict[str, tuple[str, str]]
    output_node: str
    sizes: tuple[str, ...]
    size_aliases: dict[str, str]
    default_size: str
    quality_steps: dict[str, int]
    defaults: dict[str, Any] = field(default_factory=dict)
    est_seconds_per_step: float = 0.5
    cold_load_s: float = 0.0
    operations: dict[str, Operation] = field(default_factory=dict)

    def operation(self, name: str = "generate") -> Operation | None:
        """The workflow for ``name``; ``generate`` is the top-level workflow."""
        if name == "generate":
            return Operation(
                "generate", self.workflow, self.workflow_file, self.bindings, self.output_node
            )
        return self.operations.get(name)

    @property
    def checkpoint(self) -> str:
        """Identity of the weights this model loads (for cold-load estimates)."""
        return self.required_files[0].value if self.required_files else self.id

    def resolve_size(self, size: str | None) -> tuple[int, int] | None:
        """Map a requested size to an allowed ``(w, h)``; ``None`` if not allowed."""
        if size is None or size == "auto":
            size = self.default_size
        size = self.size_aliases.get(size, size)
        if size not in self.sizes:
            return None
        return parse_size(size)

    def resolve_steps(self, quality: str | None = None, steps: int | None = None) -> int:
        """Explicit ``steps`` wins; else ``quality_steps[quality]``; else defaults."""
        if steps is not None:
            return int(steps)
        if quality is not None and quality in self.quality_steps:
            return self.quality_steps[quality]
        if "steps" in self.defaults:
            return int(self.defaults["steps"])
        return self.quality_steps.get("auto", 25)


@dataclass(frozen=True)
class GenerationParams:
    prompt: str
    width: int
    height: int
    n: int = 1
    seed: int | None = None
    steps: int | None = None
    quality: str | None = None
    cfg_scale: float | None = None
    negative_prompt: str | None = None
    filename_prefix: str = "openai-api/job"
    operation: str = "generate"
    image: str | None = None  # ComfyUI input name ("<subfolder>/<file>") of an upload
    mask: str | None = None
    denoise: float | None = None


def _require_operation(model: ModelEntry, name: str) -> Operation:
    op = model.operation(name)
    if op is None:
        raise WorkflowError(f"model {model.id!r} does not support operation {name!r}")
    return op


def resolve_params(model: ModelEntry, params: GenerationParams) -> GenerationParams:
    """Return params with seed, steps, cfg_scale, negative_prompt (and denoise) filled in."""
    op = _require_operation(model, params.operation)
    seed = params.seed if params.seed is not None else random.randint(0, SEED_MAX)
    steps = model.resolve_steps(params.quality, params.steps)
    cfg = params.cfg_scale if params.cfg_scale is not None else model.defaults.get("cfg_scale")
    neg = params.negative_prompt
    if neg is None:
        neg = model.defaults.get("negative_prompt")
    denoise = params.denoise if params.denoise is not None else op.defaults.get("denoise")
    prompt = params.prompt
    if not prompt and "prompt" in op.defaults:
        prompt = str(op.defaults["prompt"])
    return replace(
        params,
        prompt=prompt,
        seed=seed,
        steps=steps,
        cfg_scale=cfg,
        negative_prompt=neg,
        denoise=None if denoise is None else float(denoise),
    )


def build_prompt(model: ModelEntry, params: GenerationParams) -> dict[str, Any]:
    """Deep-copy the operation's workflow and patch every bound parameter."""
    p = resolve_params(model, params)
    op = _require_operation(model, p.operation)
    required = OPERATIONS.get(op.name, REQUIRED_BINDINGS)
    values: dict[str, Any] = {
        "prompt": p.prompt,
        "negative_prompt": p.negative_prompt,
        "width": p.width,
        "height": p.height,
        "n": p.n,
        "seed": p.seed,
        "steps": p.steps,
        "cfg_scale": p.cfg_scale,
        "filename_prefix": p.filename_prefix,
        "image": p.image,
        "mask": p.mask,
        "denoise": p.denoise,
    }
    graph = copy.deepcopy(op.workflow)
    for name, value in values.items():
        if value is None:
            if name in required and name in UPLOAD_BINDINGS:
                raise WorkflowError(f"operation {op.name!r} needs an uploaded {name!r}")
            continue  # keep the workflow's literal
        binding = op.bindings.get(name)
        if binding is None:
            if name not in required:
                continue  # optional for this workflow (e.g. no negative prompt, no size)
            raise WorkflowError(f"model {model.id!r} has no binding for {name!r}")
        node_id, input_name = binding
        graph[node_id]["inputs"][input_name] = value
    return graph


def compute_seconds(model: ModelEntry, params: GenerationParams) -> float:
    """Sampling part of the estimate: ``steps * est_seconds_per_step * MP * n * est_factor``."""
    op = _require_operation(model, params.operation)
    steps = model.resolve_steps(params.quality, params.steps)
    pixels = (params.width * params.height) / (1024 * 1024)
    return steps * model.est_seconds_per_step * pixels * params.n * op.est_factor


def estimate_seconds(
    model: ModelEntry,
    params: GenerationParams,
    last_checkpoint: str | None,
    overhead_s: float = 3.0,
    scale: float = 1.0,
) -> float:
    """Estimated wall time of one job (admission hint, not a guarantee). ``scale``
    multiplies the sampling part (learned calibration, see ``EstimateCalibrator``)."""
    op = _require_operation(model, params.operation)
    est = overhead_s + compute_seconds(model, params) * scale + op.est_extra_s
    if model.checkpoint != last_checkpoint:
        est += model.cold_load_s
    return float(est)


def _load_workflow(path: Path, model_id: str) -> dict[str, Any]:
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkflowError(f"model {model_id!r}: workflow file {path} not found") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowError(
            f"model {model_id!r}: workflow {path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(graph, dict) or not graph:
        raise WorkflowError(f"model {model_id!r}: workflow {path} must be a non-empty object")
    if "nodes" in graph and "links" in graph:
        raise WorkflowError(
            f"model {model_id!r}: workflow {path} is UI format; export it in API format"
        )
    for node_id, node in graph.items():
        if (
            not isinstance(node, dict)
            or not isinstance(node.get("class_type"), str)
            or not isinstance(node.get("inputs"), dict)
        ):
            raise WorkflowError(
                f"model {model_id!r}: node {node_id!r} in {path} is not API format "
                "(needs class_type and inputs)"
            )
    return graph


def _parse_bindings(
    raw_bindings: Any, graph: dict[str, Any], label: str, required: tuple[str, ...]
) -> dict[str, tuple[str, str]]:
    if not isinstance(raw_bindings, dict):
        raise WorkflowError(f"{label}: bindings must be an object")
    bindings: dict[str, tuple[str, str]] = {}
    for name, target in raw_bindings.items():
        if not (
            isinstance(target, list)
            and len(target) == 2
            and all(isinstance(t, str) for t in target)
        ):
            raise WorkflowError(f"{label}: binding {name!r} must be [node_id, input]")
        node_id, input_name = target
        if node_id not in graph:
            raise WorkflowError(f"{label}: binding {name!r} -> missing node {node_id!r}")
        if input_name not in graph[node_id]["inputs"]:
            raise WorkflowError(
                f"{label}: binding {name!r} -> node {node_id!r} has no input {input_name!r}"
            )
        bindings[name] = (node_id, input_name)
    missing = [b for b in required if b not in bindings]
    if missing:
        raise WorkflowError(f"{label}: missing required bindings {missing}")
    return bindings


def _parse_operation(name: str, raw: Any, directory: Path, model_id: str) -> Operation:
    label = f"model {model_id!r} operation {name!r}"
    if name not in OPERATIONS or name == "generate":
        raise WorkflowError(
            f"{label}: unknown operation (allowed: edit, edit_masked, variation; "
            "generate is the top-level workflow)"
        )
    if not isinstance(raw, dict):
        raise WorkflowError(f"{label}: must be an object")
    try:
        workflow_file = raw["workflow"]
        graph = _load_workflow(directory / workflow_file, model_id)
        bindings = _parse_bindings(raw["bindings"], graph, label, OPERATIONS[name])
        output_node = str(raw["output_node"])
    except KeyError as exc:
        raise WorkflowError(f"{label}: missing field {exc.args[0]!r}") from exc
    if output_node not in graph:
        raise WorkflowError(f"{label}: output_node {output_node!r} not in workflow")
    return Operation(
        name=name,
        workflow=graph,
        workflow_file=workflow_file,
        bindings=bindings,
        output_node=output_node,
        defaults=_parse_defaults(raw.get("defaults"), label),
        est_factor=float(raw.get("est_factor", 1.0)),
        est_extra_s=float(raw.get("est_extra_s", 0.0)),
    )


def _check_operation_checkpoints(
    model_id: str, required_files: tuple[RequiredFile, ...], operations: dict[str, Operation]
) -> None:
    """Operation workflows must load the model's required files (cold-load tracking and
    availability checks assume one set of weights per model). Every loader node whose
    ``(class_type, input)`` names a required file must use one of the required values."""
    required: dict[tuple[str, str], set[str]] = {}
    for f in required_files:
        required.setdefault((f.loader, f.input), set()).add(f.value)
    if not required:
        return
    for name, op in operations.items():
        for node_id, node in op.workflow.items():
            for (loader, input_name), values in required.items():
                if node["class_type"] != loader:
                    continue
                value = node["inputs"].get(input_name)
                if value not in values:
                    raise WorkflowError(
                        f"model {model_id!r} operation {name!r}: node {node_id!r} loads "
                        f"{loader}.{input_name} {value!r}, expected the model's required file "
                        f"{sorted(values)[0]!r}"
                    )


def _entry_from_dict(raw: dict[str, Any], directory: Path) -> ModelEntry:
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise WorkflowError(f"registry entry without a valid id: {raw!r}")
    try:
        workflow_file = raw["workflow"]
        graph = _load_workflow(directory / workflow_file, model_id)
        bindings = _parse_bindings(raw["bindings"], graph, f"model {model_id!r}", REQUIRED_BINDINGS)

        output_node = str(raw["output_node"])
        if output_node not in graph:
            raise WorkflowError(f"model {model_id!r}: output_node {output_node!r} not in workflow")

        sizes = tuple(raw["sizes"])
        for s in sizes:
            if parse_size(s) is None:
                raise WorkflowError(f"model {model_id!r}: invalid size {s!r}")
        aliases = dict(raw.get("size_aliases", {}))
        for src, dst in aliases.items():
            if parse_size(src) is None or dst not in sizes:
                raise WorkflowError(f"model {model_id!r}: invalid size alias {src!r} -> {dst!r}")
        default_size = raw.get("default_size", sizes[0] if sizes else "")
        if default_size not in sizes:
            raise WorkflowError(f"model {model_id!r}: default_size {default_size!r} not in sizes")

        quality_steps = {str(k): int(v) for k, v in raw.get("quality_steps", {}).items()}
        required_files = tuple(
            RequiredFile(loader=f["loader"], input=f["input"], value=f["value"])
            for f in raw.get("required_files", [])
        )
        raw_ops = raw.get("operations", {})
        if not isinstance(raw_ops, dict):
            raise WorkflowError(f"model {model_id!r}: operations must be an object")
        operations = {
            name: _parse_operation(name, op, directory, model_id) for name, op in raw_ops.items()
        }
        _check_operation_checkpoints(model_id, required_files, operations)
        return ModelEntry(
            id=model_id,
            workflow=graph,
            workflow_file=workflow_file,
            enabled=bool(raw.get("enabled", True)),
            required_files=required_files,
            bindings=bindings,
            output_node=output_node,
            sizes=sizes,
            size_aliases=aliases,
            default_size=default_size,
            quality_steps=quality_steps,
            defaults=_parse_defaults(raw.get("defaults"), f"model {model_id!r}"),
            est_seconds_per_step=float(raw.get("est_seconds_per_step", 0.5)),
            cold_load_s=float(raw.get("cold_load_s", 0.0)),
            operations=operations,
        )
    except KeyError as exc:
        raise WorkflowError(f"model {model_id!r}: missing registry field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError(f"model {model_id!r}: invalid registry entry: {exc}") from exc


class Registry:
    """Immutable set of built-in models loaded from a workflows directory."""

    def __init__(self, models: list[ModelEntry]):
        self._models = {m.id: m for m in models}

    @classmethod
    def load(cls, directory: str | Path) -> Registry:
        directory = Path(directory)
        path = directory / REGISTRY_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise WorkflowError(f"registry file {path} not found") from exc
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"registry file {path} is not valid JSON: {exc}") from exc
        raw_models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(raw_models, list):
            raise WorkflowError(f"registry file {path} must contain a 'models' list")
        models: list[ModelEntry] = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                raise WorkflowError(f"registry entry must be an object: {raw!r}")
            entry = _entry_from_dict(raw, directory)
            if any(m.id == entry.id for m in models):
                raise WorkflowError(f"duplicate model id {entry.id!r}")
            models.append(entry)
        return cls(models)

    def get(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    @property
    def models(self) -> list[ModelEntry]:
        return list(self._models.values())

    def enabled(self) -> list[ModelEntry]:
        return [m for m in self._models.values() if m.enabled]

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._models

    def __len__(self) -> int:
        return len(self._models)
