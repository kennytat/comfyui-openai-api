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
  Helpers: ``checkpoint`` (first required file value, used for cold-load tracking),
  ``resolve_size(size) -> (w, h) | None`` (``None``/``"auto"`` -> default; aliases
  applied; ``None`` if not allowed), ``resolve_steps(quality, steps) -> int``
  (explicit ``steps`` wins over ``quality``).
- ``Registry.load(dir) -> Registry`` reads ``<dir>/registry.json`` and the referenced
  workflow files and validates them. ``registry.get(model_id) -> ModelEntry | None``,
  ``registry.models`` (all entries, registry order), ``registry.enabled()``.
- ``GenerationParams``: per-request inputs (prompt, width, height, n, seed, steps,
  quality, cfg_scale, negative_prompt, filename_prefix).
- ``resolve_params(model, params) -> GenerationParams``: fills defaults (random seed in
  ``0..2**53``, steps from quality/defaults, cfg, negative prompt). Call it before
  logging so the actual seed/steps are known; ``build_prompt`` calls it too (idempotent).
- ``build_prompt(model, params) -> dict``: deep-copied graph patched via the bindings.
- ``estimate_seconds(model, params, last_checkpoint, overhead_s=3.0) -> float``:
  ``overhead + steps * est_seconds_per_step * (w*h / 1024**2) * n
  + (cold_load_s if model.checkpoint != last_checkpoint else 0)``. A hint for admission.
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
    "GenerationParams",
    "ModelEntry",
    "Registry",
    "RequiredFile",
    "WorkflowError",
    "build_prompt",
    "estimate_seconds",
    "parse_size",
    "resolve_params",
]

REGISTRY_FILE = "registry.json"
SEED_MAX = 2**53
REQUIRED_BINDINGS = ("prompt", "width", "height", "n", "seed", "steps", "filename_prefix")
_SIZE_RE = re.compile(r"^(\d{2,5})x(\d{2,5})$")


class WorkflowError(ValueError):
    """Invalid registry entry or workflow file."""


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


def resolve_params(model: ModelEntry, params: GenerationParams) -> GenerationParams:
    """Return params with seed, steps, cfg_scale and negative_prompt filled in."""
    seed = params.seed if params.seed is not None else random.randint(0, SEED_MAX)
    steps = model.resolve_steps(params.quality, params.steps)
    cfg = params.cfg_scale if params.cfg_scale is not None else model.defaults.get("cfg_scale")
    neg = params.negative_prompt
    if neg is None:
        neg = model.defaults.get("negative_prompt")
    return replace(params, seed=seed, steps=steps, cfg_scale=cfg, negative_prompt=neg)


def build_prompt(model: ModelEntry, params: GenerationParams) -> dict[str, Any]:
    """Deep-copy the model's workflow and patch every bound parameter."""
    p = resolve_params(model, params)
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
    }
    graph = copy.deepcopy(model.workflow)
    for name, value in values.items():
        if value is None:
            continue  # keep the workflow's literal
        binding = model.bindings.get(name)
        if binding is None:
            if name in ("negative_prompt", "cfg_scale"):
                continue  # optional for models without these inputs
            raise WorkflowError(f"model {model.id!r} has no binding for {name!r}")
        node_id, input_name = binding
        graph[node_id]["inputs"][input_name] = value
    return graph


def estimate_seconds(
    model: ModelEntry,
    params: GenerationParams,
    last_checkpoint: str | None,
    overhead_s: float = 3.0,
) -> float:
    """Estimated wall time of one job (admission hint, not a guarantee)."""
    steps = model.resolve_steps(params.quality, params.steps)
    pixels = (params.width * params.height) / (1024 * 1024)
    est = overhead_s + steps * model.est_seconds_per_step * pixels * params.n
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


def _entry_from_dict(raw: dict[str, Any], directory: Path) -> ModelEntry:
    model_id = raw.get("id")
    if not isinstance(model_id, str) or not model_id:
        raise WorkflowError(f"registry entry without a valid id: {raw!r}")
    try:
        workflow_file = raw["workflow"]
        graph = _load_workflow(directory / workflow_file, model_id)

        bindings: dict[str, tuple[str, str]] = {}
        for name, target in raw["bindings"].items():
            if not (
                isinstance(target, list)
                and len(target) == 2
                and all(isinstance(t, str) for t in target)
            ):
                raise WorkflowError(
                    f"model {model_id!r}: binding {name!r} must be [node_id, input]"
                )
            node_id, input_name = target
            if node_id not in graph:
                raise WorkflowError(
                    f"model {model_id!r}: binding {name!r} -> missing node {node_id!r}"
                )
            if input_name not in graph[node_id]["inputs"]:
                raise WorkflowError(
                    f"model {model_id!r}: binding {name!r} -> node {node_id!r} "
                    f"has no input {input_name!r}"
                )
            bindings[name] = (node_id, input_name)
        missing = [b for b in REQUIRED_BINDINGS if b not in bindings]
        if missing:
            raise WorkflowError(f"model {model_id!r}: missing required bindings {missing}")

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
            defaults=dict(raw.get("defaults", {})),
            est_seconds_per_step=float(raw.get("est_seconds_per_step", 0.5)),
            cold_load_s=float(raw.get("cold_load_s", 0.0)),
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
