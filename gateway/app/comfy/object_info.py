"""Parsing and validation helpers for ComfyUI ``/object_info`` data.

Public API:

- ``parse_combo(input_spec) -> list[str] | None``: option list of a combo input spec.
  Handles the legacy shape ``[["a", "b"], {...}]`` and the newer
  ``["COMBO", {"options": ["a", "b"]}]``. Returns ``None`` for non-combo inputs
  (``["INT", {...}]``, ``["MODEL"]``) and for unknown shapes (the latter logs a warning).
- ``input_spec(object_info, class_type, input_name) -> list | None``: raw spec of one
  required/optional input of a node class.
- ``combo_values(object_info, class_type, input_name) -> list[str] | None``: combo options
  of one input (``None`` if the class, the input or a parseable combo is missing).
- ``validate_graph(graph, object_info, skip_combo=()) -> list[str]``: validates an
  API-format graph: every ``class_type`` exists, every required input is present
  (literal or link), links point to existing nodes, literal combo values are legal and
  numeric literals respect ``min``/``max``. ``skip_combo`` holds ``(class_type, input)``
  pairs whose option lists are model files checked separately (``required_files``).
  Returns a list of human-readable error strings (empty == valid).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

log = logging.getLogger(__name__)

__all__ = ["combo_values", "input_spec", "parse_combo", "validate_graph"]

_NON_COMBO_MARKERS = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def parse_combo(input_spec: Any) -> list[str] | None:
    """Return the option list of a combo input spec, or ``None``.

    ``None`` means "not a combo" for scalar/link types, and "unknown shape" (with a
    warning) for anything else that cannot be interpreted.
    """
    if not isinstance(input_spec, list | tuple) or not input_spec:
        log.warning("object_info: unrecognised input spec shape: %r", input_spec)
        return None
    head = input_spec[0]
    # Legacy combo: first element is the plain option list.
    if isinstance(head, list):
        if _is_str_list(head):
            return list(head)
        log.warning("object_info: combo list with non-string options: %r", head)
        return None
    if isinstance(head, str):
        if head == "COMBO":
            opts = input_spec[1] if len(input_spec) > 1 else None
            if isinstance(opts, Mapping) and _is_str_list(opts.get("options")):
                return list(opts["options"])
            log.warning("object_info: COMBO spec without a string options list: %r", input_spec)
            return None
        # Scalar types (INT/FLOAT/...) and link types (MODEL, CLIP, ...) are not combos.
        return None
    log.warning("object_info: unrecognised input spec shape: %r", input_spec)
    return None


def _class_inputs(object_info: Mapping[str, Any], class_type: str) -> Mapping[str, Any] | None:
    node = object_info.get(class_type)
    if not isinstance(node, Mapping):
        return None
    inputs = node.get("input")
    return inputs if isinstance(inputs, Mapping) else {}


def input_spec(object_info: Mapping[str, Any], class_type: str, input_name: str) -> list | None:
    """Return the raw spec of a required or optional input of ``class_type``."""
    inputs = _class_inputs(object_info, class_type)
    if inputs is None:
        return None
    for section in ("required", "optional"):
        group = inputs.get(section)
        if isinstance(group, Mapping) and input_name in group:
            spec = group[input_name]
            return spec if isinstance(spec, list) else None
    return None


def combo_values(
    object_info: Mapping[str, Any], class_type: str, input_name: str
) -> list[str] | None:
    """Return the combo options of one input, or ``None`` when unavailable/unknown."""
    spec = input_spec(object_info, class_type, input_name)
    if spec is None:
        return None
    return parse_combo(spec)


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _check_literal(
    node_id: str, class_type: str, name: str, value: Any, spec: list, skip: bool
) -> list[str]:
    errors: list[str] = []
    head = spec[0] if spec else None
    is_combo_shape = isinstance(head, list) or head == "COMBO"
    if is_combo_shape:
        if skip:
            return errors
        options = parse_combo(spec)
        if options is None:
            errors.append(f"node {node_id} ({class_type}).{name}: cannot parse combo options")
        elif value not in options:
            errors.append(
                f"node {node_id} ({class_type}).{name}: value {value!r} not in allowed options"
            )
        return errors
    if head in ("INT", "FLOAT") and isinstance(value, int | float) and not isinstance(value, bool):
        opts = spec[1] if len(spec) > 1 and isinstance(spec[1], Mapping) else {}
        lo, hi = opts.get("min"), opts.get("max")
        if lo is not None and value < lo:
            errors.append(f"node {node_id} ({class_type}).{name}: {value} < min {lo}")
        if hi is not None and value > hi:
            errors.append(f"node {node_id} ({class_type}).{name}: {value} > max {hi}")
        if head == "INT" and isinstance(value, float) and not value.is_integer():
            errors.append(f"node {node_id} ({class_type}).{name}: expected INT, got {value!r}")
    elif head in ("INT", "FLOAT") and not isinstance(value, int | float):
        errors.append(f"node {node_id} ({class_type}).{name}: expected {head}, got {value!r}")
    elif head == "STRING" and not isinstance(value, str):
        errors.append(f"node {node_id} ({class_type}).{name}: expected STRING, got {value!r}")
    return errors


def validate_graph(
    graph: Mapping[str, Any],
    object_info: Mapping[str, Any],
    skip_combo: Iterable[tuple[str, str]] = (),
) -> list[str]:
    """Validate an API-format graph against ``object_info``; return error strings."""
    skip = set(skip_combo)
    errors: list[str] = []
    if not isinstance(graph, Mapping) or not graph:
        return ["graph is empty or not an object"]
    for node_id, node in graph.items():
        if not isinstance(node, Mapping) or not isinstance(node.get("class_type"), str):
            errors.append(f"node {node_id}: missing class_type (not API format)")
            continue
        class_type = node["class_type"]
        inputs = node.get("inputs", {})
        if not isinstance(inputs, Mapping):
            errors.append(f"node {node_id} ({class_type}): inputs is not an object")
            continue
        class_inputs = _class_inputs(object_info, class_type)
        if class_inputs is None:
            errors.append(f"node {node_id}: unknown class_type {class_type!r}")
            continue
        required = class_inputs.get("required") or {}
        optional = class_inputs.get("optional") or {}
        for name in required:
            if name not in inputs:
                errors.append(f"node {node_id} ({class_type}): missing required input {name!r}")
        for name, value in inputs.items():
            spec = required.get(name, optional.get(name))
            if _is_link(value):
                if value[0] not in graph:
                    errors.append(
                        f"node {node_id} ({class_type}).{name}: link to missing node {value[0]!r}"
                    )
                continue
            if not isinstance(spec, list):
                # Unknown/extra literal inputs are ignored by ComfyUI; not an error.
                continue
            errors.extend(
                _check_literal(node_id, class_type, name, value, spec, (class_type, name) in skip)
            )
    return errors
