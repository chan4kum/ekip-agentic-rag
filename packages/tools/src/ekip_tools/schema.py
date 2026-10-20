"""Deterministic argument validation against a tiny JSON-schema subset.

Why not full JSON Schema: tool schemas are reviewed config, and a small, closed subset is easy to
reason about, cannot be made expensive (no $ref, no combinators), and lets us force the
properties we care about: every string is length-bounded and unknown fields are rejected.

Error messages name the FIELD and the RULE only, never the value: values come from a model and
may contain secrets or injected text, and error text ends up in logs and in the model's context.
"""

import re
from collections.abc import Mapping
from typing import Any

_TYPES = ("string", "integer", "number", "boolean")
_STRING_KEYS = {"type", "minLength", "maxLength", "pattern", "enum", "description"}
_NUMBER_KEYS = {"type", "minimum", "maximum", "enum", "description"}
_BOOL_KEYS = {"type", "description"}
_TOP_KEYS = {"type", "properties", "required", "description"}
MAX_PATTERN_LENGTH = 200


class SchemaError(ValueError):
    """The schema itself is outside the supported subset (a configuration bug)."""


def check_schema(schema: Mapping[str, Any]) -> None:
    """Reject schemas we cannot validate safely. Called when the registry loads (fail fast)."""
    if set(schema) - _TOP_KEYS:
        raise SchemaError(f"unsupported top-level keys: {sorted(set(schema) - _TOP_KEYS)}")
    if schema.get("type") != "object":
        raise SchemaError("top-level type must be 'object'")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise SchemaError("properties must be a mapping")
    required = schema.get("required", [])
    if not isinstance(required, list) or any(r not in properties for r in required):
        raise SchemaError("required must list declared properties")
    for name, prop in properties.items():
        _check_property(name, prop)


def _check_property(name: str, prop: object) -> None:
    if not isinstance(prop, Mapping):
        raise SchemaError(f"property {name!r} must be a mapping")
    kind = prop.get("type")
    if kind not in _TYPES:
        raise SchemaError(f"property {name!r}: unsupported type {kind!r}")
    allowed = {"string": _STRING_KEYS, "boolean": _BOOL_KEYS}.get(kind, _NUMBER_KEYS)
    if set(prop) - allowed:
        raise SchemaError(f"property {name!r}: unsupported keys {sorted(set(prop) - allowed)}")
    if kind == "string":
        if "maxLength" not in prop and "enum" not in prop:
            raise SchemaError(f"property {name!r}: strings need maxLength or enum")
        pattern = prop.get("pattern")
        if pattern is not None:
            if not isinstance(pattern, str) or len(pattern) > MAX_PATTERN_LENGTH:
                raise SchemaError(f"property {name!r}: pattern too long")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise SchemaError(f"property {name!r}: bad pattern") from exc
    if kind in ("integer", "number") and not ("minimum" in prop and "maximum" in prop):
        raise SchemaError(f"property {name!r}: numbers need minimum and maximum")


def validate_arguments(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[str]:
    """Return a list of problems (empty means valid). Unknown fields are always a problem."""
    problems: list[str] = []
    properties: Mapping[str, Mapping[str, Any]] = schema.get("properties", {})
    for name in arguments:
        if name not in properties:
            problems.append(f"{_label(name)}: unknown field")
    for name in schema.get("required", []):
        if name not in arguments:
            problems.append(f"{name}: required")
    for name, value in arguments.items():
        prop = properties.get(name)
        if prop is not None:
            problems.extend(_check_value(name, prop, value))
    return problems


def _label(name: object) -> str:
    # Field names also come from the model: cap and strip them before they are echoed.
    text = str(name)[:40]
    return "".join(ch if ch.isalnum() or ch in "_-" else "?" for ch in text)


def _check_value(name: str, prop: Mapping[str, Any], value: Any) -> list[str]:
    kind = prop["type"]
    if kind == "string":
        return _check_string(name, prop, value)
    if kind == "boolean":
        return [] if isinstance(value, bool) else [f"{name}: must be a boolean"]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return [f"{name}: must be a number"]
    if kind == "integer" and isinstance(value, float):
        return [f"{name}: must be an integer"]
    problems = []
    if value < prop["minimum"] or value > prop["maximum"]:
        problems.append(f"{name}: out of range")
    if "enum" in prop and value not in prop["enum"]:
        problems.append(f"{name}: not an allowed value")
    return problems


def _check_string(name: str, prop: Mapping[str, Any], value: Any) -> list[str]:
    if not isinstance(value, str):
        return [f"{name}: must be a string"]
    problems = []
    if len(value) > prop.get("maxLength", len(value)):
        problems.append(f"{name}: too long")
    if len(value) < prop.get("minLength", 0):
        problems.append(f"{name}: too short")
    if "enum" in prop and value not in prop["enum"]:
        problems.append(f"{name}: not an allowed value")
    pattern = prop.get("pattern")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        problems.append(f"{name}: wrong format")
    return problems
