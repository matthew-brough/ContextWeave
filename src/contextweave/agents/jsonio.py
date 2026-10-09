import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from contextweave.agents.base import AgentConfigError
from contextweave.core.types import JSON

type JSONObject = dict[str, JSON]

_INDENT = re.compile(r"^([ \t]+)\S", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class Layout:
    """Whitespace conventions of a JSON file, kept when it is rewritten."""

    indent: str = "  "
    trailing_newline: bool = True


def validate_json(raw: object) -> JSON:
    """Narrow a decoded JSON value into the JSON type, raising TypeError on foreign values."""
    if raw is None or isinstance(raw, str | int | float | bool):
        return raw
    if isinstance(raw, list):
        items = cast(list[object], raw)  # json.loads yields list[Unknown]; elements checked below
        return [validate_json(item) for item in items]
    if isinstance(raw, dict):
        members = cast(dict[object, object], raw)  # json.loads objects always have str keys
        return {str(key): validate_json(value) for key, value in members.items()}
    raise TypeError(type(raw).__name__)


def parse_object(text: str, path: Path) -> JSONObject:
    """Parse a JSON object; blank text is an empty object, anything else non-object raises."""
    if not text.strip():
        return {}
    try:
        decoded: object = json.loads(text)
        value = validate_json(decoded)
    except (ValueError, TypeError) as exc:
        raise AgentConfigError(path, f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentConfigError(path, "top level must be a JSON object")
    return value


def detect_layout(text: str) -> Layout:
    """Infer indent and trailing newline from existing text."""
    match = _INDENT.search(text)
    indent = match.group(1) if match else "  "
    return Layout(indent, text.endswith("\n") or not text.strip())


def dump_object(value: JSONObject, layout: Layout) -> str:
    """Serialise a JSON object with the given layout, preserving key order."""
    body = json.dumps(value, indent=layout.indent, ensure_ascii=False)
    return body + "\n" if layout.trailing_newline else body


def object_at(parent: JSONObject, key: str) -> JSONObject | None:
    """Return parent[key] when it is a JSON object."""
    value = parent.get(key)
    return value if isinstance(value, dict) else None


def list_at(parent: JSONObject, key: str) -> list[JSON] | None:
    """Return parent[key] when it is a JSON array."""
    value = parent.get(key)
    return value if isinstance(value, list) else None
