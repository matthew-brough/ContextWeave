import json
from collections.abc import Mapping, Sequence
from typing import cast

from contextweave.core.types import JSON
from contextweave.models.errors import ModelResponseError


def to_json(value: object) -> JSON:
    """Validate an untyped decoded value into JSON, rejecting anything else."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)  # keys are checked as str below
        result: dict[str, JSON] = {}
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise ModelResponseError("non-string key in JSON object")
            result[key] = to_json(item)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = cast(Sequence[object], value)  # elements are validated recursively
        return [to_json(item) for item in items]
    raise ModelResponseError(f"unsupported JSON value: {type(value).__name__}")


def parse_json_text(text: str) -> JSON:
    """Decode JSON text into a validated JSON value."""
    try:
        decoded: object = json.loads(text)
    except ValueError as exc:
        raise ModelResponseError(f"invalid JSON: {exc}") from exc
    return to_json(decoded)


def as_object(value: JSON, what: str) -> dict[str, JSON]:
    """Require a JSON object."""
    if not isinstance(value, dict):
        raise ModelResponseError(f"{what}: expected object")
    return value


def as_list(value: JSON, what: str) -> list[JSON]:
    """Require a JSON array."""
    if not isinstance(value, list):
        raise ModelResponseError(f"{what}: expected array")
    return value


def as_str(value: JSON, what: str) -> str:
    """Require a JSON string."""
    if not isinstance(value, str):
        raise ModelResponseError(f"{what}: expected string")
    return value


def as_vector(value: JSON, what: str) -> list[float]:
    """Require a JSON array of numbers."""
    vector: list[float] = []
    for item in as_list(value, what):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ModelResponseError(f"{what}: expected number")
        vector.append(float(item))
    return vector


def field(obj: Mapping[str, JSON], key: str, what: str) -> JSON:
    """Fetch a required key from a JSON object."""
    if key not in obj:
        raise ModelResponseError(f"{what}: missing '{key}'")
    return obj[key]
