import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from contextweave.agents.base import AgentConfigError
from contextweave.agents.jsonio import JSONObject, validate_json
from contextweave.core.types import JSON

SERVER_NAME = "contextweave"
SERVER_ENTRY: JSONObject = {"command": "cw", "args": ["serve"]}
_TABLE = "mcp_servers"
_HEADER = re.compile(r"^[ \t]*\[\[?([^\]\n]*)\]\]?[ \t]*(?:#.*)?$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class _Span:
    start: int
    end: int


def codex_home(home: Path, env: Mapping[str, str] | None = None) -> Path:
    """Codex's config directory: $CODEX_HOME when set and non-empty, else ~/.codex."""
    configured = (env or {}).get("CODEX_HOME")
    return Path(configured).expanduser() if configured else home / ".codex"


def _parse(text: str, path: Path) -> JSONObject:
    try:
        decoded = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise AgentConfigError(path, f"invalid TOML: {exc}") from exc
    table: object = decoded.get(_TABLE)
    if table is None:
        return {}
    try:
        servers = validate_json(table)
    except TypeError as exc:
        raise AgentConfigError(path, f"unsupported value in [{_TABLE}]") from exc
    if not isinstance(servers, dict):
        raise AgentConfigError(path, f"[{_TABLE}] must be a table")
    return servers


def servers_in(text: str, path: Path) -> JSONObject:
    """The [mcp_servers] table of a Codex config, empty when absent."""
    return _parse(text, path)


def _key_path(header: str) -> list[str]:
    return [part.strip().strip("\"'") for part in header.split(".")]


def _find_span(text: str) -> _Span | None:
    ours = [SERVER_NAME]
    start: int | None = None
    for match in _HEADER.finditer(text):
        parts = _key_path(match.group(1))
        mine = parts[0] == _TABLE and parts[1:2] == ours
        if start is None and mine:
            start = match.start()
        elif start is not None and not mine:
            return _Span(start, match.start())
    return None if start is None else _Span(start, len(text))


def _section() -> str:
    return tomli_w.dumps({_TABLE: {SERVER_NAME: SERVER_ENTRY}})


def _is_current(entry: JSON | None) -> bool:
    return (
        isinstance(entry, dict)
        and entry.get("command") == SERVER_ENTRY["command"]
        and entry.get("args") == SERVER_ENTRY["args"]
    )


def _check(text: str, path: Path) -> str:
    _parse(text, path)
    return text


def merge_server(text: str | None, path: Path) -> str:
    """Ensure [mcp_servers.contextweave] exists, touching no other byte of the file."""
    current = text or ""
    entry = servers_in(current, path).get(SERVER_NAME)
    span = _find_span(current)
    if entry is not None and span is None:
        raise AgentConfigError(path, f"{SERVER_NAME} server is defined in a form cw cannot edit")
    if span is not None:
        if _is_current(entry):
            return current
        return _check(current[: span.start] + _section() + current[span.end :], path)
    base = current if not current or current.endswith("\n") else current + "\n"
    return _check(base + ("\n" if base else "") + _section(), path)


def unmerge_server(text: str, path: Path) -> str | None:
    """Remove [mcp_servers.contextweave]; None when nothing else remains."""
    servers_in(text, path)
    span = _find_span(text)
    if span is None:
        return text
    head, tail = text[: span.start], text[span.end :]
    if not tail and head.endswith("\n\n"):
        head = head[:-1]
    return _check(head + tail, path) if (head + tail).strip() else None
