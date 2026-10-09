import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from contextweave.agents.base import AgentConfigError, read_exact
from contextweave.agents.blocks import find_block
from contextweave.agents.codex.config import codex_home, servers_in
from contextweave.agents.jsonio import JSONObject, object_at, parse_object
from contextweave.core.types import JSON

PARENT_SERVERS = frozenset({"context-engine", "ast-mcp", "cce"})
PARENT_COMMANDS = frozenset({"cce", "ast-mcp"})
_CCE_COMMAND = re.compile(r"(?<![a-z0-9])(cce|ast-mcp)(?![a-z0-9])", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_CCE_HEADING = re.compile(r"context engine|\bcce\b|ast[- ]mcp", re.IGNORECASE)
_PARENT_TOOLS = re.compile(
    r"\bast-mcp\b|\bcontext_search\b|\bsession_recall\b|\brecord_decision\b|\brecord_code_area\b"
)
_ROOT_SETTINGS = (".claude/settings.json", ".claude/settings.local.json")


class ConflictKind(StrEnum):
    MCP_SERVER = "mcp_server"
    HOOK = "hook"
    MEMORY_BLOCK = "memory_block"


class Scope(StrEnum):
    PROJECT = "project"
    USER = "user"


@dataclass(frozen=True, slots=True)
class Conflict:
    """One parent-tool artefact that collides with ContextWeave, with exact removal steps."""

    kind: ConflictKind
    scope: Scope
    path: Path
    what: str
    steps: tuple[str, ...]


def detect_conflicts(
    root: Path, home: Path, env: Mapping[str, str] | None = None
) -> list[Conflict]:
    """Read-only scan of project and user-level files for CCE and ast-mcp leftovers."""
    resolved = root.resolve()
    found: list[Conflict] = []
    found.extend(_project_mcp(resolved / ".mcp.json", Scope.PROJECT))
    for name in _ROOT_SETTINGS:
        found.extend(_hooks(resolved / name, Scope.PROJECT))
    found.extend(_memory_blocks(resolved / "CLAUDE.md", Scope.PROJECT))
    found.extend(_memory_blocks(resolved / ".claude" / "CLAUDE.md", Scope.PROJECT))
    found.extend(_memory_blocks(resolved / "AGENTS.md", Scope.PROJECT))
    found.extend(_user_mcp(home / ".claude.json", resolved))
    found.extend(_hooks(home / ".claude" / "settings.json", Scope.USER))
    found.extend(_memory_blocks(home / ".claude" / "CLAUDE.md", Scope.USER))
    if home in resolved.parents and home != resolved:
        found.extend(_memory_blocks(home / "CLAUDE.md", Scope.USER))
    found.extend(_codex_mcp(codex_home(home, env) / "config.toml"))
    return found


def format_conflicts(conflicts: list[Conflict]) -> str:
    """Human-readable report with removal steps."""
    lines = ["Parent tools conflict with ContextWeave (same tool names route ambiguously):"]
    for conflict in conflicts:
        lines.append(f"- [{conflict.scope}] {conflict.what} ({conflict.path})")
        lines.extend(f"    {step}" for step in conflict.steps)
    lines.append("Remove them, or pass --force to install anyway.")
    return "\n".join(lines)


def _load(path: Path) -> JSONObject | None:
    try:
        text = read_exact(path)
        return None if text is None else parse_object(text, path)
    except AgentConfigError, OSError:
        return None


def _server_conflicts(
    servers: JSONObject, path: Path, scope: Scope, where: str
) -> Iterator[Conflict]:
    for name, entry in servers.items():
        command = entry.get("command") if isinstance(entry, dict) else None
        binary = Path(command).name if isinstance(command, str) else ""
        if name.lower() not in PARENT_SERVERS and binary not in PARENT_COMMANDS:
            continue
        steps = [f'remove the "{name}" key from {where} in {path}']
        if name.lower() in {"context-engine", "cce"} or binary == "cce":
            steps.insert(0, "run `cce uninstall`")
        yield Conflict(ConflictKind.MCP_SERVER, scope, path, f'MCP server "{name}"', tuple(steps))


def _project_mcp(path: Path, scope: Scope) -> Iterator[Conflict]:
    document = _load(path)
    servers = object_at(document, "mcpServers") if document is not None else None
    if servers is not None:
        yield from _server_conflicts(servers, path, scope, "mcpServers")


def _codex_mcp(path: Path) -> Iterator[Conflict]:
    try:
        text = read_exact(path)
        servers = servers_in(text, path) if text is not None else None
    except AgentConfigError, OSError:
        return
    if servers:
        yield from _server_conflicts(servers, path, Scope.USER, "[mcp_servers]")


def _related(root: Path, candidate: str) -> bool:
    other = Path(candidate)
    return other == root or root in other.parents or other in root.parents


def _user_mcp(path: Path, root: Path) -> Iterator[Conflict]:
    document = _load(path)
    if document is None:
        return
    top = object_at(document, "mcpServers")
    if top is not None:
        yield from _server_conflicts(top, path, Scope.USER, "mcpServers")
    projects = object_at(document, "projects")
    for project, entry in (projects or {}).items():
        servers = object_at(entry, "mcpServers") if isinstance(entry, dict) else None
        if servers is not None and _related(root, project):
            yield from _server_conflicts(
                servers, path, Scope.USER, f'projects."{project}".mcpServers'
            )


def _commands(value: JSON) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "command" and isinstance(child, str):
                yield child
            else:
                yield from _commands(child)
    elif isinstance(value, list):
        for child in value:
            yield from _commands(child)


def _hooks(path: Path, scope: Scope) -> Iterator[Conflict]:
    document = _load(path)
    hooks = document.get("hooks") if document is not None else None
    if hooks is None:
        return
    offending = [command for command in _commands(hooks) if _CCE_COMMAND.search(command)]
    if offending:
        steps = (
            "run `cce uninstall`",
            f'or delete the hook entries whose command contains "cce" from {path}',
        )
        yield Conflict(
            ConflictKind.HOOK, scope, path, f"{len(offending)} CCE hook command(s)", steps
        )


@dataclass(frozen=True, slots=True)
class _Section:
    heading: str
    first_line: int
    last_line: int
    body: str


def _blank_span(text: str) -> str:
    span = find_block(text)
    if span is None:
        return text
    covered = text[span.start : span.end]
    return text[: span.start] + re.sub(r"[^\n]", " ", covered) + text[span.end :]


def _sections(text: str) -> list[_Section]:
    lines = text.splitlines()
    starts: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match is not None:
            starts.append((index, len(match.group(1)), match.group(2)))
    sections: list[_Section] = []
    own_ends = [start[0] for start in starts[1:]]
    own_ends.extend([len(lines)] if starts else [])
    if not starts or starts[0][0] > 0:
        end = starts[0][0] if starts else len(lines)
        sections.append(_Section("", 1, max(end, 1), "\n".join(lines[:end])))
    for position, ((index, level, title), own_end) in enumerate(zip(starts, own_ends, strict=True)):
        end = len(lines)
        for next_index, next_level, _ in starts[position + 1 :]:
            if next_level <= level:
                end = next_index
                break
        sections.append(
            _Section(
                "#" * level + " " + title, index + 1, end, "\n".join(lines[index + 1 : own_end])
            )
        )
    return sections


def _memory_blocks(path: Path, scope: Scope) -> Iterator[Conflict]:
    try:
        text = read_exact(path)
    except AgentConfigError, OSError:
        return
    if text is None:
        return
    flagged: list[_Section] = []
    for section in _sections(_blank_span(text)):
        hit = _CCE_HEADING.search(section.heading) or _PARENT_TOOLS.search(section.body)
        covered = any(
            outer.first_line <= section.first_line and section.last_line <= outer.last_line
            for outer in flagged
        )
        if hit and not covered:
            flagged.append(section)
    for section in flagged:
        label = section.heading or "preamble"
        where = f"lines {section.first_line}-{section.last_line}"
        steps = (
            f'delete the section "{label}" ({where}) from {path}',
            "if it came from CCE, `cce uninstall` removes it",
        )
        yield Conflict(ConflictKind.MEMORY_BLOCK, scope, path, f'routing section "{label}"', steps)
