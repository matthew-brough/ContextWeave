from collections.abc import Mapping
from pathlib import Path

from contextweave.agents.base import AgentConfigError, HookSpec
from contextweave.agents.jsonio import (
    JSONObject,
    Layout,
    detect_layout,
    dump_object,
    list_at,
    object_at,
    parse_object,
)
from contextweave.agents.routing import routing_block
from contextweave.core.dist import package_version
from contextweave.core.types import JSON

SERVER_NAME = "contextweave"
SERVER_ENTRY: JSONObject = {"command": "cw", "args": ["serve"]}


def _load(text: str | None, path: Path) -> tuple[JSONObject, Layout]:
    if text is None:
        return {}, Layout()
    return parse_object(text, path), detect_layout(text)


def _finish(obj: JSONObject, layout: Layout) -> str | None:
    return dump_object(obj, layout) if obj else None


def _section(parent: JSONObject, key: str, path: Path) -> JSONObject | None:
    section = object_at(parent, key)
    if section is None and key in parent:
        raise AgentConfigError(path, f"'{key}' must be a JSON object")
    return section


def merge_mcp(text: str | None, path: Path) -> str:
    """Set mcpServers.contextweave, keeping every other key and its order."""
    obj, layout = _load(text, path)
    servers = _section(obj, "mcpServers", path)
    if servers is not None and servers.get(SERVER_NAME) == SERVER_ENTRY:
        return text if text is not None else dump_object(obj, layout)
    if servers is None:
        servers = {}
        obj["mcpServers"] = servers
    servers[SERVER_NAME] = dict(SERVER_ENTRY)
    return dump_object(obj, layout)


def unmerge_mcp(text: str, path: Path) -> str | None:
    """Remove mcpServers.contextweave; None when nothing else remains."""
    obj, layout = _load(text, path)
    servers = _section(obj, "mcpServers", path)
    if servers is None or SERVER_NAME not in servers:
        return text
    del servers[SERVER_NAME]
    if not servers:
        del obj["mcpServers"]
    return _finish(obj, layout)


def hook_entry(spec: HookSpec) -> JSONObject:
    """The settings.json hook object for one spec."""
    return {"type": "command", "command": spec.command, "timeout": spec.timeout_s}


def _group_hooks(group: JSON) -> list[JSON] | None:
    return list_at(group, "hooks") if isinstance(group, dict) else None


def _has_command(group: JSON, command: str) -> bool:
    entries = _group_hooks(group)
    return entries is not None and any(
        isinstance(entry, dict) and entry.get("command") == command for entry in entries
    )


def hooks_document(specs: Mapping[str, HookSpec]) -> JSONObject:
    """A complete {hooks: ...} object for a plugin hooks.json."""
    return {"hooks": {event: [{"hooks": [hook_entry(spec)]}] for event, spec in specs.items()}}


def merge_hooks(text: str | None, path: Path, specs: Mapping[str, HookSpec]) -> str:
    """Add our hook command under each event unless already present; user hooks untouched."""
    obj, layout = _load(text, path)
    hooks = _section(obj, "hooks", path)
    if hooks is None:
        hooks = {}
        obj["hooks"] = hooks
    changed = False
    for event, spec in specs.items():
        groups = list_at(hooks, event)
        if groups is None and event in hooks:
            raise AgentConfigError(path, f"hooks.{event} must be a JSON array")
        if groups is None:
            groups = []
            hooks[event] = groups
        if any(_has_command(group, spec.command) for group in groups):
            continue
        groups.append({"hooks": [hook_entry(spec)]})
        changed = True
    if not changed and text is not None:
        return text
    return dump_object(obj, layout)


def _strip_commands(groups: list[JSON], commands: frozenset[str]) -> bool:
    removed = False
    for group in list(groups):
        entries = _group_hooks(group)
        if entries is None:
            continue
        kept = [
            entry
            for entry in entries
            if not (isinstance(entry, dict) and entry.get("command") in commands)
        ]
        if len(kept) == len(entries):
            continue
        removed = True
        if kept:
            entries[:] = kept
        else:
            groups.remove(group)
    return removed


def unmerge_hooks(text: str, path: Path, specs: Mapping[str, HookSpec]) -> str | None:
    """Remove only our hook commands, dropping groups, events and hooks that become empty."""
    obj, layout = _load(text, path)
    hooks = _section(obj, "hooks", path)
    if hooks is None:
        return text
    commands = frozenset(spec.command for spec in specs.values())
    removed = False
    for event in list(hooks):
        groups = list_at(hooks, event)
        if groups is None or not _strip_commands(groups, commands):
            continue
        removed = True
        if not groups:
            del hooks[event]
    if not removed:
        return text
    if not hooks:
        del obj["hooks"]
    return _finish(obj, layout)


def plugin_manifest() -> JSONObject:
    """The .claude-plugin/plugin.json document."""
    return {
        "name": SERVER_NAME,
        "version": package_version(),
        "description": "ContextWeave code retrieval and session memory",
    }


def mcp_document() -> JSONObject:
    """A complete .mcp.json object registering the server."""
    return {"mcpServers": {SERVER_NAME: dict(SERVER_ENTRY)}}


def routing_skill() -> str:
    """The plugin's routing skill: frontmatter plus the routing block without its markers."""
    body = "\n".join(routing_block().splitlines()[1:-1])
    return (
        "---\n"
        f"name: {SERVER_NAME}-routing\n"
        "description: Route code questions to ContextWeave tools: get_symbol, search_symbols, "
        "file_outline for named things, context_search for described behaviour. Use when "
        "exploring or answering questions about code.\n"
        "---\n\n" + body + "\n"
    )
