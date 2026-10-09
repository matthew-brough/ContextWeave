import json
import posixpath
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from contextweave.index.records import EdgeRecord
from contextweave.index.resolvers import ResolveContext

if TYPE_CHECKING:
    from contextweave.index.edges import Facts, Sym

MANIFEST_NAMES = frozenset({"fxmanifest.lua", "__resource.lua"})
SCRIPT_CALL = re.compile(
    r"\b(client|server|shared)_scripts?\b\s*\(?\s*(\{[^}]*\}|'[^']*'|\"[^\"]*\")", re.DOTALL
)
QUOTED = re.compile(r"""['"]([^'"]+)['"]""")
BLOCK_COMMENT = re.compile(r"--\[(=*)\[.*?\]\1\]", re.DOTALL)
LINE_COMMENT = re.compile(r"--[^\n]*")
OPPOSITE = {"client": "server", "server": "client"}
NET_FLIP = "emitNet"


def parse_manifest(text: str) -> dict[str, tuple[str, ...]]:
    """Script globs per side declared by a resource manifest; other-resource references dropped."""
    bare = LINE_COMMENT.sub("", BLOCK_COMMENT.sub("", text))
    found: dict[str, list[str]] = defaultdict(list)
    for match in SCRIPT_CALL.finditer(bare):
        globs = [
            glob.removeprefix("./")
            for glob in QUOTED.findall(match.group(2))
            if not glob.startswith("@")
        ]
        found[match.group(1)].extend(globs)
    return {side: tuple(globs) for side, globs in found.items()}


def load_sides(context: ResolveContext) -> dict[int, frozenset[str]]:
    """Client, server or shared side of every file a resource manifest covers."""
    sides: dict[int, set[str]] = defaultdict(set)
    for manifest_id, manifest_path in sorted(context.paths.items()):
        if posixpath.basename(manifest_path) not in MANIFEST_NAMES:
            continue
        text = context.text(manifest_path)
        if text is None:
            continue
        directory = posixpath.dirname(manifest_path)
        declared = parse_manifest(text)
        for file_id, path in context.paths.items():
            relative = _relative_to(directory, path)
            if relative is None or file_id == manifest_id:
                continue
            for side, globs in declared.items():
                if any(PurePosixPath(relative).full_match(glob) for glob in globs):
                    sides[file_id].add(side)
    return {file_id: frozenset(found) for file_id, found in sides.items()}


def _relative_to(directory: str, path: str) -> str | None:
    if not directory:
        return path
    prefix = directory + "/"
    return path.removeprefix(prefix) if path.startswith(prefix) else None


def target_sides(callee: str, source: frozenset[str]) -> frozenset[str] | None:
    """Handler sides an emitter can reach; None when nothing narrows it."""
    if callee == NET_FLIP:
        flipped = {OPPOSITE[side] for side in source if side in OPPOSITE}
        return None if not flipped or "shared" in source else frozenset({*flipped, "shared"})
    if "Server" in callee:
        return frozenset({"server", "shared"})
    if "Client" in callee:
        return frozenset({"client", "shared"})
    if not source or "shared" in source:
        return None
    return frozenset({*source, "shared"})


def is_net(callee: str) -> bool:
    """True when the emitter crosses the client/server boundary."""
    return callee == NET_FLIP or "Server" in callee or "Client" in callee


def _admits(handler_sides: frozenset[str] | None, allowed: frozenset[str] | None) -> bool:
    return not handler_sides or allowed is None or bool(handler_sides & allowed)


def emits_pass(facts: Facts, _prior: Sequence[EdgeRecord]) -> list[EdgeRecord]:
    """Emitter to handler edges by exact literal event name, narrowed by manifest side."""
    registrars = frozenset(facts.events.registrars)
    emitters = frozenset(facts.events.emitters)
    handlers = _handlers_by_event(facts, registrars)
    edges: list[EdgeRecord] = []
    for ref in facts.refs:
        if ref.kind != "emit" or ref.qualifier not in emitters:
            continue
        source = facts.sides.get(ref.file_id, frozenset())
        allowed = target_sides(ref.qualifier, source)
        for handler in handlers.get(ref.name, ()):
            handler_sides = facts.sides.get(handler.file_id)
            if not _admits(handler_sides, allowed):
                continue
            edges.append(
                EdgeRecord(
                    ref.from_id,
                    ref.file_id,
                    handler.id,
                    handler.file_id,
                    None,
                    handler.qualified_name,
                    "emits",
                    "event",
                    ref.line,
                    _detail(ref.name, ref.qualifier, source, handler_sides),
                )
            )
    return edges


def _handlers_by_event(facts: Facts, registrars: frozenset[str]) -> Mapping[str, list[Sym]]:
    by_event: dict[str, list[Sym]] = defaultdict(list)
    for symbol_id, registrar in facts.handlers.items():
        symbol = facts.symbols.get(symbol_id)
        if symbol is not None and registrar in registrars:
            by_event[symbol.name].append(symbol)
    return by_event


def _detail(
    event: str, callee: str, source: frozenset[str], handler_sides: frozenset[str] | None
) -> str:
    return json.dumps(
        {
            "event": event,
            "net": is_net(callee),
            "sides": {"from": sorted(source), "to": sorted(handler_sides or ())},
        },
        sort_keys=True,
    )
