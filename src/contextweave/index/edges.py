import json
import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from contextweave.core.config import EventsConfig
from contextweave.core.registry import Registry
from contextweave.core.types import JSON
from contextweave.index.events import emits_pass, load_sides
from contextweave.index.records import EdgeRecord, Resolution
from contextweave.index.resolvers import RESOLVERS, ImportSite, ResolveContext
from contextweave.index.store import Store

NAME_CAP = 8
MAX_BASE_DEPTH = 4
CONFIG_CAP = 4
CALL_KINDS = frozenset({"function", "method", "class", "struct", "handler", "type"})
TYPE_KINDS = frozenset({"class", "struct", "interface", "enum", "type"})
INTERFACE_NAME = re.compile(r"^I[A-Z]")
TABLE_MEMBER = re.compile(r"([\w.]+)[.:](\w+)")
MEMBER_KINDS = frozenset({"function", "method"})
RANK: dict[Resolution, int] = {"exact": 0, "import": 1, "event": 2, "name": 3}


@dataclass(frozen=True, slots=True)
class Sym:
    """A symbol as the resolver sees it."""

    id: int
    file_id: int
    name: str
    qualified_name: str
    kind: str
    parent: str | None
    start_byte: int


@dataclass(frozen=True, slots=True)
class Ref:
    """A stored unresolved reference."""

    file_id: int
    from_id: int | None
    kind: str
    name: str
    qualifier: str | None
    is_self: bool
    line: int


@dataclass(frozen=True, slots=True)
class SchemaKey:
    """A key path of a data file."""

    id: int
    file_id: int
    key_path: str


@dataclass(slots=True)
class Scope:
    """What a file can name without qualification."""

    local: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    opened: set[int] = field(default_factory=set[int])


@dataclass(slots=True)
class Facts:
    """The whole project as plain lookups; edge passes read it and never touch the store."""

    context: ResolveContext
    langs: dict[int, str | None]
    profiles: dict[int, str]
    symbols: dict[int, Sym]
    by_name: dict[str, list[Sym]]
    by_file: dict[int, list[Sym]]
    refs: list[Ref]
    imports: list[ImportSite]
    schema: list[SchemaKey]
    test_globs: tuple[str, ...]
    events: EventsConfig = field(default_factory=EventsConfig)
    handlers: dict[int, str] = field(default_factory=dict[int, str])
    sides: dict[int, frozenset[str]] = field(default_factory=dict[int, frozenset[str]])
    targets: dict[int, tuple[int, ...]] = field(default_factory=dict[int, tuple[int, ...]])
    scopes: dict[int, Scope] = field(default_factory=dict[int, Scope])
    base_map: dict[int, list[Sym]] = field(default_factory=dict[int, list[Sym]])
    bases_built: bool = False
    by_member: dict[str, list[Sym]] = field(default_factory=dict[str, list[Sym]])

    def is_test(self, file_id: int) -> bool:
        """True when the file matches index.test_globs."""
        path = PurePosixPath(self.context.paths[file_id])
        return any(path.full_match(glob) for glob in self.test_globs)

    def scope(self, file_id: int) -> Scope:
        """Names a file binds through its resolved imports and sibling files."""
        cached = self.scopes.get(file_id)
        if cached is not None:
            return cached
        built = Scope()
        for site in self.imports:
            if site.file_id != file_id:
                continue
            files = self.targets.get(site.id, ())
            built.opened.update(files)
            for local in _local_names(site):
                built.local[local].update(files)
        lang = self.langs.get(file_id)
        if lang is not None and lang in RESOLVERS:
            built.opened.update(RESOLVERS.get(lang).siblings(self.context, file_id))
        self.scopes[file_id] = built
        return built

    def bases(self, type_sym: Sym) -> list[Sym]:
        """Resolved direct base types of a class, interface or struct."""
        if not self.bases_built:
            self.bases_built = True
            for ref in self.refs:
                if ref.kind not in {"inherit", "implement", "base"} or ref.from_id is None:
                    continue
                origin = self.symbols.get(ref.from_id)
                for target, resolution in _targets_of(self, ref, origin, TYPE_KINDS):
                    if resolution != "name":
                        self.base_map.setdefault(ref.from_id, []).append(target)
        return self.base_map.get(type_sym.id, [])


type EdgePass = Callable[[Facts, Sequence[EdgeRecord]], list[EdgeRecord]]

EDGE_PASSES: Registry[str, EdgePass] = Registry()


@dataclass(frozen=True, slots=True)
class EdgeReport:
    """Counts of the edges one rebuild produced."""

    edges: int
    unresolved_imports: int


def resolve_edges(
    store: Store,
    test_globs: tuple[str, ...],
    *,
    force: bool = False,
    events: EventsConfig | None = None,
) -> EdgeReport:
    """Rebuild import targets and every derived edge from the stored rows (V3: may lag writes)."""
    with store.writer_lock():
        if store.edges_current and not force:
            count = store.query("SELECT COUNT(*) FROM symbol_edges")
            return EdgeReport(int(str(count[0][0])), 0)
        facts = load_facts(store, test_globs, events or EventsConfig())
        facts.targets = resolve_imports(facts)
        produced: list[EdgeRecord] = []
        for _name, edge_pass in EDGE_PASSES.items():
            produced.extend(edge_pass(facts, tuple(produced)))
        edges = dedupe(produced)
        store.replace_edges(facts.targets, edges, facts.sides)
        unresolved = sum(1 for site in facts.imports if not facts.targets.get(site.id))
        return EdgeReport(len(edges), unresolved)


def load_facts(
    store: Store, test_globs: tuple[str, ...], events: EventsConfig | None = None
) -> Facts:
    """Read every row the edge passes need."""
    paths: dict[int, str] = {}
    langs: dict[int, str | None] = {}
    profiles: dict[int, str] = {}
    for row in store.query("SELECT id, path, lang, profile FROM files"):
        file_id = int(str(row[0]))
        paths[file_id] = str(row[1])
        langs[file_id] = None if row[2] is None else str(row[2])
        profiles[file_id] = str(row[3])
    scopes: dict[str, list[int]] = defaultdict(list)
    file_scopes: dict[int, list[str]] = defaultdict(list)
    for row in store.query("SELECT file_id, scope FROM file_scopes ORDER BY file_id, scope"):
        scopes[str(row[1])].append(int(str(row[0])))
        file_scopes[int(str(row[0]))].append(str(row[1]))
    context = ResolveContext(
        store.root,
        paths,
        {scope: tuple(ids) for scope, ids in scopes.items()},
        {file_id: tuple(found) for file_id, found in file_scopes.items()},
    )
    symbols: dict[int, Sym] = {}
    by_name: dict[str, list[Sym]] = defaultdict(list)
    by_file: dict[int, list[Sym]] = defaultdict(list)
    by_member: dict[str, list[Sym]] = defaultdict(list)
    for row in store.query(
        "SELECT id, file_id, name, qualified_name, kind, parent, start_byte FROM symbols"
        " ORDER BY file_id, start_byte, id"
    ):
        parent = None if row[5] is None else str(row[5])
        sym = Sym(
            int(str(row[0])),
            int(str(row[1])),
            str(row[2]),
            str(row[3]),
            str(row[4]),
            parent,
            int(str(row[6])),
        )
        symbols[sym.id] = sym
        by_name[sym.name].append(sym)
        by_file[sym.file_id].append(sym)
        if sym.kind in MEMBER_KINDS and (split := _table_member(sym.name)) is not None:
            by_member[split[1]].append(sym)
    refs = [
        Ref(
            int(str(row[0])),
            None if row[1] is None else int(str(row[1])),
            str(row[2]),
            str(row[3]),
            None if row[4] is None else str(row[4]),
            bool(row[5]),
            int(str(row[6])),
        )
        for row in store.query(
            "SELECT file_id, from_id, kind, name, qualifier, is_self, line FROM symbol_refs"
            " ORDER BY file_id, start_byte, id"
        )
    ]
    levels = {(r.file_id, r.line): len(r.name) for r in refs if r.kind == "relative"}
    imports = [
        ImportSite(
            int(str(row[0])),
            int(str(row[1])),
            paths[int(str(row[1]))],
            str(row[2]),
            tuple(_names(row[3])),
            None if row[4] is None else str(row[4]),
            str(row[6]),
            int(str(row[5])),
            levels.get((int(str(row[1])), int(str(row[5]))), 0),
        )
        for row in store.query(
            "SELECT id, file_id, module, names, alias, line, kind FROM imports ORDER BY id"
        )
    ]
    schema = [
        SchemaKey(int(str(row[0])), int(str(row[1])), str(row[2]))
        for row in store.query("SELECT id, file_id, key_path FROM schema_nodes ORDER BY id")
    ]
    handlers = {
        int(str(row[0])): str(row[1])
        for row in store.query("SELECT symbol_id, registrar FROM handler_meta")
    }
    return Facts(
        context,
        langs,
        profiles,
        symbols,
        by_name,
        by_file,
        refs,
        imports,
        schema,
        test_globs,
        events or EventsConfig(),
        handlers,
        load_sides(context),
        by_member=by_member,
    )


def _table_member(name: str) -> tuple[str, str] | None:
    """Split a dotted name like Lua's `vRP.log` / `M:save` into (table, member)."""
    found = TABLE_MEMBER.fullmatch(name)
    return None if found is None else (found[1], found[2])


def _names(value: object) -> list[str]:
    loaded: JSON = json.loads(str(value))
    return [item for item in loaded if isinstance(item, str)] if isinstance(loaded, list) else []


def resolve_imports(facts: Facts) -> dict[int, tuple[int, ...]]:
    """Files each import points at, through the resolver registered for its language."""
    targets: dict[int, tuple[int, ...]] = {}
    for site in facts.imports:
        lang = facts.langs.get(site.file_id)
        if lang is None or lang not in RESOLVERS:
            continue
        found = RESOLVERS.get(lang).imports(facts.context, site)
        kept = tuple(file_id for file_id in found if file_id != site.file_id)
        if kept:
            targets[site.id] = kept
    return targets


def _local_names(site: ImportSite) -> set[str]:
    module = site.module
    segments = [part for part in re.split(r"[./\\]", module) if part]
    names = set(site.names)
    if site.alias:
        names.add(site.alias)
    if segments:
        names.update((segments[0], segments[-1]))
    return names


def dedupe(edges: Sequence[EdgeRecord]) -> list[EdgeRecord]:
    """Keep one edge per endpoint pair and rel, with its best resolution and first line."""
    best: dict[tuple[object, ...], EdgeRecord] = {}
    for edge in edges:
        key = (
            edge.from_id,
            edge.from_file_id if edge.from_id is None else None,
            edge.to_id,
            edge.to_file_id if edge.to_id is None else None,
            edge.to_node_id,
            edge.to_name if edge.to_id is None and edge.to_file_id is None else None,
            edge.rel,
        )
        previous = best.get(key)
        if previous is None or RANK[edge.resolution] < RANK[previous.resolution]:
            best[key] = edge
    return list(best.values())


def _grade(found: Sequence[Sym], many: Resolution) -> list[tuple[Sym, Resolution]]:
    single: Resolution = "exact"
    return [(sym, single if len(found) == 1 else many) for sym in found]


def _named(facts: Facts, name: str, kinds: frozenset[str]) -> list[Sym]:
    return [sym for sym in facts.by_name.get(name, ()) if sym.kind in kinds]


def _within(candidates: Sequence[Sym], files: set[int]) -> list[Sym]:
    return [sym for sym in candidates if sym.file_id in files]


def _global_matches(candidates: Sequence[Sym]) -> list[tuple[Sym, Resolution]]:
    return [(sym, "name") for sym in candidates[:NAME_CAP]] if len(candidates) <= NAME_CAP else []


def _targets_of(
    facts: Facts, ref: Ref, origin: Sym | None, kinds: frozenset[str]
) -> list[tuple[Sym, Resolution]]:
    candidates = _named(facts, ref.name, kinds)
    if ref.qualifier is not None and not ref.is_self:
        candidates += [sym for sym in facts.by_member.get(ref.name, ()) if sym.kind in kinds]
    if not candidates:
        return []
    scope = facts.scope(ref.file_id)
    if ref.is_self:
        return _self_targets(facts, ref, origin, candidates)
    if ref.qualifier is not None:
        return _qualified_targets(facts, ref, scope, candidates)
    own = [
        sym
        for sym in candidates
        if sym.file_id == ref.file_id
        and sym.id != (origin.id if origin and ref.kind != "call" else -1)
        and (
            ref.kind != "call"
            or sym.parent is None
            or origin is None
            or sym.parent == origin.parent
        )
    ]
    if own:
        return _grade(own, "name")
    bound = scope.local.get(ref.name)
    if bound:
        found = _within(candidates, bound)
        if found:
            return _grade(found, "import")
    opened = _within(candidates, scope.opened)
    if opened:
        return _grade(opened, "import")
    return _global_matches(candidates)


def _self_targets(
    facts: Facts, ref: Ref, origin: Sym | None, candidates: Sequence[Sym]
) -> list[tuple[Sym, Resolution]]:
    owner = _owner_of(facts, ref.file_id, origin)
    if owner is None:
        same = [sym for sym in candidates if sym.file_id == ref.file_id]
        return _grade(same, "name") if same else _global_matches(candidates)
    seen: set[int] = set()
    frontier = [owner]
    for _depth in range(MAX_BASE_DEPTH):
        members = [
            sym
            for type_sym in frontier
            for sym in candidates
            if sym.file_id == type_sym.file_id and sym.parent == type_sym.qualified_name
        ]
        if members:
            return _grade(members, "name")
        seen.update(type_sym.id for type_sym in frontier)
        frontier = [
            base for type_sym in frontier for base in facts.bases(type_sym) if base.id not in seen
        ]
        if not frontier:
            break
    return _global_matches(candidates)


def _owner_of(facts: Facts, file_id: int, origin: Sym | None) -> Sym | None:
    if origin is None:
        return None
    if origin.kind in TYPE_KINDS:
        return origin
    return next(
        (
            sym
            for sym in facts.by_file.get(file_id, ())
            if sym.kind in TYPE_KINDS and sym.qualified_name == origin.parent
        ),
        None,
    )


def _qualified_targets(
    facts: Facts, ref: Ref, scope: Scope, candidates: Sequence[Sym]
) -> list[tuple[Sym, Resolution]]:
    qualifier = ref.qualifier or ""
    head = re.split(r"[./\\]", qualifier)[0]
    bound = scope.local.get(head)
    if bound:
        return _grade(_within(candidates, bound), "import")
    tail = re.split(r"[.:]", qualifier)[-1]
    owned = [
        sym
        for sym in candidates
        if (owner := _owner_name(sym)) is not None
        and (owner == qualifier or owner.split(".")[-1] == tail)
    ]
    same = [sym for sym in owned if sym.file_id == ref.file_id]
    if same:
        return _grade(same, "name")
    named = [sym for sym in owned if _owner_name(sym) == qualifier]
    if len(named) == 1:
        return _grade(named, "name")
    if owned:
        return _global_matches(owned)
    typed = [sym for sym in facts.by_name.get(tail, ()) if sym.kind in TYPE_KINDS]
    if typed:
        return []
    return _global_matches([sym for sym in candidates if sym.kind in {"function", "method"}])


def _owner_name(sym: Sym) -> str | None:
    if sym.parent is not None:
        return sym.parent
    split = _table_member(sym.name)
    return None if split is None else split[0]


def _edge(
    ref: Ref, rel: str, target: Sym | None, resolution: Resolution, name: str | None = None
) -> EdgeRecord:
    return EdgeRecord(
        from_id=ref.from_id,
        from_file_id=ref.file_id,
        to_id=None if target is None else target.id,
        to_file_id=None if target is None else target.file_id,
        to_node_id=None,
        to_name=name or (target.qualified_name if target else ref.name),
        rel=rel,
        resolution=resolution,
        line=ref.line,
    )


def _imports_pass(facts: Facts, _prior: Sequence[EdgeRecord]) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []
    for site in facts.imports:
        files = facts.targets.get(site.id, ())
        resolution: Resolution = "exact" if len(files) == 1 else "import"
        edges.extend(
            EdgeRecord(
                None,
                site.file_id,
                None,
                file_id,
                None,
                site.module,
                "imports",
                resolution,
                site.line,
            )
            for file_id in files
        )
    return edges


def _refs_pass(facts: Facts, _prior: Sequence[EdgeRecord]) -> list[EdgeRecord]:
    edges: list[EdgeRecord] = []
    for ref in facts.refs:
        origin = facts.symbols.get(ref.from_id) if ref.from_id is not None else None
        if ref.kind == "call":
            edges.extend(
                _edge(ref, "calls", target, resolution)
                for target, resolution in _targets_of(facts, ref, origin, CALL_KINDS)
            )
        elif ref.kind in {"inherit", "implement", "base"}:
            edges.extend(_type_edges(facts, ref, origin))
    return edges


def _type_edges(facts: Facts, ref: Ref, origin: Sym | None) -> list[EdgeRecord]:
    found = _targets_of(facts, ref, origin, TYPE_KINDS)
    if found:
        return [
            _edge(ref, _type_rel(ref, target.kind, target.name), target, res)
            for target, res in found
        ]
    shown = f"{ref.qualifier}.{ref.name}" if ref.qualifier else ref.name
    return [_edge(ref, _type_rel(ref, None, ref.name), None, "name", shown)]


def _type_rel(ref: Ref, kind: str | None, name: str) -> str:
    if ref.kind == "inherit":
        return "inherits"
    if ref.kind == "implement":
        return "implements"
    if kind is not None:
        return "implements" if kind == "interface" else "inherits"
    return "implements" if INTERFACE_NAME.match(name) else "inherits"


def _tests_pass(facts: Facts, prior: Sequence[EdgeRecord]) -> list[EdgeRecord]:
    tests: list[EdgeRecord] = []
    for edge in prior:
        if edge.rel not in {"calls", "imports"} or not facts.is_test(edge.from_file_id):
            continue
        if edge.to_file_id is None or facts.is_test(edge.to_file_id):
            continue
        tests.append(replace_rel(edge, "tests"))
    return tests


def replace_rel(edge: EdgeRecord, rel: str) -> EdgeRecord:
    """Copy of an edge under another rel."""
    return EdgeRecord(
        edge.from_id,
        edge.from_file_id,
        edge.to_id,
        edge.to_file_id,
        edge.to_node_id,
        edge.to_name,
        rel,
        edge.resolution,
        edge.line,
        edge.detail,
    )


def _config_pass(facts: Facts, _prior: Sequence[EdgeRecord]) -> list[EdgeRecord]:
    keys: dict[str, list[SchemaKey]] = defaultdict(list)
    for key in facts.schema:
        if "." in key.key_path:
            keys[key.key_path].append(key)
    config_files = {
        path: file_id
        for file_id, path in facts.context.paths.items()
        if facts.profiles.get(file_id) == "schema"
    }
    edges: list[EdgeRecord] = []
    for ref in facts.refs:
        if ref.kind != "string":
            continue
        edges.extend(_config_edges(facts, ref, keys, config_files))
    return edges


def _config_edges(
    facts: Facts, ref: Ref, keys: dict[str, list[SchemaKey]], config_files: dict[str, int]
) -> list[EdgeRecord]:
    found: list[EdgeRecord] = []
    for key in keys.get(ref.name, ())[:CONFIG_CAP]:
        found.append(
            EdgeRecord(
                ref.from_id,
                ref.file_id,
                None,
                key.file_id,
                key.id,
                ref.name,
                "config",
                "name",
                ref.line,
            )
        )
    matches = [
        file_id
        for path, file_id in sorted(config_files.items())
        if path == ref.name or path.endswith("/" + ref.name)
    ]
    found.extend(
        EdgeRecord(
            ref.from_id, ref.file_id, None, file_id, None, ref.name, "config", "name", ref.line
        )
        for file_id in matches[:CONFIG_CAP]
        if file_id != ref.file_id
    )
    return found


EDGE_PASSES.register("imports", _imports_pass)
EDGE_PASSES.register("refs", _refs_pass)
EDGE_PASSES.register("tests", _tests_pass)
EDGE_PASSES.register("config", _config_pass)
EDGE_PASSES.register("emits", emits_pass)
