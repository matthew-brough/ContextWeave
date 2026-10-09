from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from contextweave.core.refs import ByteAnchor, Ref, RefError, SymbolAnchor, encode_ref
from contextweave.core.registry import Registry
from contextweave.index.records import Resolution
from contextweave.index.store import Row, Store, hash_bytes
from contextweave.retrieve.rows import as_int, as_opt_int, as_str

HASH_LENGTH = 12
MAX_FANOUT = 60
MAX_VISITED = 400
DEFAULT_RESOLUTIONS: tuple[Resolution, ...] = ("exact", "import", "event")
ALL_RESOLUTIONS: tuple[Resolution, ...] = (*DEFAULT_RESOLUTIONS, "name")

type EndpointKind = Literal["symbol", "file", "node"]
type Direction = Literal["out", "in", "both"]


@dataclass(frozen=True, slots=True)
class RelView:
    """How a public rel reads the stored edges: which stored rel, which way, at what level."""

    stored: str
    direction: Direction
    file_level: bool = False


REL_VIEWS: Registry[str, RelView] = Registry()
REL_VIEWS.register("callers", RelView("calls", "in"))
REL_VIEWS.register("callees", RelView("calls", "out"))
REL_VIEWS.register("imports", RelView("imports", "out", file_level=True))
REL_VIEWS.register("importers", RelView("imports", "in", file_level=True))
REL_VIEWS.register("inherits", RelView("inherits", "out"))
REL_VIEWS.register("implements", RelView("implements", "out"))
REL_VIEWS.register("tests", RelView("tests", "in"))
REL_VIEWS.register("config", RelView("config", "both"))
REL_VIEWS.register("emitters", RelView("emits", "in"))
REL_VIEWS.register("listeners", RelView("emits", "out"))


@dataclass(frozen=True, slots=True)
class Endpoint:
    """A node of the edge graph: a symbol, a file seen as its module, or a schema node."""

    kind: EndpointKind
    id: int


@dataclass(frozen=True, slots=True)
class Neighbor:
    """One edge seen from an endpoint; other is None when the target never resolved."""

    rel: str
    other: Endpoint | None
    name: str
    resolution: Resolution
    line: int


@dataclass(frozen=True, slots=True)
class Hop:
    """An endpoint reached by a walk, with its distance and the endpoint it was reached from."""

    endpoint: Endpoint
    hops: int
    via: Endpoint
    rel: str
    resolution: Resolution


def resolutions_for(include_name: bool) -> tuple[Resolution, ...]:
    """The resolution classes a caller may see."""
    return ALL_RESOLUTIONS if include_name else DEFAULT_RESOLUTIONS


def symbol_file(store: Store, symbol_id: int) -> int | None:
    """File id of a symbol."""
    rows = store.query("SELECT file_id FROM symbols WHERE id = ?", (symbol_id,))
    return as_int(rows[0][0]) if rows else None


def node_file(store: Store, node_id: int) -> int | None:
    """File id of a schema node."""
    rows = store.query("SELECT file_id FROM schema_nodes WHERE id = ?", (node_id,))
    return as_int(rows[0][0]) if rows else None


def file_of(store: Store, endpoint: Endpoint) -> int | None:
    """File id an endpoint lives in."""
    match endpoint.kind:
        case "file":
            return endpoint.id
        case "symbol":
            return symbol_file(store, endpoint.id)
        case "node":
            return node_file(store, endpoint.id)


def neighbors(
    store: Store,
    endpoint: Endpoint,
    rel: str,
    resolutions: Sequence[Resolution] = DEFAULT_RESOLUTIONS,
) -> list[Neighbor]:
    """Edges of one public rel around an endpoint, ordered by line then target."""
    view = REL_VIEWS.get(rel)
    start = _lifted(store, endpoint) if view.file_level else endpoint
    if start is None:
        return []
    found: list[Neighbor] = []
    if view.direction in {"out", "both"}:
        found.extend(_outgoing(store, start, rel, view.stored, resolutions))
    if view.direction in {"in", "both"}:
        found.extend(_incoming(store, start, rel, view.stored, resolutions))
    return found[:MAX_FANOUT]


def _lifted(store: Store, endpoint: Endpoint) -> Endpoint | None:
    if endpoint.kind == "file":
        return endpoint
    file_id = file_of(store, endpoint)
    return None if file_id is None else Endpoint("file", file_id)


def _marks(resolutions: Sequence[Resolution]) -> str:
    return ", ".join("?" for _ in resolutions)


def _outgoing(
    store: Store, start: Endpoint, rel: str, stored: str, resolutions: Sequence[Resolution]
) -> list[Neighbor]:
    match start.kind:
        case "symbol":
            where = "from_id = ?"
        case "file":
            where = "from_id IS NULL AND from_file_id = ?"
        case "node":
            return []
    rows = store.query(
        "SELECT to_id, to_file_id, to_node_id, to_name, resolution, line FROM symbol_edges"
        f" WHERE {where} AND rel = ? AND resolution IN ({_marks(resolutions)})"
        " ORDER BY line, to_name, to_id, to_file_id",
        (start.id, stored, *resolutions),
    )
    return [
        Neighbor(rel, _target(row), as_str(row[3]), _resolution(row[4]), as_int(row[5]))
        for row in rows
    ]


def _incoming(
    store: Store, start: Endpoint, rel: str, stored: str, resolutions: Sequence[Resolution]
) -> list[Neighbor]:
    match start.kind:
        case "symbol":
            where = "to_id = ?"
        case "file":
            where = "to_file_id = ? AND to_id IS NULL AND to_node_id IS NULL"
        case "node":
            where = "to_node_id = ?"
    rows = store.query(
        "SELECT from_id, from_file_id, to_name, resolution, line FROM symbol_edges"
        f" WHERE {where} AND rel = ? AND resolution IN ({_marks(resolutions)})"
        " ORDER BY from_file_id, line, from_id",
        (start.id, stored, *resolutions),
    )
    return [
        Neighbor(
            rel,
            Endpoint("file", as_int(row[1]))
            if row[0] is None
            else Endpoint("symbol", as_int(row[0])),
            as_str(row[2]),
            _resolution(row[3]),
            as_int(row[4]),
        )
        for row in rows
    ]


def _target(row: Row) -> Endpoint | None:
    if (to_id := as_opt_int(row[0])) is not None:
        return Endpoint("symbol", to_id)
    if (to_node := as_opt_int(row[2])) is not None:
        return Endpoint("node", to_node)
    if (to_file := as_opt_int(row[1])) is not None:
        return Endpoint("file", to_file)
    return None


def _resolution(value: object) -> Resolution:
    text = as_str(value)
    for known in ALL_RESOLUTIONS:
        if known == text:
            return known
    raise ValueError(f"unknown resolution {text}")


@dataclass(frozen=True, slots=True)
class Link:
    """A traversed edge between two resolved endpoints."""

    source: Endpoint
    rel: str
    target: Endpoint
    resolution: Resolution


@dataclass(frozen=True, slots=True)
class Walk:
    """Endpoints a walk reached, nearest first, and every edge it traversed."""

    hops: list[Hop]
    links: list[Link]


def walk(
    store: Store,
    seeds: Sequence[Endpoint],
    rels: Sequence[str],
    depth: int,
    resolutions: Sequence[Resolution] = DEFAULT_RESOLUTIONS,
) -> Walk:
    """Breadth-first walk from the seeds over the given rels; each endpoint is reached once."""
    seen = set(seeds)
    queue: deque[tuple[Endpoint, int]] = deque((seed, 0) for seed in seeds)
    reached: list[Hop] = []
    links: list[Link] = []
    while queue and len(seen) < MAX_VISITED:
        current, hops = queue.popleft()
        if hops >= depth:
            continue
        for rel in rels:
            if rel not in REL_VIEWS:
                continue
            for edge in neighbors(store, current, rel, resolutions):
                if edge.other is None:
                    continue
                links.append(Link(current, rel, edge.other, edge.resolution))
                if edge.other in seen:
                    continue
                seen.add(edge.other)
                reached.append(Hop(edge.other, hops + 1, current, rel, edge.resolution))
                queue.append((edge.other, hops + 1))
    return Walk(reached, links)


@dataclass(frozen=True, slots=True)
class Placed:
    """An endpoint resolved to the file path and byte range it covers."""

    endpoint: Endpoint
    path: str
    start_byte: int
    end_byte: int
    label: str
    ref: str | None


def place(store: Store, key: str, endpoint: Endpoint) -> Placed | None:
    """Path, byte range, display name and cw:// ref of an endpoint; None when its rows are gone."""
    match endpoint.kind:
        case "symbol":
            rows = store.query(
                "SELECT f.path, s.start_byte, s.end_byte, s.qualified_name, s.content_hash"
                " FROM symbols s JOIN files f ON f.id = s.file_id WHERE s.id = ?",
                (endpoint.id,),
            )
            if not rows:
                return None
            row = rows[0]
            path, start, end = as_str(row[0]), as_int(row[1]), as_int(row[2])
            name = as_str(row[3])
            return Placed(
                endpoint, path, start, end, name, _symbol_ref(key, path, name, start, end, row[4])
            )
        case "file":
            rows = store.query(
                "SELECT path, size, content_hash FROM files WHERE id = ?", (endpoint.id,)
            )
            if not rows:
                return None
            path, size = as_str(rows[0][0]), as_int(rows[0][1])
            digest = as_str(rows[0][2])[:HASH_LENGTH]
            return Placed(
                endpoint, path, 0, size, path, _byte_ref(key, path, 0, max(size, 0), digest)
            )
        case "node":
            return _place_node(store, key, endpoint)


def _place_node(store: Store, key: str, endpoint: Endpoint) -> Placed | None:
    rows = store.query(
        "SELECT f.path, n.start_byte, n.end_byte, n.key_path FROM schema_nodes n"
        " JOIN files f ON f.id = n.file_id WHERE n.id = ?",
        (endpoint.id,),
    )
    if not rows:
        return None
    path, start, end = as_str(rows[0][0]), as_int(rows[0][1]), as_int(rows[0][2])
    label = f"{path}:{as_str(rows[0][3])}"
    try:
        source = (store.root / path).read_bytes()
    except OSError:
        return Placed(endpoint, path, start, end, label, None)
    digest = hash_bytes(source[start:end])[:HASH_LENGTH]
    return Placed(endpoint, path, start, end, label, _byte_ref(key, path, start, end, digest))


def _symbol_ref(
    key: str, path: str, name: str, start: int, end: int, content_hash: object
) -> str | None:
    digest = as_str(content_hash)[:HASH_LENGTH]
    try:
        return encode_ref(Ref(key, path, SymbolAnchor(name), digest))
    except RefError:
        return _byte_ref(key, path, start, end, digest)


def _byte_ref(key: str, path: str, start: int, end: int, digest: str) -> str | None:
    try:
        return encode_ref(Ref(key, path, ByteAnchor(start, end), digest))
    except RefError:
        return None
