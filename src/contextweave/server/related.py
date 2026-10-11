from collections.abc import Sequence

from contextweave.core.errors import Envelope, ErrorEntry, OmittedEntry
from contextweave.core.project import ProjectEcho
from contextweave.core.refs import ByteAnchor, PartAnchor, RefError, SymbolAnchor, parse_ref
from contextweave.retrieve.graph import (
    Endpoint,
    Hop,
    Placed,
    neighbors,
    place,
    resolutions_for,
    walk,
)
from contextweave.retrieve.rows import as_group, as_int, as_profile, as_str
from contextweave.server.fit import Call, Omitted, fit_items
from contextweave.server.payloads import (
    Candidate,
    Common,
    RelatedArgs,
    RelatedEdge,
    RelatedPayload,
)
from contextweave.server.render import head_lines

DEFAULT_DEPTH = 1
CANDIDATE_LIMIT = 20
EDGE_LIMIT = 200


def _tail(payload: Common) -> list[str]:
    lines = [f"!{error['code']} {error.get('message', '')}".rstrip() for error in payload["errors"]]
    lines.extend(f"~{entry['stage']} {entry['reason']}" for entry in payload["degraded"])
    for entry in payload.get("omitted", []):
        if "qualified_name" in entry:
            lines.append(f"-omitted {entry['qualified_name']} {entry['ref']}")
        else:
            lines.append(f"-omitted {entry['count']} more {entry['ref']}")
    if "hint" in payload:
        lines.append(f"hint: {payload['hint']}")
    return lines


def _edge_where(edge: RelatedEdge) -> str:
    if "path" in edge:
        return edge["path"]
    if "ref" in edge:
        try:
            return parse_ref(edge["ref"]).path
        except RefError:
            return edge["ref"]
    return ""


def render_related(payload: RelatedPayload) -> str:
    """Agent-facing text: one line per edge, candidates when the target was ambiguous."""
    lines = head_lines(payload)
    if not payload["found"] and not payload.get("candidates"):
        lines.append("not found")
    for item in payload.get("candidates", []):
        lines.append(
            f"? {item['kind']} {item['qualified_name']} {item['path']}:{item['start_line']}"
        )
    if payload["found"] and not payload["edges"]:
        lines.append(f"0 {payload['rel']}")
    for edge in payload["edges"]:
        resolution = "" if edge["resolution"] == "exact" else f" [{edge['resolution']}]"
        lines.append(
            f"{'  ' * (edge['hops'] - 1)}{edge['rel']} {edge['name']}{resolution} {_edge_where(edge)}".rstrip()
        )
    lines.extend(_tail(payload))
    return "\n".join(lines)


def _payload(
    call: Call[RelatedPayload],
    target: str,
    rel: str,
    found: bool,
    ambiguous: bool,
    edges: Sequence[RelatedEdge],
    candidates: Sequence[Candidate] | None,
    omitted: Omitted,
    hint: str | None,
) -> RelatedPayload:
    payload = RelatedPayload(
        project=call.echo,
        errors=call.env.errors,
        degraded=call.env.degraded,
        truncated=hint is not None,
        target=target,
        rel=rel,
        found=found,
        ambiguous=ambiguous,
        edges=list(edges),
    )
    if candidates is not None:
        payload["candidates"] = list(candidates)
    if omitted:
        payload["omitted"] = omitted
    if hint is not None:
        payload["hint"] = hint
    return payload


def empty_related(echo: ProjectEcho | None, env: Envelope) -> RelatedPayload:
    return RelatedPayload(
        project=echo,
        errors=env.errors,
        degraded=env.degraded,
        truncated=False,
        target="",
        rel="",
        found=False,
        ambiguous=False,
        edges=[],
    )


def _symbols_named(call: Call[RelatedPayload], name: str) -> list[int]:
    store = call.snap.store
    for column in ("qualified_name", "name"):
        rows = store.query(
            f"SELECT s.id FROM symbols s JOIN files f ON f.id = s.file_id"
            f" WHERE s.{column} = ? ORDER BY f.path, s.start_byte LIMIT ?",
            (name, CANDIDATE_LIMIT),
        )
        if rows:
            return [as_int(row[0]) for row in rows]
    return []


def _file_id(call: Call[RelatedPayload], raw: str) -> int | None:
    path = call.snap.normalize(raw) if "/" in raw or "." in raw else None
    if path is None:
        return None
    rows = call.snap.store.query("SELECT id FROM files WHERE path = ?", (path,))
    return as_int(rows[0][0]) if rows else None


def _ref_targets(call: Call[RelatedPayload], raw: str) -> list[Endpoint] | None:
    try:
        ref = parse_ref(raw)
    except RefError as exc:
        call.env.add_error(ErrorEntry(code="bad_ref", message=str(exc), stage="related"))
        return None
    store = call.snap.store
    match ref.anchor:
        case SymbolAnchor(name) | PartAnchor(name, _):
            rows = store.query(
                "SELECT s.id FROM symbols s JOIN files f ON f.id = s.file_id"
                " WHERE f.path = ? AND s.qualified_name = ? ORDER BY s.start_byte",
                (ref.path, name),
            )
            return [Endpoint("symbol", as_int(row[0])) for row in rows]
        case ByteAnchor(start, end):
            rows = store.query(
                "SELECT s.id FROM symbols s JOIN files f ON f.id = s.file_id"
                " WHERE f.path = ? AND s.start_byte = ? AND s.end_byte = ?",
                (ref.path, start, end),
            )
            if rows:
                return [Endpoint("symbol", as_int(row[0])) for row in rows]
            files = store.query("SELECT id FROM files WHERE path = ?", (ref.path,))
            return [Endpoint("file", as_int(files[0][0]))] if files else []


def _targets(call: Call[RelatedPayload], raw: str) -> list[Endpoint] | None:
    if raw.startswith("cw://"):
        return _ref_targets(call, raw)
    named = [Endpoint("symbol", symbol_id) for symbol_id in _symbols_named(call, raw)]
    if named:
        return named
    file_id = _file_id(call, raw)
    return [] if file_id is None else [Endpoint("file", file_id)]


def _candidate(call: Call[RelatedPayload], placed: Placed) -> Candidate | None:
    if placed.endpoint.kind != "symbol":
        return None
    rows = call.snap.store.query(
        "SELECT s.kind, s.start_line, f.profile, f.grp FROM symbols s"
        " JOIN files f ON f.id = s.file_id WHERE s.id = ?",
        (placed.endpoint.id,),
    )
    return Candidate(
        ref=placed.ref or "",
        qualified_name=placed.label,
        kind=as_str(rows[0][0]),
        profile=as_profile(rows[0][2]),
        group=as_group(rows[0][3]),
        path=placed.path,
        start_line=as_int(rows[0][1]),
    )


def _edge(call: Call[RelatedPayload], hop: Hop) -> RelatedEdge | None:
    placed = place(call.snap.store, call.project.key, hop.endpoint)
    if placed is None:
        return None
    edge = RelatedEdge(
        rel=hop.rel,
        resolution=hop.resolution,
        hops=hop.hops,
        name=placed.label,
        path=placed.path,
    )
    if placed.ref is not None:
        edge["ref"] = placed.ref
    return edge


def _unresolved(
    call: Call[RelatedPayload], endpoint: Endpoint, args: RelatedArgs
) -> list[RelatedEdge]:
    found = neighbors(
        call.snap.store,
        endpoint,
        args["rel"],
        resolutions_for(args.get("include_name_edges", False)),
    )
    return [
        RelatedEdge(rel=edge.rel, resolution=edge.resolution, hops=1, name=edge.name)
        for edge in found
        if edge.other is None and edge.name
    ]


def _omit(edge: RelatedEdge) -> OmittedEntry:
    return OmittedEntry(
        ref=edge.get("ref", ""),
        qualified_name=edge["name"],
        path=edge.get("path", ""),
        reason="budget",
    )


def related_body(call: Call[RelatedPayload], args: RelatedArgs) -> RelatedPayload:
    """Edges of one rel around a target; ambiguous targets list candidates instead (V8)."""
    raw = args["target"]
    targets = _targets(call, raw)
    if not targets:
        payload = _payload(call, raw, args["rel"], False, False, [], None, [], None)
        if targets is not None:
            payload["hint"] = "no such symbol or file; try search_symbols"
        return payload
    if len(targets) > 1:
        places = [place(call.snap.store, call.project.key, t) for t in targets]
        candidates = [c for p in places if p is not None and (c := _candidate(call, p)) is not None]
        return fit_items(
            call,
            candidates,
            lambda kept, omitted, hint: _payload(
                call, raw, args["rel"], False, True, [], kept, omitted, hint
            ),
            lambda item: OmittedEntry(
                ref=item["ref"],
                qualified_name=item["qualified_name"],
                path=item["path"],
                reason="budget",
            ),
            "pass a ref or a path-qualified name to pick one candidate",
        )
    seed = targets[0]
    depth = args.get("depth", DEFAULT_DEPTH)
    reached = walk(
        call.snap.store,
        [seed],
        [args["rel"]],
        depth,
        resolutions_for(args.get("include_name_edges", False)),
    )
    edges = [e for hop in reached.hops if (e := _edge(call, hop)) is not None][:EDGE_LIMIT]
    edges.extend(_unresolved(call, seed, args))
    placed = place(call.snap.store, call.project.key, seed)
    shown = placed.ref if placed is not None and placed.ref is not None else raw
    return fit_items(
        call,
        edges,
        lambda kept, omitted, hint: _payload(
            call, shown, args["rel"], True, False, kept, None, omitted, hint
        ),
        _omit,
        "lower depth or pass max_tokens to see more",
    )
