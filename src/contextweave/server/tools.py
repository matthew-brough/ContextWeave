import fnmatch
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import TypeAdapter, ValidationError
from tree_sitter import Query, QueryError

from contextweave.core.budget import Ceiling, measure_payload
from contextweave.core.errors import DegradedEntry, Envelope, ErrorEntry, OmittedEntry
from contextweave.core.project import ProjectEcho, ResolvedProject, resolve_project
from contextweave.core.refs import ByteAnchor, PartAnchor, Ref, RefError, SymbolAnchor, parse_ref
from contextweave.core.registry import Registry
from contextweave.core.types import JSON
from contextweave.lang.parse import parse_bytes
from contextweave.lang.registry import Group, load_language
from contextweave.lang.ts import run_matches
from contextweave.server.context import ToolContext
from contextweave.server.fit import Call, Clip, Omitted, fit_items, fit_text
from contextweave.server.nodes import Node, short_hash
from contextweave.server.payloads import (
    AstQueryArgs,
    AstQueryPayload,
    Candidate,
    ExpandArgs,
    ExpandPayload,
    FileOutlineArgs,
    FileOutlinePayload,
    GetSymbolArgs,
    GetSymbolPayload,
    Hit,
    ImportItem,
    Match,
    OutlineNode,
    SearchSymbolsArgs,
    SearchSymbolsPayload,
    SymbolItem,
    SymbolView,
    json_schema,
)
from contextweave.server.queries import SearchRow
from contextweave.server.render import (
    render_ast_query,
    render_expand,
    render_file_outline,
    render_get_symbol,
    render_search_symbols,
)
from contextweave.server.snapshot import Snapshot, lang_error_entry

MAX_LINKS = 50
CANDIDATE_LIMIT = 200
DEFAULT_SEARCH_LIMIT = 20
SCHEMA_DEPTH = 4
DEFAULT_DEPTH = 2

_JSON_OBJECT: TypeAdapter[dict[str, JSON]] = TypeAdapter(dict[str, JSON])


@dataclass(frozen=True, slots=True)
class ToolResult:
    """A finished tool response: structured payload, text fallback and ref links."""

    structured: dict[str, JSON]
    text: str
    links: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A registered tool as the transport sees it."""

    name: str
    title: str
    description: str
    input_schema: dict[str, JSON]
    output_schema: dict[str, JSON]
    run: Callable[[ToolContext, Mapping[str, JSON]], ToolResult]


@dataclass(frozen=True, slots=True)
class Tool[A: Mapping[str, object], P: Mapping[str, object]]:
    """Definition of one tool: argument and payload types, body, empty payload and renderer."""

    name: str
    title: str
    description: str
    args_type: type[A]
    payload_type: type[P]
    body: Callable[[Call[P], A], P]
    empty: Callable[[ProjectEcho | None, Envelope], P]
    render: Callable[[P], str]


TOOLS: Registry[str, ToolSpec] = Registry()


def _links(structured: JSON, found: dict[str, None]) -> None:
    if isinstance(structured, dict):
        for key, value in structured.items():
            if key in ("ref", "current") and isinstance(value, str) and value.startswith("cw://"):
                found.setdefault(value, None)
            else:
                _links(value, found)
    elif isinstance(structured, list):
        for item in structured:
            _links(item, found)


def collect_links(structured: dict[str, JSON]) -> tuple[str, ...]:
    """Distinct refs mentioned in a payload, capped for resource_link emission."""
    found: dict[str, None] = {}
    _links(structured, found)
    return tuple(found)[:MAX_LINKS]


def _first_problem(exc: ValidationError) -> str:
    details = exc.errors(include_url=False, include_context=False, include_input=False)
    if not details:
        return "invalid arguments"
    first = details[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}" if location else first["msg"]


def define[A: Mapping[str, object], P: Mapping[str, object]](tool: Tool[A, P]) -> ToolSpec:
    """Build and register the transport-facing spec of a tool."""
    args_adapter = TypeAdapter(tool.args_type)
    payload_adapter = TypeAdapter(tool.payload_type)

    def dump(payload: P) -> dict[str, JSON]:
        return _JSON_OBJECT.validate_json(payload_adapter.dump_json(payload))

    def measure(payload: P) -> int:
        return measure_payload(dump(payload), tool.render(payload))

    def finish(payload: P) -> ToolResult:
        structured = dump(payload)
        return ToolResult(structured, tool.render(payload), collect_links(structured))

    def fail(env: Envelope, project: ProjectEcho | None, entry: ErrorEntry) -> ToolResult:
        env.add_error(entry)
        return finish(tool.empty(project, env))

    def execute(ctx: ToolContext, raw: Mapping[str, JSON]) -> ToolResult:
        env = Envelope()
        try:
            return _execute(ctx, raw, env)
        except Exception as exc:
            return fail(
                env,
                None,
                ErrorEntry(
                    code="internal_error",
                    message=f"{type(exc).__name__}: {exc}",
                    stage=tool.name,
                ),
            )

    def _execute(ctx: ToolContext, raw: Mapping[str, JSON], env: Envelope) -> ToolResult:
        try:
            args = args_adapter.validate_python(raw)
        except ValidationError as exc:
            return fail(env, None, ErrorEntry(code="bad_args", message=_first_problem(exc)))
        resolution = resolve_project(raw, ctx.roots, ctx.cwd, ctx.registry.known_roots())
        if not isinstance(resolution, ResolvedProject):
            return fail(env, None, resolution)
        echo = resolution.echo()
        requested = args.get("max_tokens")
        limit = (
            requested
            if isinstance(requested, int) and requested > 0
            else ctx.config.budget.default_max_tokens
        )
        try:
            store = ctx.open_store(resolution.root)
        except Exception as exc:
            return fail(
                env,
                echo,
                ErrorEntry(
                    code="store_unavailable",
                    message=f"{type(exc).__name__}: {exc}",
                    stage="open",
                ),
            )
        with store:
            snap = Snapshot(store, env, resolution.key)
            try:
                call = Call(ctx, resolution, snap, env, Ceiling(limit), measure)
                try:
                    payload = tool.body(call, args)
                except Exception as exc:
                    env.add_error(
                        ErrorEntry(
                            code="internal_error",
                            message=f"{type(exc).__name__}: {exc}",
                            stage=tool.name,
                        )
                    )
                    payload = tool.empty(echo, env)
                return finish(payload)
            finally:
                snap.close()

    spec = ToolSpec(
        tool.name,
        tool.title,
        tool.description,
        json_schema(tool.args_type, mode="validation"),
        json_schema(tool.payload_type, mode="serialization"),
        execute,
    )
    return TOOLS.register(tool.name, spec)


def _omit(ref: str, qualified_name: str, path: str) -> OmittedEntry:
    return OmittedEntry(ref=ref, qualified_name=qualified_name, path=path, reason="budget")


def _rank(row: SearchRow, needle: str) -> int:
    if row.name.lower() == needle:
        return 0
    if row.name.lower().startswith(needle):
        return 1
    if row.qualified_name.lower() == needle:
        return 2
    return 3


def search_nodes(
    snap: Snapshot,
    query: str,
    *,
    kind: str | None = None,
    lang: str | None = None,
    group: Group | None = None,
    path_glob: str | None = None,
    only_path: str | None = None,
    limit: int | None = None,
) -> tuple[list[Node], int]:
    """Ranked nodes whose name or qualified name contains query, and the total match count."""
    needle = query.lower()
    if only_path is None:
        snap.sync_all()
    else:
        snap.sync(only_path)
    rows = [
        row
        for row in snap.queries.search(needle, kind=kind, lang=lang, group=group)
        if row.path not in snap.overlay
    ]
    for path in sorted(snap.overlay):
        for node in snap.nodes(path):
            if (
                (kind is None or node.kind == kind)
                and (lang is None or node.lang == lang)
                and (group is None or node.group == group)
            ):
                rows.append(
                    SearchRow(
                        path,
                        node.name,
                        node.qualified_name,
                        node.kind,
                        node.start_byte,
                        node.start_line,
                        node.lang,
                        node.group,
                    )
                )
    rows = [
        row
        for row in rows
        if (needle in row.name.lower() or needle in row.qualified_name.lower())
        and (only_path is None or row.path == only_path)
        and (path_glob is None or fnmatch.fnmatch(row.path, path_glob))
    ]
    rows.sort(key=lambda row: (_rank(row, needle), len(row.path), row.path, row.start_line))
    total = len(rows)
    selected = rows if limit is None else rows[:limit]
    found: list[Node] = []
    for row in selected:
        index = {
            (node.start_byte, node.qualified_name, node.kind): node for node in snap.nodes(row.path)
        }
        node = index.get((row.start_byte, row.qualified_name, row.kind))
        if node is not None:
            found.append(node)
    return found, total


def _text(snap: Snapshot, node: Node) -> str:
    return snap.source(node.path)[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _symbol_item(call: Call[GetSymbolPayload], node: Node, view: SymbolView) -> SymbolItem:
    item = SymbolItem(
        ref=call.snap.symbol_ref(node),
        name=node.name,
        qualified_name=node.qualified_name,
        kind=node.kind,
        lang=node.lang,
        group=node.group,
        profile=node.profile,
        path=node.path,
        start_line=node.start_line,
        end_line=node.end_line,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
    )
    if view == "annotation":
        item["annotation"] = None
        return item
    item["signature"] = node.signature
    if view == "doc":
        item["docstring"] = node.docstring
    if view == "source":
        item["source"] = _text(call.snap, node)
    return item


def _get_symbol_payload(
    call: Call[GetSymbolPayload],
    found: bool,
    ambiguous: bool,
    symbols: Sequence[SymbolItem],
    total: int,
    candidates: Sequence[Candidate] | None,
    omitted: Omitted,
    hint: str | None,
) -> GetSymbolPayload:
    payload = GetSymbolPayload(
        project=call.echo,
        errors=call.env.errors,
        degraded=call.env.degraded,
        truncated=hint is not None,
        found=found,
        ambiguous=ambiguous,
        symbols=list(symbols),
        total_matches=total,
    )
    if candidates is not None:
        payload["candidates"] = list(candidates)
    if omitted:
        payload["omitted"] = omitted
    if hint is not None:
        payload["hint"] = hint
    return payload


def _clip_symbol(item: SymbolItem, length: int) -> SymbolItem:
    clipped = item.copy()
    clipped["source"] = item.get("source", "")[:length]
    return clipped


def _select_exact(nodes: list[Node], name: str) -> list[Node]:
    needle = name.lower()
    exact = [
        node
        for node in nodes
        if node.qualified_name.lower() == needle or node.name.lower() == needle
    ]
    return exact or nodes


def _get_symbol(call: Call[GetSymbolPayload], args: GetSymbolArgs) -> GetSymbolPayload:
    view: SymbolView = args.get("view", "source")
    only_path: str | None = None
    if "path" in args:
        only_path = call.snap.normalize(args["path"])
        if only_path is None:
            return _get_symbol_payload(call, False, False, [], 0, None, [], None)
    nodes, total = search_nodes(call.snap, args["name"], only_path=only_path, limit=CANDIDATE_LIMIT)
    chosen = _select_exact(nodes, args["name"])
    if "line" in args:
        line = args["line"]
        at_line = [node for node in chosen if node.start_line == line] or [
            node for node in chosen if node.start_line <= line <= node.end_line
        ]
        chosen = at_line
    if view == "annotation":
        call.env.add_degraded(
            DegradedEntry(stage="annotate", reason="annotations are not available yet")
        )
    if not chosen:
        payload = _get_symbol_payload(call, False, False, [], total, None, [], None)
        payload["hint"] = "no match; try search_symbols with a shorter query"
        return payload
    if len(chosen) > 1 and len({(node.name, node.kind) for node in chosen}) > 1:
        candidates = [
            Candidate(
                ref=call.snap.symbol_ref(node),
                qualified_name=node.qualified_name,
                kind=node.kind,
                profile=node.profile,
                group=node.group,
                path=node.path,
                start_line=node.start_line,
            )
            for node in chosen
        ]
        return fit_items(
            call,
            candidates,
            lambda kept, omitted, hint: _get_symbol_payload(
                call, False, True, [], total, kept, omitted, hint
            ),
            lambda item: _omit(item["ref"], item["qualified_name"], item["path"]),
            "pass path or line to pick one candidate",
        )
    items = [_symbol_item(call, node, view) for node in chosen]
    return fit_items(
        call,
        items,
        lambda kept, omitted, hint: _get_symbol_payload(
            call, True, False, kept, total, None, omitted, hint
        ),
        lambda item: _omit(item["ref"], item["qualified_name"], item["path"]),
        "pass path or line, or view=signature, to narrow",
        Clip(lambda item: len(item.get("source", "")), _clip_symbol),
    )


def _empty_symbol(echo: ProjectEcho | None, env: Envelope) -> GetSymbolPayload:
    return GetSymbolPayload(
        project=echo,
        errors=env.errors,
        degraded=env.degraded,
        truncated=False,
        found=False,
        ambiguous=False,
        symbols=[],
    )


def _hit(snap: Snapshot, node: Node) -> Hit:
    hit = Hit(
        ref=snap.symbol_ref(node),
        name=node.name,
        qualified_name=node.qualified_name,
        kind=node.kind,
        lang=node.lang,
        group=node.group,
        profile=node.profile,
        path=node.path,
        start_line=node.start_line,
        end_line=node.end_line,
    )
    if node.signature:
        hit["signature"] = node.signature
    return hit


def _search_symbols(
    call: Call[SearchSymbolsPayload], args: SearchSymbolsArgs
) -> SearchSymbolsPayload:
    nodes, total = search_nodes(
        call.snap,
        args["query"],
        kind=args.get("kind"),
        lang=args.get("lang"),
        group=args.get("group"),
        path_glob=args.get("path_glob"),
        limit=args.get("limit", DEFAULT_SEARCH_LIMIT),
    )
    hits = [_hit(call.snap, node) for node in nodes]

    def assemble(kept: Sequence[Hit], omitted: Omitted, hint: str | None) -> SearchSymbolsPayload:
        payload = SearchSymbolsPayload(
            project=call.echo,
            errors=call.env.errors,
            degraded=call.env.degraded,
            truncated=hint is not None,
            hits=list(kept),
            total_matches=total,
        )
        if omitted:
            payload["omitted"] = omitted
        if hint is not None:
            payload["hint"] = hint
        return payload

    return fit_items(
        call,
        hits,
        assemble,
        lambda hit: _omit(hit["ref"], hit["qualified_name"], hit["path"]),
        "lower limit, or narrow with kind, lang, group or path_glob",
    )


def _empty_search(echo: ProjectEcho | None, env: Envelope) -> SearchSymbolsPayload:
    return SearchSymbolsPayload(
        project=echo,
        errors=env.errors,
        degraded=env.degraded,
        truncated=False,
        hits=[],
        total_matches=0,
    )


def _outline_node(
    node: Node,
    children: Mapping[str, list[Node]],
    first: Mapping[str, Node],
    snap: Snapshot,
    *,
    names_only: bool,
    docs: bool,
    depth: int,
    max_depth: int,
) -> OutlineNode:
    out = OutlineNode(
        ref=snap.symbol_ref(node), qualified_name=node.qualified_name, start_line=node.start_line
    )
    if not names_only:
        out["kind"] = node.kind
        out["end_line"] = node.end_line
        out["name"] = node.name
        if node.signature:
            out["signature"] = node.signature
        if docs and node.docstring:
            out["docstring"] = node.docstring
        if node.level is not None:
            out["level"] = node.level
        if node.children_count is not None:
            out["children_count"] = node.children_count
        if node.uniformity is not None:
            out["uniformity"] = node.uniformity
        if node.truncated_subtree:
            out["truncated_subtree"] = True
    below = children.get(node.qualified_name, []) if first[node.qualified_name] is node else []
    if below and depth < max_depth:
        out["children"] = [
            _outline_node(
                child,
                children,
                first,
                snap,
                names_only=names_only,
                docs=docs,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for child in below
        ]
    return out


def _import_items(call: Call[FileOutlinePayload], path: str) -> list[ImportItem]:
    data = call.snap.sync(path)
    if data is None:
        return []
    return [
        ImportItem(
            module=item.module,
            names=list(item.names),
            alias=item.alias,
            line=item.line,
            kind=item.kind,
        )
        for item in data.imports
    ]


def _line_count(source: bytes) -> int:
    if not source:
        return 0
    return source.count(b"\n") + (0 if source.endswith(b"\n") else 1)


def _file_outline(call: Call[FileOutlinePayload], args: FileOutlineArgs) -> FileOutlinePayload:
    snap = call.snap
    path = snap.normalize(args["path"])
    if path is None:
        return _empty_outline(call.echo, call.env)
    data = snap.sync(path)
    if data is None:
        call.env.add_error(
            ErrorEntry(code="not_found", message=f"{path} is not a readable indexed file")
        )
        return _empty_outline(call.echo, call.env)
    profile = data.file.profile
    key: Literal["symbols", "schema", "outline"] = (
        "schema" if profile == "schema" else "outline" if profile == "outline" else "symbols"
    )
    max_depth = args.get("max_depth", SCHEMA_DEPTH if profile == "schema" else DEFAULT_DEPTH)
    names_only = args.get("mode", "full") == "names"
    include = {part.strip() for part in args.get("include", "docs,imports").split(",")}
    nodes = snap.nodes(path)
    first: dict[str, Node] = {}
    for node in nodes:
        first.setdefault(node.qualified_name, node)
    children: dict[str, list[Node]] = {}
    roots: list[Node] = []
    for node in nodes:
        if node.parent is not None and node.parent in first and first[node.parent] is not node:
            children.setdefault(node.parent, []).append(node)
        else:
            roots.append(node)
    trees = [
        _outline_node(
            root,
            children,
            first,
            snap,
            names_only=names_only,
            docs="docs" in include,
            depth=1,
            max_depth=max_depth,
        )
        for root in roots
    ]
    imports = [] if names_only or "imports" not in include else _import_items(call, path)
    source = snap.source(path)
    import_entry = _omit(snap.byte_ref(path, 0, len(source)), "<imports>", path)

    def assemble(
        kept: Sequence[OutlineNode], omitted: Omitted, hint: str | None, with_imports: bool
    ) -> FileOutlinePayload:
        payload = FileOutlinePayload(
            project=call.echo,
            errors=call.env.errors,
            degraded=call.env.degraded,
            truncated=hint is not None,
            path=path,
            lang=data.file.lang,
            group=data.file.group,
            profile=profile,
            line_count=_line_count(source),
        )
        payload[key] = list(kept)
        if with_imports and imports:
            payload["imports"] = imports
        if omitted:
            payload["omitted"] = omitted
        if hint is not None:
            payload["hint"] = hint
        return payload

    hint = "lower max_depth, or use mode=names; expand omitted refs"
    full = assemble(trees, [], None, True)
    if call.fits(full):
        return full
    if imports:
        without = assemble(trees, [import_entry], "drop include=imports to omit them", False)
        if call.fits(without):
            return without
    lead: Omitted = [import_entry] if imports else []
    return fit_items(
        call,
        trees,
        lambda kept, omitted, note: assemble(kept, [*lead, *omitted], note, False),
        lambda tree: _omit(tree["ref"], tree["qualified_name"], path),
        hint,
    )


def _empty_outline(echo: ProjectEcho | None, env: Envelope) -> FileOutlinePayload:
    return FileOutlinePayload(
        project=echo,
        errors=env.errors,
        degraded=env.degraded,
        truncated=False,
        path="",
        lang=None,
        group=None,
        profile=None,
        line_count=0,
    )


def _ast_query(call: Call[AstQueryPayload], args: AstQueryArgs) -> AstQueryPayload:
    snap = call.snap
    path = snap.normalize(args["path"])
    empty = _empty_ast(call.echo, call.env)
    if path is None:
        return empty
    data = snap.sync(path)
    if data is None:
        call.env.add_error(ErrorEntry(code="not_found", message=f"{path} is not a readable file"))
        return empty
    source = snap.source(path)
    result = parse_bytes(path, source)
    for error in result.errors:
        if error.code != "syntax_error":
            call.env.add_error(lang_error_entry(error))
    parsed = result.parsed
    base = AstQueryPayload(
        project=call.echo,
        errors=call.env.errors,
        degraded=call.env.degraded,
        truncated=False,
        path=path,
        lang=data.file.lang,
        group=data.file.group,
        profile=data.file.profile,
        matches=[],
    )
    if parsed is None:
        return base
    language = load_language(parsed.spec.lang)
    if language is None:
        call.env.add_error(ErrorEntry(code="no_grammar", message=parsed.spec.lang))
        return base
    try:
        query = Query(language, args["query"])
    except QueryError as exc:
        call.env.add_error(ErrorEntry(code="bad_query", message=str(exc), stage="ast_query"))
        return base
    wanted = set(args.get("captures", []))
    matches: list[Match] = []
    seen: set[tuple[str, int, int]] = set()
    for captures in run_matches(query, parsed.tree.root_node):
        for capture, found in captures.items():
            if wanted and capture not in wanted:
                continue
            for node in found:
                marker = (capture, node.start_byte, node.end_byte)
                if marker in seen:
                    continue
                seen.add(marker)
                matches.append(
                    Match(
                        capture=capture,
                        node_type=node.type,
                        start_line=node.start_point.row + 1,
                        end_line=node.end_point.row + 1,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        text=parsed.text(node),
                    )
                )

    def assemble(kept: Sequence[Match], omitted: Omitted, hint: str | None) -> AstQueryPayload:
        payload = AstQueryPayload(
            project=call.echo,
            errors=call.env.errors,
            degraded=call.env.degraded,
            truncated=hint is not None,
            path=path,
            lang=data.file.lang,
            group=data.file.group,
            profile=data.file.profile,
            matches=list(kept),
        )
        if omitted:
            payload["omitted"] = omitted
        if hint is not None:
            payload["hint"] = hint
        return payload

    return fit_items(
        call,
        matches,
        assemble,
        lambda match: _omit(
            snap.byte_ref(path, match["start_byte"], match["end_byte"]),
            f"@{match['capture']}",
            path,
        ),
        "narrow the query or pass captures",
    )


def _empty_ast(echo: ProjectEcho | None, env: Envelope) -> AstQueryPayload:
    return AstQueryPayload(
        project=echo,
        errors=env.errors,
        degraded=env.degraded,
        truncated=False,
        path="",
        lang=None,
        group=None,
        profile=None,
        matches=[],
    )


def _line_of(source: bytes, offset: int) -> int:
    return source.count(b"\n", 0, offset) + 1


def _find_hash(snap: Snapshot, path: str, content_hash: str) -> Node | None:
    for node in snap.nodes(path):
        if node.content_hash == content_hash:
            return node
    for row in snap.queries.symbols_with_hash(content_hash):
        for node in snap.nodes(row.path):
            if node.content_hash == content_hash:
                return node
    return None


def _expand_payload(call: Call[ExpandPayload], ref: str) -> ExpandPayload:
    return ExpandPayload(
        project=call.echo,
        errors=call.env.errors,
        degraded=call.env.degraded,
        truncated=False,
        ref=ref,
    )


def _moved(call: Call[ExpandPayload], ref: str, node: Node) -> ExpandPayload:
    payload = _expand_payload(call, ref)
    payload["status"] = "moved"
    payload["current"] = call.snap.symbol_ref(node)
    payload["path"] = node.path
    payload["qualified_name"] = node.qualified_name
    return payload


def _expand(call: Call[ExpandPayload], args: ExpandArgs) -> ExpandPayload:
    raw = args["ref"]
    try:
        ref = parse_ref(raw)
    except RefError as exc:
        call.env.add_error(ErrorEntry(code="bad_ref", message=str(exc)))
        return _expand_payload(call, raw)
    if ref.key != call.project.key:
        call.env.add_error(
            ErrorEntry(code="ref_project_mismatch", message=f"ref belongs to {ref.key}")
        )
        return _expand_payload(call, raw)
    snap = call.snap
    view = args.get("view", "source")
    snap.sync(ref.path)
    match ref.anchor:
        case SymbolAnchor(name):
            nodes = snap.nodes(ref.path)
            hit = next(
                (
                    n
                    for n in nodes
                    if n.qualified_name == name and n.content_hash == ref.content_hash
                ),
                None,
            )
            if hit is not None:
                return _expanded_node(call, raw, hit, view)
            moved = _find_hash(snap, ref.path, ref.content_hash)
            if moved is not None:
                return _moved(call, raw, moved)
            return _gone(call, raw, next((n for n in nodes if n.qualified_name == name), None))
        case ByteAnchor(start, end):
            source = snap.source(ref.path)
            if short_hash(source[start:end]) == ref.content_hash and end <= len(source):
                return _expanded_bytes(call, raw, ref.path, start, end, view)
            moved = _find_hash(snap, ref.path, ref.content_hash)
            return _moved(call, raw, moved) if moved is not None else _gone(call, raw, None)
        case PartAnchor(name, part):
            return _expand_part(call, raw, ref, name, part, view)


def _gone(call: Call[ExpandPayload], raw: str, current: Node | None) -> ExpandPayload:
    payload = _expand_payload(call, raw)
    payload["status"] = "gone"
    if current is not None:
        payload["current"] = call.snap.symbol_ref(current)
    return payload


def _expand_part(
    call: Call[ExpandPayload], raw: str, ref: Ref, name: str, part: int, view: str
) -> ExpandPayload:
    try:
        chunks = call.snap.store.chunks_in_file(ref.path)
    except Exception as exc:
        call.env.add_degraded(DegradedEntry(stage="chunks", reason=f"{type(exc).__name__}: {exc}"))
        chunks = ()
    for chunk in chunks:
        if chunk.part == part and chunk.qualified_names[:1] == (name,):
            if chunk.content_hash[:12] == ref.content_hash:
                return _expanded_bytes(call, raw, ref.path, chunk.start_byte, chunk.end_byte, view)
            break
    return _gone(call, raw, None)


def _fit_source(
    call: Call[ExpandPayload], payload: ExpandPayload, text: str, view: str
) -> ExpandPayload:
    if view != "source":
        return payload

    def assemble(body: str, hint: str | None) -> ExpandPayload:
        sized = payload.copy()
        sized["source"] = body
        if hint is not None:
            sized["truncated"] = True
            sized["hint"] = hint
        return sized

    return fit_text(call, text, assemble, "raise max_tokens, or expand a smaller byte range")


def _expanded_node(call: Call[ExpandPayload], raw: str, node: Node, view: str) -> ExpandPayload:
    payload = _expand_payload(call, raw)
    payload["status"] = "ok"
    payload["path"] = node.path
    payload["qualified_name"] = node.qualified_name
    payload["kind"] = node.kind
    payload["profile"] = node.profile
    payload["group"] = node.group
    payload["start_line"] = node.start_line
    payload["end_line"] = node.end_line
    payload["start_byte"] = node.start_byte
    payload["end_byte"] = node.end_byte
    payload["signature"] = node.signature
    if view == "doc":
        payload["docstring"] = node.docstring
    return _fit_source(call, payload, _text(call.snap, node), view)


def _expanded_bytes(
    call: Call[ExpandPayload], raw: str, path: str, start: int, end: int, view: str
) -> ExpandPayload:
    snap = call.snap
    source = snap.source(path)
    data = snap.sync(path)
    payload = _expand_payload(call, raw)
    payload["status"] = "ok"
    payload["path"] = path
    payload["kind"] = "bytes"
    if data is not None:
        payload["profile"] = data.file.profile
        payload["group"] = data.file.group
    payload["start_line"] = _line_of(source, start)
    payload["end_line"] = _line_of(source, end)
    payload["start_byte"] = start
    payload["end_byte"] = end
    text = source[start:end].decode("utf-8", errors="replace")
    return _fit_source(call, payload, text, "source")


def _empty_expand(echo: ProjectEcho | None, env: Envelope) -> ExpandPayload:
    return ExpandPayload(
        project=echo, errors=env.errors, degraded=env.degraded, truncated=False, ref=""
    )


def _render_expand_ref(payload: ExpandPayload) -> str:
    return render_expand(payload)


define(
    Tool(
        "get_symbol",
        "Get symbol",
        "Fetch one definition, key path or heading by name: source, signature or doc.",
        GetSymbolArgs,
        GetSymbolPayload,
        _get_symbol,
        _empty_symbol,
        render_get_symbol,
    )
)
define(
    Tool(
        "file_outline",
        "File outline",
        "Shape of one file: nested symbols, schema key paths or headings, plus imports.",
        FileOutlineArgs,
        FileOutlinePayload,
        _file_outline,
        _empty_outline,
        render_file_outline,
    )
)
define(
    Tool(
        "search_symbols",
        "Search symbols",
        "Find definitions, key paths and headings by name across the project.",
        SearchSymbolsArgs,
        SearchSymbolsPayload,
        _search_symbols,
        _empty_search,
        render_search_symbols,
    )
)
define(
    Tool(
        "ast_query",
        "AST query",
        "Run a tree-sitter query on one file; the escape hatch when typed tools do not fit.",
        AstQueryArgs,
        AstQueryPayload,
        _ast_query,
        _empty_ast,
        render_ast_query,
    )
)
define(
    Tool(
        "expand",
        "Expand ref",
        "Resolve a cw:// ref to its current bytes; reports moved or gone when it went stale.",
        ExpandArgs,
        ExpandPayload,
        _expand,
        _empty_expand,
        _render_expand_ref,
    )
)
