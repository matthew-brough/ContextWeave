from collections.abc import Iterable, Mapping, Sequence

from contextweave.server.payloads import (
    AstQueryPayload,
    Candidate,
    Common,
    ExpandPayload,
    FileOutlinePayload,
    GetSymbolPayload,
    Hit,
    OutlineNode,
    SearchSymbolsPayload,
    SymbolItem,
)


def _head(payload: Common) -> list[str]:
    project = payload["project"]
    return [] if project is None else [f"project {project['key']} via {project['via']}"]


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


def _file_header(path: str, group: str | None, lang: str | None, profile: str) -> str:
    label = "/".join(part for part in (group, lang) if part)
    return f"== {path} [{label} {profile}] =="


def _by_path[T: Mapping[str, object]](items: Iterable[T]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = {}
    for item in items:
        path = item["path"]
        grouped.setdefault(path if isinstance(path, str) else "", []).append(item)
    return grouped


def _symbol_lines(item: SymbolItem) -> list[str]:
    lines = [
        f"{item['kind']} {item['qualified_name']} L{item['start_line']}-{item['end_line']} {item['ref']}"
    ]
    if signature := item.get("signature"):
        lines.append(f"  sig: {signature}")
    if docstring := item.get("docstring"):
        lines.append(f"  doc: {docstring}")
    if "source" in item:
        lines.append(item["source"])
    if "annotation" in item:
        lines.append("  annotation: none")
    return lines


def _candidate_lines(items: Sequence[Candidate]) -> list[str]:
    return [
        f"? {c['kind']} {c['qualified_name']} {c['path']}:{c['start_line']} {c['ref']}"
        for c in items
    ]


def render_get_symbol(payload: GetSymbolPayload) -> str:
    lines = _head(payload)
    lines.append(f"found={payload['found']} ambiguous={payload['ambiguous']}")
    for path, items in _by_path(payload["symbols"]).items():
        first = items[0]
        lines.append(_file_header(path, first["group"], first["lang"], first["profile"]))
        for item in items:
            lines.extend(_symbol_lines(item))
    lines.extend(_candidate_lines(payload.get("candidates", [])))
    lines.extend(_tail(payload))
    return "\n".join(lines)


def _outline_lines(nodes: Sequence[OutlineNode], depth: int) -> list[str]:
    lines: list[str] = []
    pad = "  " * depth
    for node in nodes:
        span = f"L{node['start_line']}" + (f"-{node['end_line']}" if "end_line" in node else "")
        kind = f"{node['kind']} " if "kind" in node else ""
        lines.append(f"{pad}{kind}{node['qualified_name']} {span} {node['ref']}")
        if signature := node.get("signature"):
            lines.append(f"{pad}  sig: {signature}")
        if docstring := node.get("docstring"):
            lines.append(f"{pad}  doc: {docstring}")
        lines.extend(_outline_lines(node.get("children", []), depth + 1))
    return lines


def render_file_outline(payload: FileOutlinePayload) -> str:
    lines = _head(payload)
    profile = payload["profile"]
    if profile is not None:
        lines.append(_file_header(payload["path"], payload["group"], payload["lang"], profile))
        lines.append(f"lines {payload['line_count']}")
    for nodes in (
        payload.get("symbols", []),
        payload.get("schema", []),
        payload.get("outline", []),
    ):
        lines.extend(_outline_lines(nodes, 0))
    for item in payload.get("imports", []):
        names = f" {{{', '.join(item['names'])}}}" if item["names"] else ""
        lines.append(f"import {item['module']}{names} L{item['line']}")
    lines.extend(_tail(payload))
    return "\n".join(lines)


def _hit_lines(items: Sequence[Hit]) -> list[str]:
    lines: list[str] = []
    for path, group in _by_path(items).items():
        first = group[0]
        lines.append(_file_header(path, first["group"], first["lang"], first["profile"]))
        for hit in group:
            lines.append(
                f"{hit['kind']} {hit['qualified_name']} L{hit['start_line']}-{hit['end_line']} {hit['ref']}"
            )
            if signature := hit.get("signature"):
                lines.append(f"  sig: {signature}")
    return lines


def render_search_symbols(payload: SearchSymbolsPayload) -> str:
    lines = _head(payload)
    lines.append(f"{len(payload['hits'])} of {payload['total_matches']}")
    lines.extend(_hit_lines(payload["hits"]))
    lines.extend(_tail(payload))
    return "\n".join(lines)


def render_ast_query(payload: AstQueryPayload) -> str:
    lines = _head(payload)
    profile = payload["profile"]
    if profile is not None:
        lines.append(_file_header(payload["path"], payload["group"], payload["lang"], profile))
    for match in payload["matches"]:
        lines.append(
            f"@{match['capture']} {match['node_type']} L{match['start_line']}-{match['end_line']}"
            f" b{match['start_byte']}-{match['end_byte']}"
        )
        lines.append(match["text"])
    lines.extend(_tail(payload))
    return "\n".join(lines)


def render_expand(payload: ExpandPayload) -> str:
    lines = _head(payload)
    status = payload.get("status")
    lines.append(f"{status or 'error'} {payload['ref']}")
    if "current" in payload:
        lines.append(f"current {payload['current']}")
    if "path" in payload and "profile" in payload:
        lines.append(_file_header(payload["path"], payload.get("group"), None, payload["profile"]))
    if "start_line" in payload:
        lines.append(f"L{payload['start_line']}-{payload.get('end_line', payload['start_line'])}")
    if signature := payload.get("signature"):
        lines.append(f"sig: {signature}")
    if docstring := payload.get("docstring"):
        lines.append(f"doc: {docstring}")
    if "source" in payload:
        lines.append(payload["source"])
    lines.extend(_tail(payload))
    return "\n".join(lines)
