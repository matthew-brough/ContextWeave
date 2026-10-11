from collections.abc import Iterable, Mapping, Sequence

from contextweave.retrieve.pack import ContextPack
from contextweave.retrieve.pack_text import render_pack_text
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

CWD_VIA = "cwd"
OUTLINE_PROFILE = "outline"


def _head(payload: Common) -> list[str]:
    project = payload["project"]
    if project is None or project["via"] == CWD_VIA:
        return []
    return [f"project {project['key']} via {project['via']}"]


def _tail(payload: Common) -> list[str]:
    lines = [f"!{error['code']} {error.get('message', '')}".rstrip() for error in payload["errors"]]
    lines.extend(f"~{entry['stage']} {entry['reason']}" for entry in payload["degraded"])
    for entry in payload.get("omitted", []):
        if "qualified_name" in entry:
            lines.append(f"-omitted {entry['qualified_name']} {entry['path']}")
        else:
            lines.append(f"-omitted {entry['count']} more {entry['ref']}")
    if "hint" in payload:
        lines.append(f"hint: {payload['hint']}")
    return lines


def _file_header(path: str) -> str:
    return f"== {path} =="


def _by_path[T: Mapping[str, object]](items: Iterable[T]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = {}
    for item in items:
        path = item["path"]
        grouped.setdefault(path if isinstance(path, str) else "", []).append(item)
    return grouped


def _span(start: int, end: int) -> str:
    return f"L{start}" if end <= start else f"L{start}-{end}"


def _symbol_lines(item: SymbolItem) -> list[str]:
    lines = [
        f"{item['kind']} {item['qualified_name']} {_span(item['start_line'], item['end_line'])}"
    ]
    signature = item.get("signature")
    if signature and "source" not in item and item["profile"] != OUTLINE_PROFILE:
        lines.append(f"  sig: {signature}")
    if docstring := item.get("docstring"):
        lines.append(f"  doc: {docstring}")
    if "source" in item:
        lines.append(item["source"])
    if "annotation" in item:
        lines.append("  annotation: none")
    return lines


def _candidate_lines(items: Sequence[Candidate]) -> list[str]:
    return [f"? {c['kind']} {c['qualified_name']} {c['path']}:{c['start_line']}" for c in items]


def render_get_symbol(payload: GetSymbolPayload) -> str:
    lines = _head(payload)
    if payload["ambiguous"]:
        lines.append(f"ambiguous {len(payload.get('candidates', []))}")
    elif not payload["found"]:
        lines.append("not found")
    for path, items in _by_path(payload["symbols"]).items():
        lines.append(_file_header(path))
        for item in items:
            lines.extend(_symbol_lines(item))
    lines.extend(_candidate_lines(payload.get("candidates", [])))
    lines.extend(_tail(payload))
    return "\n".join(lines)


def _outline_lines(nodes: Sequence[OutlineNode], depth: int, profile: str | None) -> list[str]:
    lines: list[str] = []
    pad = "  " * depth
    for node in nodes:
        span = _span(node["start_line"], node.get("end_line", node["start_line"]))
        kind = f"{node['kind']} " if "kind" in node else ""
        shape = ""
        if "children_count" in node:
            shape = f" [{node['children_count']} {node.get('uniformity', '')}]".replace(" ]", "]")
        lines.append(f"{pad}{kind}{node['qualified_name']} {span}{shape}")
        signature = node.get("signature")
        if signature and profile != OUTLINE_PROFILE:
            lines.append(f"{pad}  sig: {signature}")
        if docstring := node.get("docstring"):
            lines.append(f"{pad}  doc: {docstring}")
        lines.extend(_outline_lines(node.get("children", []), depth + 1, profile))
    return lines


def render_file_outline(payload: FileOutlinePayload) -> str:
    lines = _head(payload)
    profile = payload["profile"]
    if profile is not None:
        lines.append(f"{_file_header(payload['path'])} {payload['line_count']} lines")
    for nodes in (
        payload.get("symbols", []),
        payload.get("schema", []),
        payload.get("outline", []),
    ):
        lines.extend(_outline_lines(nodes, 0, profile))
    for item in payload.get("imports", []):
        names = f" {{{', '.join(item['names'])}}}" if item["names"] else ""
        lines.append(f"import {item['module']}{names} L{item['line']}")
    lines.extend(_tail(payload))
    return "\n".join(lines)


def _hit_lines(items: Sequence[Hit]) -> list[str]:
    lines: list[str] = []
    for path, group in _by_path(items).items():
        lines.append(_file_header(path))
        for hit in group:
            lines.append(
                f"{hit['kind']} {hit['qualified_name']} {_span(hit['start_line'], hit['end_line'])}"
            )
            signature = hit.get("signature")
            if signature and hit["profile"] != OUTLINE_PROFILE:
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
    if payload["profile"] is not None:
        lines.append(_file_header(payload["path"]))
    for match in payload["matches"]:
        lines.append(
            f"@{match['capture']} {match['node_type']} {_span(match['start_line'], match['end_line'])}"
        )
        lines.append(match["text"])
    lines.extend(_tail(payload))
    return "\n".join(lines)


def render_expand(payload: ExpandPayload) -> str:
    lines = _head(payload)
    status = payload.get("status")
    if status != "ok":
        lines.append(f"{status or 'error'} {payload['ref']}")
    if "current" in payload:
        lines.append(f"current {payload['current']}")
    if status == "ok" and "path" in payload and "start_line" in payload:
        name = payload.get("qualified_name", "")
        span = _span(payload["start_line"], payload.get("end_line", payload["start_line"]))
        lines.append(f"{_file_header(payload['path'])} {name} {span}".replace("  ", " "))
    if (signature := payload.get("signature")) and "source" not in payload:
        lines.append(f"sig: {signature}")
    if docstring := payload.get("docstring"):
        lines.append(f"doc: {docstring}")
    if "source" in payload:
        lines.append(payload["source"])
    lines.extend(_tail(payload))
    return "\n".join(lines)


def render_context_search(pack: ContextPack) -> str:
    return render_pack_text(pack)
