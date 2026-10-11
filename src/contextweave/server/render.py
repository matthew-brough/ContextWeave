import re
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
    ImportItem,
    OutlineNode,
    SearchSymbolsPayload,
    SymbolItem,
)

UNPROMPTED_VIA = "roots"
CODE_PROFILE = "symbols"
NAME_SEGMENT = re.compile(r"[.:]")


def head_lines(payload: Common) -> list[str]:
    """Project line only when the client, not the agent, chose the project."""
    project = payload["project"]
    if project is None or project["via"] != UNPROMPTED_VIA:
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


def _signs(signature: str | None, name: str, nested: bool) -> bool:
    if not signature:
        return False
    if name in signature:
        return True
    last = NAME_SEGMENT.split(name)[-1]
    return nested and re.search(rf"\b{re.escape(last)}\b", signature) is not None


def _named_line(
    kind: str, name: str, span: str, signature: str | None, profile: str | None, nested: bool
) -> list[str]:
    """`L1-5 <signature>` when the signature names the symbol, else `kind name span` + `sig:`."""
    code = profile == CODE_PROFILE
    if code and signature and _signs(signature, name, nested):
        return [f"{span} {signature}"]
    lines = [f"{kind}{name} {span}"]
    if code and signature:
        lines.append(f"  sig: {signature}")
    return lines


def _symbol_lines(item: SymbolItem) -> list[str]:
    span = _span(item["start_line"], item["end_line"])
    signature = None if "source" in item else item.get("signature")
    lines = _named_line(
        f"{item['kind']} ", item["qualified_name"], span, signature, item["profile"], False
    )
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
    lines = head_lines(payload)
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
        if "children_count" in node:
            shape = f" [{node['children_count']} {node.get('uniformity', '')}]".replace(" ]", "]")
            span = f"{span}{shape}"
        named = _named_line(
            kind, node["qualified_name"], span, node.get("signature"), profile, True
        )
        lines.extend(f"{pad}{line}" for line in named)
        if docstring := node.get("docstring"):
            lines.append(f"{pad}  doc: {docstring}")
        lines.extend(_outline_lines(node.get("children", []), depth + 1, profile))
    return lines


def _import_entry(item: ImportItem) -> str:
    names = f" {{{', '.join(item['names'])}}}" if item["names"] else ""
    return f"{item['module']}{names} L{item['line']}"


def render_file_outline(payload: FileOutlinePayload) -> str:
    lines = head_lines(payload)
    profile = payload["profile"]
    if profile is not None:
        lines.append(f"{_file_header(payload['path'])} {payload['line_count']} lines")
    for nodes in (
        payload.get("symbols", []),
        payload.get("schema", []),
        payload.get("outline", []),
    ):
        lines.extend(_outline_lines(nodes, 0, profile))
    if imports := [_import_entry(item) for item in payload.get("imports", [])]:
        lines.append(f"imports: {', '.join(imports)}")
    lines.extend(_tail(payload))
    return "\n".join(lines)


def _hit_lines(items: Sequence[Hit]) -> list[str]:
    lines: list[str] = []
    for path, group in _by_path(items).items():
        lines.append(_file_header(path))
        for hit in group:
            span = _span(hit["start_line"], hit["end_line"])
            lines.extend(
                _named_line(
                    f"{hit['kind']} ",
                    hit["qualified_name"],
                    span,
                    hit.get("signature"),
                    hit["profile"],
                    False,
                )
            )
    return lines


def render_search_symbols(payload: SearchSymbolsPayload) -> str:
    lines = head_lines(payload)
    lines.append(f"{len(payload['hits'])} of {payload['total_matches']}")
    lines.extend(_hit_lines(payload["hits"]))
    lines.extend(_tail(payload))
    return "\n".join(lines)


def render_ast_query(payload: AstQueryPayload) -> str:
    lines = head_lines(payload)
    if payload["profile"] is not None:
        lines.append(_file_header(payload["path"]))
    for match in payload["matches"]:
        head = f"@{match['capture']} {_span(match['start_line'], match['end_line'])}"
        text = match["text"]
        lines.extend([f"{head} {text}"] if "\n" not in text else [head, text])
    lines.extend(_tail(payload))
    return "\n".join(lines)


def render_expand(payload: ExpandPayload) -> str:
    lines = head_lines(payload)
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
