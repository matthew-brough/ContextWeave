from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

from tree_sitter import Node, Query

from contextweave.lang.model import Export, Import, LangError, Symbol
from contextweave.lang.parse import ParsedFile
from contextweave.lang.registry import load_language, query_text
from contextweave.lang.ts import Captures, compile_query, run_matches

MAX_SIGNATURE = 500

WRAPPERS = frozenset(
    {
        "decorated_definition",
        "export_statement",
        "type_declaration",
        "const_declaration",
        "var_declaration",
        "lexical_declaration",
        "variable_declaration",
    }
)

SIGNATURE_WRAPPERS = WRAPPERS - {"decorated_definition"}

COMMENT_TYPES = frozenset({"comment", "line_comment", "block_comment"})

MODULE_TYPES = frozenset(
    {
        "string",
        "string_literal",
        "interpreted_string_literal",
        "dotted_name",
        "raw_string_literal",
        "quoted_template",
        "string_content",
    }
)

KIND_RANK = {
    "var": 0,
    "const": 1,
    "property": 1,
    "module": 2,
    "type": 2,
    "rule": 3,
    "index": 3,
    "function": 4,
    "class": 5,
    "interface": 5,
    "struct": 5,
    "enum": 5,
    "message": 5,
    "service": 5,
    "table": 5,
    "view": 5,
    "resource": 5,
    "stage": 5,
    "block": 5,
    "handler": 4,
    "method": 6,
    "rpc": 6,
}

IMPORT_CALLEES = frozenset(
    {"require", "require_relative", "import", "library", "source", "include", ".", "module"}
)

METHOD_PARENTS = frozenset({"class", "interface", "struct", "message", "service"})

BODY_DOCSTRING_LANGS = frozenset({"python"})

IMPORT_ROLES = frozenset({"callee", "module", "alias"})


@dataclass(frozen=True, slots=True)
class SymbolExtraction:
    """Symbols, imports and exports of one file with the errors met on the way."""

    symbols: tuple[Symbol, ...]
    imports: tuple[Import, ...]
    exports: tuple[Export, ...]
    errors: tuple[LangError, ...]


@dataclass(frozen=True, slots=True)
class _RawDef:
    kind: str
    node: Node
    names: list[Node]
    registrar: Node | None = None


type _RawDefs = dict[tuple[int, int], _RawDef]


def preceding_comment_block(node: Node, src: bytes) -> str | None:
    """The contiguous comment block directly above node, verbatim, else None."""
    block: list[Node] = []
    anchor = node
    while anchor.prev_sibling is None and anchor.parent is not None:
        anchor = anchor.parent
    cursor = anchor.prev_sibling
    while cursor is not None and cursor.type in COMMENT_TYPES:
        if cursor.end_point[0] != anchor.start_point[0] - 1:
            break
        block.append(cursor)
        anchor = cursor
        cursor = cursor.prev_sibling
    if not block:
        trailing = _trailing_comment(anchor.prev_sibling)
        if trailing is None or trailing.end_point[0] != anchor.start_point[0] - 1:
            return None
        block = [trailing]
        cursor = trailing.prev_sibling
        while cursor is not None and cursor.type in COMMENT_TYPES:
            if cursor.end_point[0] != block[-1].start_point[0] - 1:
                break
            block.append(cursor)
            cursor = cursor.prev_sibling
    block.reverse()
    return src[block[0].start_byte : block[-1].end_byte].decode("utf-8", errors="replace")


def _trailing_comment(node: Node | None) -> Node | None:
    seen = 0
    while node is not None and seen < 32:
        if node.type in COMMENT_TYPES:
            return node
        if node.named_child_count == 0:
            return None
        node, seen = node.named_children[-1], seen + 1
    return None


def python_docstring(body: Node | None, src: bytes) -> str | None:
    """First string-literal statement of a Python body, verbatim, else None."""
    if body is None or not body.named_children:
        return None
    first = body.named_children[0]
    if first.type == "expression_statement":
        first = first.named_children[0] if first.named_children else first
    if first.type != "string":
        return None
    return src[first.start_byte : first.end_byte].decode("utf-8", errors="replace")


@lru_cache(maxsize=64)
def _compiled(lang: str, source: str) -> Query | None:
    language = load_language(lang)
    return None if language is None else compile_query(language, source)


def extract_symbols(parsed: ParsedFile) -> SymbolExtraction:
    """Run the language's query file over a parsed tree; never raises."""
    path = str(parsed.path)
    text = query_text(parsed.spec)
    if text is None:
        return _failed(LangError("no_query", path, f"{parsed.spec.lang} has no query file"))
    query = _compiled(parsed.spec.lang, text)
    if query is None:
        return _failed(LangError("bad_query", path, f"{parsed.spec.lang}.scm failed to compile"))
    try:
        matches = run_matches(query, parsed.tree.root_node)
    except Exception as exc:
        return _failed(LangError("query_failed", path, f"{type(exc).__name__}: {exc}"))

    raw: _RawDefs = {}
    imports: list[Import] = []
    for captures in matches:
        def_key = next((key for key in captures if key.startswith("def.")), None)
        if def_key is not None:
            _collect_definition(def_key, captures, raw)
            continue
        imported = _build_import(captures, parsed)
        if imported is not None:
            imports.append(imported)

    symbols = _nest(raw, parsed)
    return SymbolExtraction(
        symbols=tuple(symbols),
        imports=tuple(imports),
        exports=tuple(_collect_exports(symbols, raw, parsed)),
        errors=(),
    )


def _failed(error: LangError) -> SymbolExtraction:
    return SymbolExtraction((), (), (), (error,))


def _collect_definition(def_key: str, captures: Captures, raw: _RawDefs) -> None:
    nodes = captures.get(def_key) or []
    names = captures.get("name") or []
    if not nodes or not names:
        return
    node = nodes[0]
    outer = _outer(node)
    key = (outer.start_byte, outer.end_byte)
    kind = def_key.removeprefix("def.")
    previous = raw.get(key)
    rank = KIND_RANK.get(kind, 0)
    if (
        previous is None
        or rank > KIND_RANK.get(previous.kind, 0)
        or (rank == KIND_RANK.get(previous.kind, 0) and len(names) > len(previous.names))
    ):
        registrar = captures.get("handler.registrar") or []
        raw[key] = _RawDef(kind, node, names, registrar[0] if registrar else None)


def _outer(node: Node) -> Node:
    return _climb(node, WRAPPERS)


def _climb(node: Node, through: frozenset[str]) -> Node:
    current = node
    while current.parent is not None and current.parent.type in through:
        current = current.parent
    return current


def _signature(node: Node, src: bytes) -> str:
    prefix = src[_climb(node, SIGNATURE_WRAPPERS).start_byte : node.start_byte]
    body = node.child_by_field_name("body")
    if body is None:
        value = node.child_by_field_name("value")
        if value is not None:
            body = value.child_by_field_name("body")
    if body is not None and body.start_byte > node.start_byte:
        text = src[node.start_byte : _header_end(node, body)]
    else:
        text = src[node.start_byte : node.end_byte].split(b"\n", 1)[0]
    flat = " ".join((prefix + text).decode("utf-8", errors="replace").split())
    return flat[: MAX_SIGNATURE - 1] + "…" if len(flat) > MAX_SIGNATURE else flat


def _header_end(node: Node, body: Node) -> int:
    for child in node.children:
        if child.start_byte >= body.start_byte:
            break
        if child.type in COMMENT_TYPES:
            return child.start_byte
    return body.start_byte


def _docstring(parsed: ParsedFile, node: Node, outer: Node) -> str | None:
    if parsed.spec.lang in BODY_DOCSTRING_LANGS:
        found = python_docstring(node.child_by_field_name("body"), parsed.source)
        if found is not None:
            return found
    return preceding_comment_block(outer, parsed.source)


def _nest(raw: _RawDefs, parsed: ParsedFile) -> list[Symbol]:
    ordered = sorted(raw.items(), key=lambda item: (item[0][0], -item[0][1]))
    symbols: list[Symbol] = []
    stack: list[tuple[tuple[int, int], Symbol]] = []

    for (start, end), entry in ordered:
        while stack and not (stack[-1][0][0] <= start and end <= stack[-1][0][1]):
            stack.pop()
        parent = stack[-1][1] if stack else None
        kind = entry.kind
        if kind == "function" and parent is not None and parent.kind in METHOD_PARENTS:
            kind = "method"

        name = ".".join(parsed.text(part) for part in entry.names)
        qualified = f"{parent.qualified_name}.{name}" if parent else name
        outer = _outer(entry.node)
        symbol = Symbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            lang=parsed.spec.lang,
            group=parsed.spec.group,
            profile=parsed.spec.profile,
            path=str(parsed.path),
            start_byte=start,
            end_byte=end,
            start_line=outer.start_point[0] + 1,
            end_line=outer.end_point[0] + 1,
            signature=_signature(entry.node, parsed.source),
            docstring=_docstring(parsed, entry.node, outer),
            parent=parent.qualified_name if parent else None,
            registrar=parsed.text(entry.registrar) if entry.registrar is not None else None,
        )
        symbols.append(symbol)
        stack.append(((start, end), symbol))
    return symbols


def _collect_exports(symbols: list[Symbol], raw: _RawDefs, parsed: ParsedFile) -> list[Export]:
    exports: list[Export] = []
    for symbol in symbols:
        entry = raw.get((symbol.start_byte, symbol.end_byte))
        if entry is not None and _outer(entry.node).type == "export_statement":
            exports.append(
                Export(
                    name=symbol.name,
                    kind=symbol.kind,
                    lang=symbol.lang,
                    group=parsed.spec.group,
                    profile=parsed.spec.profile,
                    start_byte=symbol.start_byte,
                    end_byte=symbol.end_byte,
                    line=symbol.start_line,
                )
            )
    return exports


def _build_import(captures: Captures, parsed: ParsedFile) -> Import | None:
    if not any(key.startswith("import.") for key in captures):
        return None
    kinds = [key.removeprefix("import.") for key in captures if key.startswith("import.")]
    kind = next((k for k in kinds if k not in IMPORT_ROLES), "import")

    callee_nodes = captures.get("import.callee")
    callee = callee_nodes[0] if callee_nodes else None
    if callee is not None and parsed.text(callee) not in IMPORT_CALLEES:
        return None

    kind_nodes = captures.get(f"import.{kind}")
    node = kind_nodes[0] if kind_nodes else None
    module_nodes = captures.get("import.module")
    module_node = module_nodes[0] if module_nodes else None
    if module_node is None and node is not None:
        module_node = _first_of_type(node, MODULE_TYPES)
    if module_node is None:
        return None
    module = parsed.text(module_node).strip("\"'`")

    alias = _alias_for(node, module, parsed, callee) if node is not None else None

    names: list[str] = []
    if node is not None:
        callee_id = callee.id if callee is not None else None
        for descendant in _walk(node):
            if callee_id is not None and descendant.id == callee_id:
                continue
            if module_node.start_byte <= descendant.start_byte < module_node.end_byte:
                continue
            if descendant.type in {"identifier", "dotted_name", "type_identifier"}:
                text = parsed.text(descendant)
                if text != module and text != alias and text not in names:
                    names.append(text)

    anchor = node if node is not None else module_node
    return Import(
        module=module,
        names=tuple(names),
        alias=alias,
        kind=kind,
        lang=parsed.spec.lang,
        group=parsed.spec.group,
        profile=parsed.spec.profile,
        start_byte=anchor.start_byte,
        end_byte=anchor.end_byte,
        line=anchor.start_point[0] + 1,
    )


def _alias_for(node: Node, module: str, parsed: ParsedFile, callee: Node | None) -> str | None:
    direct = node.child_by_field_name("alias")
    if direct is not None:
        return parsed.text(direct)
    named = node.child_by_field_name("name")
    if named is not None and (callee is None or named.id != callee.id):
        text = parsed.text(named)
        if text != module and text.isidentifier():
            return text
    for descendant in _walk(node):
        nested = descendant.child_by_field_name("alias")
        if nested is not None:
            return parsed.text(nested)
    return None


def _walk(node: Node) -> Iterator[Node]:
    for child in node.named_children:
        yield child
        yield from _walk(child)


def _first_of_type(node: Node, types: frozenset[str]) -> Node | None:
    for child in _walk(node):
        if child.type in types:
            return child
    return None
