import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from tree_sitter import Node

from contextweave.lang.extract_symbols import preceding_comment_block
from contextweave.lang.model import LangError
from contextweave.lang.parse import ParsedFile
from contextweave.lang.registry import Group, Profile

DEFAULT_MAX_DEPTH = 4
PREVIEW_CHARS = 80
SCAN_LIMIT = 5000
UNIFORM, MIXED, UNVERIFIED = "uniform", "mixed", "unverified"

COLLECTION_KINDS = frozenset({"object", "array", "table", "element"})


@dataclass(frozen=True, slots=True)
class NodeKinds:
    """Grammar node names for the roles the tree walker needs."""

    mapping: frozenset[str]
    sequence: frozenset[str]
    pair: frozenset[str]
    unwrap: frozenset[str]
    key_field: str = "key"
    value_field: str = "value"


NODE_KINDS: dict[str, NodeKinds] = {
    "json": NodeKinds(
        mapping=frozenset({"object"}),
        sequence=frozenset({"array"}),
        pair=frozenset({"pair"}),
        unwrap=frozenset({"document"}),
    ),
    "json5": NodeKinds(
        mapping=frozenset({"object"}),
        sequence=frozenset({"array"}),
        pair=frozenset({"pair", "member"}),
        unwrap=frozenset({"file"}),
        key_field="name",
    ),
    "yaml": NodeKinds(
        mapping=frozenset({"block_mapping", "flow_mapping"}),
        sequence=frozenset({"block_sequence", "flow_sequence"}),
        pair=frozenset({"block_mapping_pair", "flow_pair"}),
        unwrap=frozenset({"stream", "document", "block_node", "flow_node", "block_sequence_item"}),
    ),
}

VALUE_KINDS = {
    "object": "object",
    "inline_table": "object",
    "block_mapping": "object",
    "flow_mapping": "object",
    "table": "object",
    "array": "array",
    "block_sequence": "array",
    "flow_sequence": "array",
    "string": "string",
    "string_scalar": "string",
    "number": "number",
    "integer": "number",
    "float": "number",
    "true": "bool",
    "false": "bool",
    "boolean": "bool",
    "boolean_scalar": "bool",
    "null": "null",
    "null_scalar": "null",
}

_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_BOOL = frozenset({"true", "false", "yes", "no", "on", "off"})
_NULL = frozenset({"null", "~", "nil", ""})


@dataclass(frozen=True, slots=True)
class SchemaNode:
    """One key path with its inferred value type; never a function, class or method."""

    key_path: str
    kind: str
    lang: str
    group: Group
    profile: Profile
    path: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    value_preview: str | None
    comment: str | None
    children_count: int
    truncated_subtree: bool
    parent: str | None
    uniformity: str | None = None


@dataclass(frozen=True, slots=True)
class SchemaExtraction:
    """Schema nodes of one file with the errors met on the way."""

    nodes: tuple[SchemaNode, ...]
    errors: tuple[LangError, ...]


@dataclass(slots=True)
class _Sink:
    nodes: list[SchemaNode] = field(default_factory=list[SchemaNode])


type _Walker = Callable[[ParsedFile, int, _Sink], None]


def extract_schema(parsed: ParsedFile, max_depth: int = DEFAULT_MAX_DEPTH) -> SchemaExtraction:
    """Walk a data file into key paths; never raises."""
    sink = _Sink()
    path = str(parsed.path)
    walker = _WALKERS.get(parsed.spec.lang)
    if walker is None:
        error = LangError("unsupported_lang", path, f"{parsed.spec.lang} has no schema walker")
        return SchemaExtraction((), (error,))
    try:
        walker(parsed, max_depth, sink)
    except Exception as exc:
        error = LangError("schema_failed", path, f"{type(exc).__name__}: {exc}")
        return SchemaExtraction(tuple(sink.nodes), (error,))
    return SchemaExtraction(tuple(sink.nodes), ())


def _walk_tree(
    parsed: ParsedFile,
    node: Node,
    prefix: str,
    parent: str | None,
    depth: int,
    max_depth: int,
    sink: _Sink,
    in_collection: bool = False,
) -> None:
    table = NODE_KINDS[parsed.spec.lang]
    unwrapped = _unwrap(node, table)
    if unwrapped is None:
        return
    node = unwrapped

    if node.type in table.mapping:
        for pair in node.named_children:
            if pair.type in table.pair:
                _emit_pair(
                    parsed, pair, table, prefix, parent, depth, max_depth, sink, in_collection
                )
        return

    if node.type in table.sequence:
        _emit_value(parsed, node, node, prefix, parent, depth, max_depth, sink, in_collection)
        return

    if node.type == "ERROR" or node.has_error:
        for child in node.named_children:
            if child.type in table.pair:
                _emit_pair(
                    parsed, child, table, prefix, parent, depth, max_depth, sink, in_collection
                )
            elif child.type != "comment" and child.named_child_count:
                _walk_tree(parsed, child, prefix, parent, depth, max_depth, sink, in_collection)


def _emit_pair(
    parsed: ParsedFile,
    pair: Node,
    table: NodeKinds,
    prefix: str,
    parent: str | None,
    depth: int,
    max_depth: int,
    sink: _Sink,
    in_collection: bool,
) -> None:
    key_node = pair.child_by_field_name(table.key_field)
    if key_node is None:
        return
    value_node = pair.child_by_field_name(table.value_field)
    key = _scalar_text(parsed, key_node)
    path = f"{prefix}.{key}" if prefix else key
    _emit_value(parsed, pair, value_node, path, parent, depth, max_depth, sink, in_collection)


def _emit_value(
    parsed: ParsedFile,
    owner: Node,
    value: Node | None,
    path: str,
    parent: str | None,
    depth: int,
    max_depth: int,
    sink: _Sink,
    in_collection: bool = False,
) -> None:
    table = NODE_KINDS[parsed.spec.lang]
    value = _unwrap(value, table) if value is not None else None
    kind = _kind_of(parsed, value)
    is_mapping = value is not None and value.type in table.mapping
    is_sequence = value is not None and value.type in table.sequence
    uniformity: str | None = None
    elements: list[Node] = []
    count = 0

    if value is not None and is_sequence:
        elements = [
            element
            for element in (_unwrap(child, table) for child in value.named_children)
            if element is not None and element.type != "comment"
        ]
        count = len(elements)
        uniformity = _sequence_uniformity(parsed, elements, table)
    elif value is not None and is_mapping:
        count = sum(1 for child in value.named_children if child.type in table.pair)

    over_depth = depth >= max_depth and (is_mapping or is_sequence)
    sink.nodes.append(
        _node(
            parsed,
            owner,
            path,
            kind,
            parent,
            value,
            count,
            over_depth,
            preview=not in_collection,
            uniformity=uniformity,
        )
    )
    if over_depth or value is None:
        return

    if is_mapping:
        _walk_tree(parsed, value, path, path, depth + 1, max_depth, sink, in_collection)
    elif is_sequence and elements:
        first = elements[0]
        if first.type in table.mapping or first.type in table.sequence:
            _walk_tree(parsed, first, f"{path}[]", path, depth + 1, max_depth, sink, True)


def _sequence_uniformity(parsed: ParsedFile, elements: list[Node], table: NodeKinds) -> str | None:
    if not elements:
        return None
    if len(elements) > SCAN_LIMIT:
        return UNVERIFIED
    shapes = {_shape_signature(parsed, element, table) for element in elements}
    return UNIFORM if len(shapes) == 1 else MIXED


def _shape_signature(parsed: ParsedFile, node: Node, table: NodeKinds) -> str:
    if node.type in table.mapping:
        keys = sorted(
            _scalar_text(parsed, key)
            for key in (
                pair.child_by_field_name(table.key_field)
                for pair in node.named_children
                if pair.type in table.pair
            )
            if key is not None
        )
        return "object:" + ",".join(keys)
    if node.type in table.sequence:
        return "array"
    return _kind_of(parsed, node)


def _unwrap(node: Node | None, table: NodeKinds) -> Node | None:
    seen = 0
    while node is not None and node.type in table.unwrap and seen < 16:
        following = next((c for c in node.named_children if c.type != "comment"), None)
        if following is None:
            return None
        node, seen = following, seen + 1
    return node


def _walk_json_like(parsed: ParsedFile, max_depth: int, sink: _Sink) -> None:
    _walk_tree(parsed, parsed.tree.root_node, "", None, 1, max_depth, sink)


def _walk_toml_root(parsed: ParsedFile, max_depth: int, sink: _Sink) -> None:
    _walk_toml(parsed, parsed.tree.root_node, "", None, 1, max_depth, sink)


def _walk_xml_root(parsed: ParsedFile, max_depth: int, sink: _Sink) -> None:
    _walk_xml(parsed, parsed.tree.root_node, "", None, 1, max_depth, sink)


def _walk_csv_root(parsed: ParsedFile, _max_depth: int, sink: _Sink) -> None:
    _walk_csv(parsed, sink)


def _walk_toml(
    parsed: ParsedFile,
    node: Node,
    prefix: str,
    parent: str | None,
    depth: int,
    max_depth: int,
    sink: _Sink,
    in_collection: bool = False,
) -> None:
    seen_arrays: set[str] = set()
    repeats: dict[str, list[Node]] = {}
    for child in node.named_children:
        if child.type == "table_array_element" and child.named_child_count:
            name = _scalar_text(parsed, child.named_children[0])
            repeats.setdefault(name, []).append(child)
    for child in node.named_children:
        if child.type == "pair":
            key_node = child.named_children[0]
            value_node = child.named_children[-1] if child.named_child_count > 1 else None
            key = _scalar_text(parsed, key_node)
            path = f"{prefix}.{key}" if prefix else key
            sink.nodes.append(
                _node(
                    parsed,
                    child,
                    path,
                    _kind_of(parsed, value_node),
                    parent,
                    value_node,
                    0,
                    False,
                    preview=not in_collection,
                )
            )
        elif child.type in {"table", "table_array_element"}:
            if not child.named_child_count:
                continue
            name = _scalar_text(parsed, child.named_children[0])
            is_array = child.type == "table_array_element"
            path = f"{name}[]" if is_array else name
            if is_array and path in seen_arrays:
                continue
            seen_arrays.add(path)
            siblings = repeats[name] if is_array else []
            count = (
                len(siblings)
                if is_array
                else sum(1 for c in child.named_children if c.type == "pair")
            )
            over = depth >= max_depth
            sink.nodes.append(
                _node(
                    parsed,
                    child,
                    path,
                    "array" if is_array else "table",
                    parent,
                    None,
                    count,
                    over,
                    uniformity=_toml_uniformity(parsed, siblings) if is_array else None,
                )
            )
            if not over:
                _walk_toml(
                    parsed, child, path, path, depth + 1, max_depth, sink, in_collection or is_array
                )


def _toml_uniformity(parsed: ParsedFile, siblings: list[Node]) -> str | None:
    if not siblings:
        return None
    if len(siblings) > SCAN_LIMIT:
        return UNVERIFIED
    shapes = {
        ",".join(
            sorted(
                _scalar_text(parsed, c.named_children[0])
                for c in element.named_children
                if c.type == "pair" and c.named_child_count
            )
        )
        for element in siblings
    }
    return UNIFORM if len(shapes) == 1 else MIXED


def _walk_xml(
    parsed: ParsedFile,
    node: Node,
    prefix: str,
    parent: str | None,
    depth: int,
    max_depth: int,
    sink: _Sink,
    in_collection: bool = False,
) -> None:
    elements = [c for c in node.named_children if c.type == "element"]
    if not elements and node.type == "document":
        root = node.child_by_field_name("root")
        elements = [root] if root is not None else []

    grouped: dict[str, list[Node]] = {}
    for element in elements:
        grouped.setdefault(_xml_name(parsed, element) or "?", []).append(element)

    for name, siblings in grouped.items():
        path = f"{prefix}/{name}" if prefix else name
        first = siblings[0]
        over = depth >= max_depth
        collapsed = in_collection or len(siblings) > 1
        sink.nodes.append(
            _node(
                parsed,
                first,
                path,
                "element",
                parent,
                None,
                len(siblings),
                over,
                uniformity=_xml_uniformity(parsed, siblings),
            )
        )
        if over:
            continue
        for attr_name, attr_node in _xml_attributes(parsed, first):
            value = next((c for c in attr_node.named_children if c.type == "AttValue"), attr_node)
            sink.nodes.append(
                _node(
                    parsed,
                    attr_node,
                    f"{path}@{attr_name}",
                    "attr",
                    path,
                    value,
                    0,
                    False,
                    preview=not collapsed,
                )
            )
        content = next((c for c in first.named_children if c.type == "content"), None)
        if content is not None:
            _walk_xml(parsed, content, path, path, depth + 1, max_depth, sink, collapsed)


def _xml_uniformity(parsed: ParsedFile, siblings: list[Node]) -> str | None:
    if len(siblings) < 2:
        return None
    if len(siblings) > SCAN_LIMIT:
        return UNVERIFIED
    shapes = {_xml_shape(parsed, element) for element in siblings}
    return UNIFORM if len(shapes) == 1 else MIXED


def _xml_shape(parsed: ParsedFile, element: Node) -> str:
    attrs = sorted(name for name, _ in _xml_attributes(parsed, element))
    content = next((c for c in element.named_children if c.type == "content"), None)
    children = sorted(
        {
            _xml_name(parsed, c) or "?"
            for c in (content.named_children if content is not None else [])
            if c.type == "element"
        }
    )
    return f"@{','.join(attrs)}|{','.join(children)}"


def _xml_tag(element: Node) -> Node | None:
    return next((c for c in element.named_children if c.type in {"STag", "EmptyElemTag"}), None)


def _xml_name(parsed: ParsedFile, element: Node) -> str | None:
    tag = _xml_tag(element)
    if tag is None:
        return None
    name = next((c for c in tag.named_children if c.type == "Name"), None)
    return _scalar_text(parsed, name) if name is not None else None


def _xml_attributes(parsed: ParsedFile, element: Node) -> Iterator[tuple[str, Node]]:
    tag = _xml_tag(element)
    if tag is None:
        return
    for attribute in tag.named_children:
        if attribute.type != "Attribute":
            continue
        name = next((c for c in attribute.named_children if c.type == "Name"), None)
        if name is not None:
            yield _scalar_text(parsed, name), attribute


def _walk_csv(parsed: ParsedFile, sink: _Sink) -> None:
    rows = [r for r in parsed.tree.root_node.named_children if r.type == "row"]
    if not rows:
        return
    header, data = rows[0], rows[1:]
    header_fields = [f for f in header.named_children if f.type == "field"]

    scanned = data[:SCAN_LIMIT]
    for index, field_node in enumerate(header_fields):
        name = _scalar_text(parsed, field_node)
        samples: list[Node] = []
        for row in scanned:
            cells = [c for c in row.named_children if c.type == "field"]
            if index < len(cells):
                samples.append(cells[index])
        kinds = {_infer_from_text(_scalar_text(parsed, s)) for s in samples}
        kinds.discard("null")
        mixed = len(kinds) > 1
        sink.nodes.append(
            SchemaNode(
                key_path=name,
                kind=next(iter(kinds)) if len(kinds) == 1 else "string" if mixed else "null",
                lang=parsed.spec.lang,
                group=parsed.spec.group,
                profile=parsed.spec.profile,
                path=str(parsed.path),
                start_byte=field_node.start_byte,
                end_byte=field_node.end_byte,
                start_line=field_node.start_point[0] + 1,
                end_line=field_node.end_point[0] + 1,
                value_preview=None,
                comment=None,
                children_count=len(data),
                truncated_subtree=False,
                parent=None,
                uniformity=(UNVERIFIED if len(data) > SCAN_LIMIT else MIXED if mixed else UNIFORM),
            )
        )


def _node(
    parsed: ParsedFile,
    owner: Node,
    path: str,
    kind: str,
    parent: str | None,
    value: Node | None,
    children_count: int,
    truncated: bool,
    preview: bool = True,
    uniformity: str | None = None,
) -> SchemaNode:
    scalar = value is not None and kind not in COLLECTION_KINDS
    return SchemaNode(
        key_path=path,
        kind=kind,
        lang=parsed.spec.lang,
        group=parsed.spec.group,
        profile=parsed.spec.profile,
        path=str(parsed.path),
        start_byte=owner.start_byte,
        end_byte=owner.end_byte,
        start_line=owner.start_point[0] + 1,
        end_line=owner.end_point[0] + 1,
        value_preview=_preview(parsed, value) if scalar and preview else None,
        comment=preceding_comment_block(owner, parsed.source),
        children_count=children_count,
        truncated_subtree=truncated,
        parent=parent,
        uniformity=uniformity,
    )


def _kind_of(parsed: ParsedFile, value: Node | None) -> str:
    if value is None:
        return "null"
    mapped = VALUE_KINDS.get(value.type)
    if mapped is not None:
        return mapped
    inner = next((c for c in value.named_children if c.type != "comment"), None)
    if inner is not None and inner.type in VALUE_KINDS:
        return VALUE_KINDS[inner.type]
    return _infer_from_text(_scalar_text(parsed, value))


def _infer_from_text(text: str) -> str:
    lowered = text.strip().strip("\"'").lower()
    if lowered in _NULL:
        return "null"
    if lowered in _BOOL:
        return "bool"
    if _NUMBER.match(lowered):
        return "number"
    return "string"


def _scalar_text(parsed: ParsedFile, node: Node) -> str:
    return parsed.text(node).strip().strip("\"'")


def _preview(parsed: ParsedFile, node: Node | None) -> str | None:
    if node is None:
        return None
    text = " ".join(parsed.text(node).split())
    if len(text) > PREVIEW_CHARS:
        return text[: PREVIEW_CHARS - 1] + "…"
    return text or None


_WALKERS: dict[str, _Walker] = {
    "json": _walk_json_like,
    "json5": _walk_json_like,
    "yaml": _walk_json_like,
    "toml": _walk_toml_root,
    "xml": _walk_xml_root,
    "csv": _walk_csv_root,
}
