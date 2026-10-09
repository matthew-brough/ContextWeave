from collections.abc import Callable, Sequence
from dataclasses import dataclass

from tree_sitter import Node

from contextweave.index.records import SchemaNodeRecord
from contextweave.lang.extract_schema import COLLECTION_KINDS, NODE_KINDS, UNIFORM, UNVERIFIED

HOMOGENEOUS = frozenset({UNIFORM, UNVERIFIED})
TOML_TABLES = frozenset({"table", "table_array_element"})
XML_TAGS = frozenset({"STag", "EmptyElemTag"})
ROOT_LABEL = "<root>"

type KeyOf = Callable[[Node], str]


@dataclass(frozen=True, slots=True)
class Collapse:
    """A homogeneous collection folded to one chunk: last sibling index and its descriptor."""

    last: int
    descriptor: str


class SchemaView:
    """Schema-profile structure the chunker needs: entry nodes, child keys, collection collapse."""

    def __init__(self, lang: str, nodes: Sequence[SchemaNodeRecord], source: bytes) -> None:
        self.lang = lang
        self.nodes = tuple(nodes)
        self._source = source
        self._table = NODE_KINDS.get(lang)
        self._by_range: dict[tuple[int, int], SchemaNodeRecord] = {}
        for record in self.nodes:
            self._by_range.setdefault((record.start_byte, record.end_byte), record)

    def entry_nodes(self, root: Node) -> list[Node]:
        """The top-level units of a data file."""
        if self._table is None:
            return [root] if self.lang == "csv" else list(root.named_children)
        current = root
        while current.type in self._table.unwrap:
            inner = [child for child in current.named_children if child.type != "comment"]
            if len(inner) != 1:
                break
            current = inner[0]
        return list(current.named_children) if current.type in self._table.mapping else [current]

    def children(self, node: Node, header: str) -> tuple[list[Node], str]:
        """The nodes an oversize node splits into, and the key path that heads them."""
        table = self._table
        if table is not None and node.type in table.pair:
            value = node.child_by_field_name(table.value_field)
            record = self._by_range.get((node.start_byte, node.end_byte))
            if record is not None:
                header = record.key_path
            return ([] if value is None else [value]), header
        if self.lang == "toml" and node.type in TOML_TABLES | {"pair"}:
            kids = list(node.named_children)
            if node.type == "pair":
                return kids[1:], header
            suffix = "[]" if node.type == "table_array_element" else ""
            return kids[1:], (self._text(kids[0]) + suffix if kids else header)
        if self.lang == "xml" and node.type == "element":
            content = next((c for c in node.named_children if c.type == "content"), None)
            name = self._xml_name(node)
            path = f"{header}/{name}" if header and name else name or header
            return ([] if content is None else list(content.named_children)), path
        return list(node.named_children), header

    def collapse(self, nodes: Sequence[Node], index: int) -> Collapse | None:
        """The collection the unit at index folds into, when it is a homogeneous one."""
        node = nodes[index]
        if self._table is not None:
            return self._collapse_sequence(node, index)
        if self.lang == "toml" and node.type == "table_array_element":
            return self._collapse_run(nodes, index, self._toml_key, "array")
        if self.lang == "xml" and node.type == "element":
            return self._collapse_run(nodes, index, self._xml_name, "element")
        if self.lang == "csv" and node.type == "document" and self.nodes:
            return Collapse(index, self._describe_columns())
        return None

    def _collapse_sequence(self, node: Node, index: int) -> Collapse | None:
        table = self._table
        if table is None:
            return None
        inner = node
        while inner.type in table.unwrap and len(inner.named_children) == 1:
            inner = inner.named_children[0]
        if inner.type not in table.sequence:
            return None
        parent = node.parent
        owner = parent if parent is not None and parent.type in table.pair else inner
        record = self._by_range.get((owner.start_byte, owner.end_byte))
        if record is None or record.kind != "array" or record.uniformity not in HOMOGENEOUS:
            return None
        return Collapse(index, self._describe(record))

    def _collapse_run(
        self, nodes: Sequence[Node], index: int, key_of: KeyOf, kind: str
    ) -> Collapse | None:
        first = nodes[index]
        record = self._by_range.get((first.start_byte, first.end_byte))
        if record is None or record.kind != kind or record.uniformity not in HOMOGENEOUS:
            return None
        key = key_of(first)
        last = index
        while (
            last + 1 < len(nodes)
            and nodes[last + 1].type == first.type
            and key_of(nodes[last + 1]) == key
        ):
            last += 1
        if record.children_count != last - index + 1:
            return None
        return Collapse(last, self._describe(record))

    def _describe(self, record: SchemaNodeRecord) -> str:
        lines = [_line(record, "collection")]
        seen = {record.key_path}
        for node in self.nodes:
            if node is not record and node.parent in seen:
                seen.add(node.key_path)
                lines.append("  " + _line(node, ""))
        return "\n".join(lines)

    def _describe_columns(self) -> str:
        rows = self.nodes[0].children_count
        lines = [f"collection {ROOT_LABEL}: table children_count={rows}"]
        lines.extend("  " + _line(node, "") for node in self.nodes)
        return "\n".join(lines)

    def _text(self, node: Node) -> str:
        return self._source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _toml_key(self, node: Node) -> str:
        return self._text(node.named_children[0]) if node.named_children else ""

    def _xml_name(self, node: Node) -> str:
        for tag in node.named_children:
            if tag.type in XML_TAGS:
                name = next((c for c in tag.named_children if c.type == "Name"), None)
                return "" if name is None else self._text(name)
        return ""


def _line(record: SchemaNodeRecord, label: str) -> str:
    path = record.key_path or ROOT_LABEL
    head = f"{label} {path}: " if label else f"{path}: "
    parts = [f"{record.kind}"]
    if record.kind in COLLECTION_KINDS:
        parts.append(f"children_count={record.children_count}")
    if record.uniformity is not None:
        parts.append(f"uniformity={record.uniformity}")
    if record.truncated_subtree:
        parts.append("truncated")
    return head + " ".join(parts)
