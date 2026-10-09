from dataclasses import dataclass

from contextweave.index.records import FileData, FileProfile
from contextweave.index.store import hash_bytes
from contextweave.lang.registry import Group

HASH_LENGTH = 12


@dataclass(frozen=True, slots=True)
class Node:
    """A symbol, schema key path or document heading viewed uniformly."""

    path: str
    lang: str | None
    group: Group | None
    profile: FileProfile
    name: str
    qualified_name: str
    kind: str
    parent: str | None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    signature: str | None
    docstring: str | None
    content_hash: str
    level: int | None = None
    children_count: int | None = None
    uniformity: str | None = None
    truncated_subtree: bool = False


def short_hash(source: bytes) -> str:
    """First 12 hex characters of the sha256 of the bytes."""
    return hash_bytes(source)[:HASH_LENGTH]


def nodes_of(data: FileData, source: bytes) -> tuple[Node, ...]:
    """Every node of a file in source order, outer nodes before the ones they contain."""
    file = data.file
    found: list[Node] = [
        Node(
            file.path,
            file.lang,
            file.group,
            file.profile,
            item.name,
            item.qualified_name,
            item.kind,
            item.parent,
            item.start_byte,
            item.end_byte,
            item.start_line,
            item.end_line,
            item.signature,
            item.docstring,
            item.content_hash[:HASH_LENGTH],
        )
        for item in data.symbols
    ]
    found.extend(
        Node(
            file.path,
            file.lang,
            file.group,
            file.profile,
            item.key_path,
            item.key_path,
            item.kind,
            item.parent,
            item.start_byte,
            item.end_byte,
            item.start_line,
            item.end_line,
            item.value_preview,
            item.comment,
            short_hash(source[item.start_byte : item.end_byte]),
            None,
            item.children_count,
            item.uniformity,
            item.truncated_subtree,
        )
        for item in data.schema_nodes
    )
    found.extend(
        Node(
            file.path,
            file.lang,
            file.group,
            file.profile,
            item.title,
            item.slug,
            item.kind,
            item.parent,
            item.start_byte,
            item.end_byte,
            item.start_line,
            item.end_line,
            item.title,
            item.info,
            short_hash(source[item.start_byte : item.end_byte]),
            item.level,
        )
        for item in data.doc_nodes
    )
    return tuple(sorted(found, key=lambda node: (node.start_byte, -node.end_byte)))
