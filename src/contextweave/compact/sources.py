from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.parse import ParsedFile, parse_bytes

IMPORT_NODE_MARK = "import"


@dataclass(frozen=True, slots=True)
class ImportSpan:
    """The byte range of one whole import declaration and the module it names."""

    start_byte: int
    end_byte: int
    module: str


def declaration_of(node: Node) -> Node:
    """The outermost enclosing node whose grammar type is an import form."""
    current = node
    while current.parent is not None and IMPORT_NODE_MARK in current.parent.type:
        current = current.parent
    return current


def import_spans(parsed: ParsedFile) -> tuple[ImportSpan, ...]:
    spans: dict[tuple[int, int], ImportSpan] = {}
    for found in extract_symbols(parsed).imports:
        node = parsed.tree.root_node.descendant_for_byte_range(found.start_byte, found.end_byte)
        if node is None:
            continue
        outer = declaration_of(node)
        key = (outer.start_byte, outer.end_byte)
        spans.setdefault(key, ImportSpan(outer.start_byte, outer.end_byte, found.module))
    return tuple(sorted(spans.values(), key=lambda span: (span.start_byte, span.end_byte)))


class SourceCache:
    """Per-call cache of file bytes and import statements, primed with already-served bytes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._bytes: dict[str, bytes | None] = {}
        self._imports: dict[str, tuple[ImportSpan, ...]] = {}

    def preload(self, path: str, source: bytes) -> None:
        self._bytes[path] = source

    def read(self, path: str) -> bytes | None:
        if path not in self._bytes:
            try:
                self._bytes[path] = (self.root / path).read_bytes()
            except OSError:
                self._bytes[path] = None
        return self._bytes[path]

    def imports(self, path: str) -> tuple[ImportSpan, ...]:
        """Import statements of a file in source order; empty when it cannot be parsed."""
        if path not in self._imports:
            source = self.read(path)
            parsed = None if source is None else parse_bytes(path, source).parsed
            self._imports[path] = () if parsed is None else import_spans(parsed)
        return self._imports[path]
