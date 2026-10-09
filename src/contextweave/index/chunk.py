import hashlib
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, replace

from tree_sitter import Node

from contextweave.core.config import ChunkConfig
from contextweave.core.refs import Anchor, ByteAnchor, PartAnchor, SymbolAnchor
from contextweave.index.chunk_schema import SchemaView
from contextweave.index.chunk_tokens import ChunkTokenizer, default_tokenizer
from contextweave.index.records import BodyMode, Chunk, SchemaNodeRecord, SymbolRecord
from contextweave.lang.parse import ParsedFile

FUNCTION_KINDS = frozenset({"function", "method", "handler"})
GLUE_KINDS = frozenset({"glue", "keys", "section", "text"})
HTML_TAGS = frozenset({"start_tag", "end_tag", "self_closing_tag"})
HEADING_TYPES = frozenset({"atx_heading", "setext_heading"})
SECTION_TYPES = frozenset({"section"})
BODY_SEARCH_DEPTH = 4
HEADER_CAP = 160
HEADER_JOIN = " > "
SIGNATURE_JOIN = "\n"


def chunk_anchor(chunk: Chunk) -> Anchor:
    """The ref anchor of a chunk: split part, first symbol, else byte range."""
    if chunk.qualified_names:
        name = chunk.qualified_names[0]
        return SymbolAnchor(name) if chunk.part is None else PartAnchor(name, chunk.part)
    return ByteAnchor(chunk.start_byte, chunk.end_byte)


def chunk_file(
    source: bytes,
    parsed: ParsedFile | None,
    symbols: Sequence[SymbolRecord] = (),
    schema_nodes: Sequence[SchemaNodeRecord] = (),
    config: ChunkConfig | None = None,
    tokenizer: ChunkTokenizer | None = None,
    file_id: int = 0,
) -> tuple[Chunk, ...]:
    """Cut a file into chunks at AST node bounds, ordered by start byte."""
    chunker = _Chunker(
        source,
        parsed,
        symbols,
        schema_nodes,
        config or ChunkConfig(),
        tokenizer or default_tokenizer(),
    )
    return tuple(chunker.chunk(file_id))


@dataclass(frozen=True, slots=True)
class _Ctx:
    header: str
    coalesce: bool
    members: bool
    kind: str


@dataclass(frozen=True, slots=True)
class _Piece:
    start: int
    end: int
    tokens: int
    kind: str
    header: str
    primary: SymbolRecord | None = None
    also: tuple[SymbolRecord, ...] = ()
    part: int | None = None
    mergeable: bool = False
    oversize: bool = False
    descriptor: str | None = None


class _Chunker:
    def __init__(
        self,
        source: bytes,
        parsed: ParsedFile | None,
        symbols: Sequence[SymbolRecord],
        schema_nodes: Sequence[SchemaNodeRecord],
        config: ChunkConfig,
        tokenizer: ChunkTokenizer,
    ) -> None:
        self.source = source
        self.root = None if parsed is None else parsed.tree.root_node
        self.profile = "text" if parsed is None else parsed.spec.profile
        self.max = config.max_tokens
        self.min = config.min_tokens
        self.tokenizer = tokenizer
        self.symbols = sorted(symbols, key=lambda s: (s.start_byte, -s.end_byte))
        self._starts = [s.start_byte for s in self.symbols]
        self._by_range = {(s.start_byte, s.end_byte): s for s in self.symbols}
        self._parents = {s.parent for s in self.symbols if s.parent is not None}
        self.schema = (
            SchemaView(parsed.spec.lang, schema_nodes, source)
            if parsed is not None and self.profile == "schema"
            else None
        )

    def chunk(self, file_id: int) -> list[Chunk]:
        pieces = sorted(self._cover(self._pieces()), key=lambda p: (p.start, p.end, p.part or 0))
        return [self._to_chunk(piece, file_id) for piece in pieces]

    def _pieces(self) -> list[_Piece]:
        if self.root is None:
            return self._text_pieces()
        if self.schema is not None:
            return self._sequence(
                self.schema.entry_nodes(self.root), _Ctx("", False, False, "keys")
            )
        nodes = list(self.root.named_children)
        if self.profile == "outline":
            return self._sequence(nodes, _Ctx("", False, False, "section"))
        return self._sequence(nodes, _Ctx("", True, True, "glue"))

    def _tokens(self, start: int, end: int) -> int:
        return self.tokenizer.count(self.source[start:end].decode("utf-8", errors="replace"))

    def _text(self, node: Node) -> str:
        return self.source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _sequence(self, nodes: Sequence[Node], ctx: _Ctx) -> list[_Piece]:
        pieces: list[_Piece] = []
        run: list[Node] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            symbol = self._by_range.get((node.start_byte, node.end_byte)) if ctx.members else None
            collapsed = (
                None
                if symbol is not None or self.schema is None
                else self._collapse(nodes, index, ctx)
            )
            if symbol is not None:
                pieces.extend(self._pack(run, ctx))
                run = []
                pieces.extend(self._symbol_pieces(symbol, ctx))
            elif collapsed is not None:
                pieces.extend(self._pack(run, ctx))
                run = []
                pieces.append(collapsed[1])
                index = collapsed[0]
            else:
                run.append(node)
            index += 1
        pieces.extend(self._pack(run, ctx))
        return self._merge(pieces)

    def _collapse(self, nodes: Sequence[Node], index: int, ctx: _Ctx) -> tuple[int, _Piece] | None:
        if self.schema is None:
            return None
        found = self.schema.collapse(nodes, index)
        if found is None:
            return None
        start, end = nodes[index].start_byte, nodes[found.last].end_byte
        size = self._tokens(start, end)
        if size <= self.max:
            return None
        piece = _Piece(start, end, size, "collection", ctx.header, descriptor=found.descriptor)
        return found.last, piece

    def _pack(self, run: list[Node], ctx: _Ctx) -> list[_Piece]:
        if not ctx.coalesce:
            return [piece for node in run for piece in self._unit(node, ctx)]
        pieces: list[_Piece] = []
        group: list[Node] = []
        total = 0
        for node in run:
            size = self._tokens(node.start_byte, node.end_byte)
            if size > self.max or node.type in SECTION_TYPES:
                pieces.extend(self._closed(group, total, ctx))
                group, total = [], 0
                pieces.extend(self._unit(node, ctx))
                continue
            if group:
                gap = self._tokens(group[-1].end_byte, node.start_byte)
                if total + gap + size > self.max:
                    pieces.extend(self._closed(group, total, ctx))
                    group, total = [], 0
                else:
                    total += gap
            group.append(node)
            total += size
        pieces.extend(self._closed(group, total, ctx))
        return pieces

    def _closed(self, group: list[Node], total: int, ctx: _Ctx) -> list[_Piece]:
        if not group:
            return []
        piece = _Piece(
            group[0].start_byte, group[-1].end_byte, total, ctx.kind, ctx.header, mergeable=True
        )
        return [piece]

    def _unit(self, node: Node, ctx: _Ctx) -> list[_Piece]:
        size = self._tokens(node.start_byte, node.end_byte)
        if size <= self.max:
            return [
                _Piece(node.start_byte, node.end_byte, size, ctx.kind, ctx.header, mergeable=True)
            ]
        return self._split_node(node, size, ctx)

    def _split_node(self, node: Node, size: int, ctx: _Ctx) -> list[_Piece]:
        children, header = self._children(node, ctx.header)
        if not children or not self._descends(node):
            return [
                _Piece(node.start_byte, node.end_byte, size, ctx.kind, ctx.header, oversize=True)
            ]
        pieces = self._sequence(children, replace(ctx, header=header, coalesce=True))
        return [replace(piece, mergeable=False) for piece in pieces]

    def _descends(self, node: Node) -> bool:
        if self.schema is not None or self.profile == "outline":
            return True
        index = bisect_left(self._starts, node.start_byte)
        return any(
            symbol.end_byte <= node.end_byte
            for symbol in self.symbols[index:]
            if symbol.start_byte < node.end_byte
        )

    def _children(self, node: Node, header: str) -> tuple[list[Node], str]:
        if self.schema is not None:
            return self.schema.children(node, header)
        named = list(node.named_children)
        if self.profile != "outline":
            return named, header
        if node.type in SECTION_TYPES:
            title = next((self._heading(c) for c in named if c.type in HEADING_TYPES), "")
            return named, _join(header, title)
        tags = [c for c in named if c.type in HTML_TAGS]
        start = next((c for c in tags if c.type != "end_tag"), None)
        if start is None:
            return named, header
        body = [c for c in named if c.type not in HTML_TAGS]
        return body, _join(header, " ".join(self._text(start).split())[:HEADER_CAP])

    def _heading(self, node: Node) -> str:
        first_line = self._text(node).strip().splitlines()[0] if self._text(node).strip() else ""
        return first_line.lstrip("#").strip()[:HEADER_CAP]

    def _symbol_pieces(self, symbol: SymbolRecord, ctx: _Ctx) -> list[_Piece]:
        size = self._tokens(symbol.start_byte, symbol.end_byte)
        if size <= self.max:
            return [
                _Piece(
                    symbol.start_byte,
                    symbol.end_byte,
                    size,
                    symbol.kind,
                    ctx.header,
                    primary=symbol,
                    mergeable=True,
                )
            ]
        return self._split_symbol(symbol, size, ctx)

    def _split_symbol(self, symbol: SymbolRecord, size: int, ctx: _Ctx) -> list[_Piece]:
        whole = _Piece(
            symbol.start_byte,
            symbol.end_byte,
            size,
            symbol.kind,
            ctx.header,
            primary=symbol,
            oversize=True,
        )
        located = self._locate_body(symbol)
        if located is None:
            return [whole]
        owner, body = located
        inner = replace(
            ctx,
            header=_join(ctx.header, symbol.signature, SIGNATURE_JOIN),
            coalesce=True,
            kind="glue",
        )
        container = symbol.kind not in FUNCTION_KINDS and symbol.qualified_name in self._parents
        children = list(body.named_children)
        if not container:
            parts = self._sequence(children, replace(inner, members=False))
            return [
                replace(
                    piece,
                    kind=symbol.kind,
                    primary=symbol,
                    part=number,
                    mergeable=False,
                )
                for number, piece in enumerate(parts, start=1)
            ]
        head = [c for c in owner.children if c.end_byte <= body.start_byte]
        pieces = self._sequence(children, replace(inner, members=True))
        if not head:
            return [whole]
        head_tokens = self._tokens(head[0].start_byte, head[-1].end_byte)
        header_piece = _Piece(
            head[0].start_byte,
            head[-1].end_byte,
            head_tokens,
            symbol.kind,
            ctx.header,
            primary=symbol,
            oversize=head_tokens > self.max,
        )
        return [header_piece, *pieces]

    def _locate_body(self, symbol: SymbolRecord) -> tuple[Node, Node] | None:
        if self.root is None:
            return None
        found = self.root.descendant_for_byte_range(symbol.start_byte, symbol.end_byte)
        level = [] if found is None else [found]
        for _ in range(BODY_SEARCH_DEPTH):
            following: list[Node] = []
            for node in level:
                owner, body = node, node.child_by_field_name("body")
                if body is None:
                    value = node.child_by_field_name("value")
                    owner = value if value is not None else node
                    body = None if value is None else value.child_by_field_name("body")
                if (
                    body is not None
                    and body.named_children
                    and symbol.start_byte <= body.start_byte
                    and body.end_byte <= symbol.end_byte
                ):
                    return owner, body
                following.extend(
                    child
                    for child in node.named_children
                    if symbol.start_byte <= child.start_byte and child.end_byte <= symbol.end_byte
                )
            level = following
        return None

    def _merge(self, pieces: list[_Piece]) -> list[_Piece]:
        merged: list[_Piece] = []
        for piece in pieces:
            if merged and self._can_merge(merged[-1], piece):
                merged[-1] = self._join_pieces(merged[-1], piece)
            else:
                merged.append(piece)
        return merged

    def _can_merge(self, left: _Piece, right: _Piece) -> bool:
        if not (left.mergeable and right.mergeable and left.header == right.header):
            return False
        if left.tokens >= self.min and right.tokens >= self.min:
            return False
        gap = self._tokens(left.end, right.start)
        return left.tokens + gap + right.tokens <= self.max

    def _join_pieces(self, left: _Piece, right: _Piece) -> _Piece:
        gap = self._tokens(left.end, right.start)
        if left.kind == right.kind or right.kind in GLUE_KINDS:
            kind = left.kind
        elif left.kind in GLUE_KINDS:
            kind = right.kind
        else:
            kind = "group"
        return _Piece(
            left.start,
            right.end,
            left.tokens + gap + right.tokens,
            kind,
            left.header,
            mergeable=True,
        )

    def _cover(self, pieces: list[_Piece]) -> list[_Piece]:
        if not self.symbols:
            return pieces
        covered = list(pieces)
        held = {id(symbol) for piece in pieces for symbol in self._members(piece)}
        for symbol in self.symbols:
            if id(symbol) in held:
                continue
            overlapping = [
                index
                for index, piece in enumerate(covered)
                if piece.start < symbol.end_byte and symbol.start_byte < piece.end
            ]
            if overlapping:
                target = covered[overlapping[0]]
                covered[overlapping[0]] = replace(target, also=(*target.also, symbol))
                continue
            size = self._tokens(symbol.start_byte, symbol.end_byte)
            covered.append(
                _Piece(
                    symbol.start_byte,
                    symbol.end_byte,
                    size,
                    symbol.kind,
                    "",
                    primary=symbol,
                    oversize=size > self.max,
                )
            )
        return covered

    def _members(self, piece: _Piece) -> list[SymbolRecord]:
        found: list[SymbolRecord] = [] if piece.primary is None else [piece.primary]
        index = bisect_left(self._starts, piece.start)
        while index < len(self.symbols) and self.symbols[index].start_byte < piece.end:
            symbol = self.symbols[index]
            if symbol.end_byte <= piece.end and symbol is not piece.primary:
                found.append(symbol)
            index += 1
        found.extend(symbol for symbol in piece.also if symbol not in found)
        return found

    def _to_chunk(self, piece: _Piece, file_id: int) -> Chunk:
        members = self._members(piece)
        body = self.source[piece.start : piece.end]
        schema_mode = piece.descriptor is not None
        mode: BodyMode = "schema" if schema_mode else "bytes"
        text = piece.descriptor if piece.descriptor is not None else body.decode(errors="replace")
        return Chunk(
            id=0,
            file_id=file_id,
            kind=piece.kind,
            start_byte=piece.start,
            end_byte=piece.end,
            context_header=piece.header,
            body_mode=mode,
            oversize=piece.oversize,
            content_hash=hashlib.sha256(piece.header.encode() + body).hexdigest(),
            symbol_ids=tuple(s.id for s in members),
            qualified_names=tuple(s.qualified_name for s in members),
            part=piece.part,
            text=text,
        )

    def _text_pieces(self) -> list[_Piece]:
        pieces: list[_Piece] = []
        for start, end in self._blocks():
            size = self._tokens(start, end)
            if size <= self.max:
                pieces.append(_Piece(start, end, size, "text", "", mergeable=True))
            else:
                pieces.extend(self._pack_lines(start, end))
        return self._merge(pieces)

    def _line_spans(self, low: int, high: int) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        position = low
        while position < high:
            newline = self.source.find(b"\n", position, high)
            stop = high if newline < 0 else newline
            end = stop - 1 if stop > position and self.source[stop - 1] == 13 else stop
            spans.append((position, end))
            position = stop + 1
        return spans

    def _blocks(self) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        current: tuple[int, int] | None = None
        for start, end in self._line_spans(0, len(self.source)):
            if not self.source[start:end].strip():
                if current is not None:
                    blocks.append(current)
                current = None
            else:
                current = (start if current is None else current[0], end)
        if current is not None:
            blocks.append(current)
        return blocks

    def _pack_lines(self, low: int, high: int) -> list[_Piece]:
        pieces: list[_Piece] = []
        group: tuple[int, int] | None = None
        total = 0
        for start, end in self._line_spans(low, high):
            size = self._tokens(start, end)
            if group is not None:
                gap = self._tokens(group[1], start)
                if size <= self.max and total + gap + size <= self.max:
                    group, total = (group[0], end), total + gap + size
                    continue
                pieces.append(_Piece(group[0], group[1], total, "text", ""))
                group = None
            if size > self.max:
                pieces.append(_Piece(start, end, size, "text", "", oversize=True))
            else:
                group, total = (start, end), size
        if group is not None:
            pieces.append(_Piece(group[0], group[1], total, "text", ""))
        return pieces


def _join(header: str, addition: str, separator: str = HEADER_JOIN) -> str:
    return f"{header}{separator}{addition}" if header and addition else header or addition
