from dataclasses import replace
from pathlib import Path

from contextweave.compact.sources import SourceCache
from contextweave.core.config import ChunkConfig, Config
from contextweave.core.refs import ByteAnchor, PartAnchor, SymbolAnchor, parse_ref
from contextweave.index.records import FileStamp
from contextweave.index.store import Store, hash_bytes
from contextweave.lang.parse import parse_bytes
from contextweave.retrieve.candidates import Candidate, Scores
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.pack import PackItem

HASH_LENGTH = 12
SMALL = ChunkConfig(max_tokens=30, min_tokens=6)


def ingest(store: Store, relative: str, text: str) -> None:
    target = store.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    stat = target.stat()
    source = text.encode()
    store.upsert_file(
        relative, parse_bytes(relative, source), source, FileStamp(stat.st_mtime_ns, stat.st_size)
    )


def make_ctx(store: Store, query: str = "x", max_tokens: int = 6000) -> QueryContext:
    from contextweave.core.refs import project_key

    return QueryContext(
        query=query,
        store=store,
        config=Config(),
        project_key=project_key(store.root),
        root=store.root,
        max_tokens=max_tokens,
    )


def span(cand: Candidate, start: int, end: int, final: float = 0.5) -> Candidate:
    return replace(
        cand,
        start_byte=start,
        end_byte=end,
        anchor=ByteAnchor(start, end),
        qualified_name=None,
        scores=Scores(final=final),
    )


def resolve_ref(store: Store, ref_text: str) -> bytes | None:
    """The bytes a ref expands to, or None when its content hash no longer matches."""
    ref = parse_ref(ref_text)
    source = (store.root / ref.path).read_bytes()
    match ref.anchor:
        case ByteAnchor(start, end):
            body = source[start:end]
            return body if hash_bytes(body)[:HASH_LENGTH] == ref.content_hash else None
        case SymbolAnchor(name):
            for symbol in store.symbols_in_file(ref.path):
                if (
                    symbol.qualified_name == name
                    and symbol.content_hash[:HASH_LENGTH] == ref.content_hash
                ):
                    return source[symbol.start_byte : symbol.end_byte]
        case PartAnchor(name, part):
            for chunk in store.chunks_in_file(ref.path):
                if (
                    chunk.part == part
                    and chunk.qualified_names[:1] == (name,)
                    and chunk.content_hash[:HASH_LENGTH] == ref.content_hash
                ):
                    return source[chunk.start_byte : chunk.end_byte]
    return None


def sources_for(root: Path, files: dict[str, str]) -> SourceCache:
    cache = SourceCache(root)
    for path, text in files.items():
        cache.preload(path, text.encode())
    return cache


def pack_item(cand: Candidate, text: str, ref: str) -> PackItem:
    return PackItem(
        cand=cand,
        ref=ref,
        rep="source",
        text=text,
        signature=None,
        start_line=1,
        end_line=1,
        oversize=False,
    )
