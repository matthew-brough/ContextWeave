import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from contextweave.core.config import ChunkConfig
from contextweave.core.refs import ByteAnchor, PartAnchor, SymbolAnchor
from contextweave.index.chunk import chunk_anchor
from contextweave.index.chunk_tokens import (
    EstimatorTokenizer,
    FileTokenizer,
    default_tokenizer,
    tokenizers,
)
from contextweave.index.records import Chunk, FileStamp
from contextweave.index.root import state_dir
from contextweave.index.store import Store
from contextweave.lang.parse import parse_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lang"
SMALL = ChunkConfig(max_tokens=30, min_tokens=6)


def fixture_files() -> list[Path]:
    return sorted(
        path for path in FIXTURES.rglob("*") if path.is_file() and path.name != "ruff.toml"
    )


def stored_chunks(
    tmp_path: Path, relative: str, source: bytes, config: ChunkConfig
) -> tuple[Chunk, ...]:
    with Store.open(tmp_path, chunk_config=config) as store:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        stat = target.stat()
        store.upsert_file(
            relative,
            parse_bytes(relative, source),
            source,
            FileStamp(stat.st_mtime_ns, stat.st_size),
        )
        return store.chunks_in_file(relative)


def chunks_of(
    tmp_path: Path, relative: str, text: str, config: ChunkConfig = SMALL
) -> tuple[Chunk, ...]:
    return stored_chunks(tmp_path, relative, text.encode(), config)


def real_name(path: Path) -> str:
    return path.name.removesuffix(".txt")


PY_BIG = "\n".join(
    [
        "class Big:",
        '    """doc"""',
        "",
        "    def alpha(self, value: int) -> int:",
        "        total = value + 1",
        "        total = total * 2",
        "        total = total - 3",
        "        return total",
        "",
        "    def beta(self) -> int:",
        "        return 2",
        "",
        "def tiny_a() -> int:",
        "    return 1",
        "",
        "def tiny_b() -> int:",
        "    return 2",
        "",
    ]
)


def test_estimator_registered() -> None:
    assert isinstance(default_tokenizer(), EstimatorTokenizer)
    assert tokenizers.get("bytes35").count("a" * 70) == default_tokenizer().count("a" * 70)


def test_oversize_class_splits_header_and_members(tmp_path: Path) -> None:
    chunks = chunks_of(tmp_path, "big.py", PY_BIG)
    names = [chunk.qualified_names for chunk in chunks]
    assert any("Big" in n and "Big.alpha" not in n for n in names)
    assert any("Big.alpha" in n for n in names)


def test_tiny_siblings_merge(tmp_path: Path) -> None:
    chunks = chunks_of(tmp_path, "big.py", PY_BIG, ChunkConfig(max_tokens=30, min_tokens=14))
    merged = [c for c in chunks if {"tiny_a", "tiny_b"} <= set(c.qualified_names)]
    assert merged


def test_oversize_leaf_single_chunk(tmp_path: Path) -> None:
    source = "X = [" + ", ".join(str(n) for n in range(200)) + "]\n"
    chunks = chunks_of(tmp_path, "const.py", source)
    assert any(chunk.oversize for chunk in chunks)


def test_split_function_parts_and_headers(tmp_path: Path) -> None:
    body = "\n".join(
        f"    if x == {n}:\n        y = {n} * x + {n}\n        x = y" for n in range(12)
    )
    source = f"def long_fn(x: int) -> int:\n    y = 0\n{body}\n    return y\n"
    chunks = chunks_of(tmp_path, "fn.py", source)
    parts = [c for c in chunks if c.part is not None]
    assert len(parts) >= 2
    assert [c.part for c in parts] == list(range(1, len(parts) + 1))
    assert all(c.context_header.startswith("def long_fn") for c in parts)
    assert chunk_anchor(parts[0]) == PartAnchor("long_fn", 1)


def test_anchors(tmp_path: Path) -> None:
    chunks = chunks_of(tmp_path, "big.py", PY_BIG)
    anchors = [chunk_anchor(c) for c in chunks]
    assert SymbolAnchor("Big.beta") in anchors
    nameless = [c for c in chunks if not c.qualified_names]
    for chunk in nameless:
        assert chunk_anchor(chunk) == ByteAnchor(chunk.start_byte, chunk.end_byte)


def test_content_hash_includes_header(tmp_path: Path) -> None:
    source = PY_BIG.encode()
    for chunk in chunks_of(tmp_path, "big.py", PY_BIG):
        expected = hashlib.sha256(
            chunk.context_header.encode() + source[chunk.start_byte : chunk.end_byte]
        ).hexdigest()
        assert chunk.content_hash == expected


def test_schema_collection_collapses(tmp_path: Path) -> None:
    rows = ",\n".join(f'  {{"id": {n}, "name": "item{n}", "tags": ["a", "b"]}}' for n in range(60))
    chunks = chunks_of(tmp_path, "data.json", f'{{"items": [\n{rows}\n]}}\n')
    schema = [c for c in chunks if c.body_mode == "schema"]
    assert len(schema) == 1
    assert "items" in schema[0].text
    assert "item7" not in schema[0].text
    assert "children_count=60" in schema[0].text


def test_markdown_sections(tmp_path: Path) -> None:
    text = "# Top\n\nintro\n\n## One\n\n" + "word " * 40 + "\n\n## Two\n\n" + "more " * 40 + "\n"
    chunks = chunks_of(tmp_path, "doc.md", text)
    assert len(chunks) >= 2
    assert all(chunk.start_byte < chunk.end_byte for chunk in chunks)


def test_text_blocks(tmp_path: Path) -> None:
    text = "\n\n".join("line a\nline b\nline c" for _ in range(20)) + "\n"
    chunks = chunks_of(tmp_path, "notes.txt", text)
    assert len(chunks) > 1
    source = text.encode()
    for chunk in chunks:
        assert chunk.start_byte == 0 or source[chunk.start_byte - 1 : chunk.start_byte] == b"\n"


def test_reupsert_reflects_current_parse(tmp_path: Path) -> None:
    first = chunks_of(tmp_path, "m.py", PY_BIG)
    second = chunks_of(tmp_path, "m.py", "def only() -> int:\n    return 1\n")
    assert len(second) == 1
    assert second[0].qualified_names == ("only",)
    assert len(first) > len(second)


def test_fts_rowid_is_chunk_id(tmp_path: Path) -> None:
    chunks = chunks_of(tmp_path, "big.py", PY_BIG)
    connection = sqlite3.connect(state_dir(tmp_path) / "index.db")
    hits = {
        int(row[0])
        for row in connection.execute("SELECT rowid FROM fts_code WHERE fts_code MATCH 'beta'")
    }
    connection.close()
    expected = {c.id for c in chunks if "Big.beta" in c.qualified_names}
    assert expected
    assert hits == expected


@pytest.mark.parametrize("path", fixture_files(), ids=lambda p: p.relative_to(FIXTURES).as_posix())
def test_fixture_invariants(tmp_path: Path, path: Path) -> None:
    source = path.read_bytes()
    chunks = stored_chunks(tmp_path, real_name(path), source, SMALL)
    assert chunks
    starts = [(c.start_byte, c.end_byte) for c in chunks]
    assert starts == sorted(starts)
    for chunk in chunks:
        assert 0 <= chunk.start_byte < chunk.end_byte <= len(source)
        assert chunk.body_mode in ("bytes", "schema")
        if chunk.kind in ("symbol", "glue") and chunk.body_mode == "bytes":
            assert chunk.text
        if chunk.body_mode == "bytes":
            expected = hashlib.sha256(
                chunk.context_header.encode() + source[chunk.start_byte : chunk.end_byte]
            ).hexdigest()
            assert chunk.content_hash == expected
    anchors = [chunk_anchor(c) for c in chunks]
    positional = [a for a in anchors if not isinstance(a, SymbolAnchor)]
    assert len(set(positional)) == len(positional)


def word_tokenizer(text: str) -> FileTokenizer:
    vocab = {word: index for index, word in enumerate([*sorted(set(text.split())), "[UNK]"])}
    spec = {
        "version": "1.0",
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
        "pre_tokenizer": {"type": "WhitespaceSplit"},
    }
    return FileTokenizer(Tokenizer.from_str(json.dumps(spec)))


def test_file_tokenizer_spans_cover_text_within_limit() -> None:
    text = " ".join(f"w{index}" for index in range(25))
    tokenizer = word_tokenizer(text)
    spans = tokenizer.spans(text, 4)
    assert "".join(text[start:end] for start, end in spans) == text
    assert all(tokenizer.count(text[start:end]) <= 4 for start, end in spans)
    assert len(spans) == 7


def test_file_tokenizer_spans_prefer_a_line_end() -> None:
    text = "a b c\nd e f g h"
    spans = word_tokenizer(text).spans(text, 4)
    assert spans[0] == (0, 6)
