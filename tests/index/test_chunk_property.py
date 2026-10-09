import json
import tempfile
from itertools import pairwise
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from tree_sitter import Node

from contextweave.core.config import ChunkConfig
from contextweave.index.records import Chunk, FileStamp
from contextweave.index.store import Store
from contextweave.lang.parse import parse_bytes

CONFIG = ChunkConfig(max_tokens=24, min_tokens=6)
PROFILE = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
WORDS = st.sampled_from(["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"])


def indented(lines: list[str]) -> list[str]:
    return [f"    {line}" for line in lines]


@st.composite
def statements(draw: st.DrawFn, depth: int) -> list[str]:
    choice = draw(st.integers(0, 3)) if depth > 0 else 0
    name = draw(WORDS)
    if choice == 0:
        return [f"{name} = {draw(st.integers(0, 999))}"]
    inner: list[str] = []
    for _ in range(draw(st.integers(1, 6))):
        inner.extend(draw(statements(depth - 1)))
    if choice == 1:
        return [f"def {name}_fn(x: int) -> int:", *indented(inner), "    return x"]
    if choice == 2:
        return [f"class {name.title()}K:", *indented(inner)]
    return ["if True:", *indented(inner)]


@st.composite
def python_sources(draw: st.DrawFn) -> str:
    lines: list[str] = []
    for _ in range(draw(st.integers(1, 6))):
        lines.extend(draw(statements(2)))
    return "\n".join(lines) + "\n"


@st.composite
def markdown_sources(draw: st.DrawFn) -> str:
    blocks: list[str] = []
    for _ in range(draw(st.integers(1, 8))):
        level = draw(st.integers(1, 4))
        blocks.append("#" * level + " " + draw(WORDS))
        for _ in range(draw(st.integers(0, 3))):
            if draw(st.booleans()):
                blocks.append(" ".join(draw(st.lists(WORDS, min_size=1, max_size=30))))
            else:
                code = "\n".join(draw(st.lists(WORDS, min_size=1, max_size=8)))
                blocks.append(f"```\n{code}\n```")
    return "\n\n".join(blocks) + "\n"


type JsonValue = bool | int | str | list[JsonValue] | dict[str, JsonValue] | None


@st.composite
def json_values(draw: st.DrawFn, depth: int) -> JsonValue:
    choice = draw(st.integers(0, 3)) if depth > 0 else 0
    if choice == 0:
        return draw(st.one_of(st.integers(-99, 99), WORDS, st.booleans(), st.none()))
    if choice == 1:
        return [draw(json_values(depth - 1)) for _ in range(draw(st.integers(0, 5)))]
    if choice == 2:
        return {draw(WORDS): draw(json_values(depth - 1)) for _ in range(draw(st.integers(1, 4)))}
    count = draw(st.integers(3, 25))
    tags = draw(st.lists(WORDS, max_size=2))
    return [{"id": n, "name": draw(WORDS), "tags": list(tags)} for n in range(count)]


@st.composite
def json_sources(draw: st.DrawFn) -> str:
    value = {draw(WORDS): draw(json_values(3)) for _ in range(draw(st.integers(1, 4)))}
    return json.dumps(value, indent=1) + "\n"


def chunks_for(name: str, source: bytes) -> tuple[Chunk, ...]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with Store.open(root, chunk_config=CONFIG) as store:
            (root / name).write_bytes(source)
            store.upsert_file(name, parse_bytes(name, source), source, FileStamp(1, len(source)))
            return store.chunks_in_file(name)


Index = dict[int, tuple[dict[int, int], dict[int, int]]]


def parent_index(root: Node) -> Index:
    index: Index = {}
    stack = [root]
    while stack:
        node = stack.pop()
        named = node.named_children
        index[node.id] = (
            {child.start_byte: i for i, child in reversed(list(enumerate(named)))},
            {child.end_byte: i for i, child in enumerate(named)},
        )
        stack.extend(named)
    return index


def is_sibling_run(index: Index, start: int, end: int) -> bool:
    for starts, ends in index.values():
        first, last = starts.get(start), ends.get(end)
        if first is not None and last is not None and first <= last:
            return True
    return False


def is_header_prefix(root: Node, raw: bytes, start: int, end: int) -> bool:
    stack = [root]
    while stack:
        node = stack.pop()
        stack.extend(node.named_children)
        if node.start_byte == start and node.end_byte > end:
            for child in node.named_children:
                if child.start_byte >= end and not raw[end : child.start_byte].strip():
                    return True
    return False


def check(name: str, source: str) -> None:
    raw = source.encode()
    parsed = parse_bytes(name, raw).parsed
    assert parsed is not None
    index = parent_index(parsed.tree.root_node)
    chunks = chunks_for(name, raw)
    assert chunks
    for chunk in chunks:
        assert is_sibling_run(index, chunk.start_byte, chunk.end_byte) or (
            chunk.kind != "glue"
            and is_header_prefix(parsed.tree.root_node, raw, chunk.start_byte, chunk.end_byte)
        ), (
            chunk.start_byte,
            chunk.end_byte,
            raw[chunk.start_byte : chunk.end_byte],
        )
    for left, right in pairwise(chunks):
        assert (left.start_byte, left.end_byte) <= (right.start_byte, right.end_byte)
        assert left.end_byte <= right.start_byte


@PROFILE
@given(python_sources())
def test_python_bounds_are_node_runs(source: str) -> None:
    check("gen.py", source)


@PROFILE
@given(markdown_sources())
def test_markdown_bounds_are_node_runs(source: str) -> None:
    check("gen.md", source)


@PROFILE
@given(json_sources())
def test_json_bounds_are_node_runs(source: str) -> None:
    check("gen.json", source)


@PROFILE
@given(st.lists(st.lists(WORDS, min_size=1, max_size=12), min_size=1, max_size=10))
def test_text_bounds_are_blank_line_blocks(blocks: list[list[str]]) -> None:
    source = "\n\n".join("\n".join(line + " word" * 3 for line in block) for block in blocks) + "\n"
    raw = source.encode()
    chunks = chunks_for("notes.txt", raw)
    assert chunks
    for chunk in chunks:
        assert chunk.start_byte == 0 or raw[chunk.start_byte - 1 : chunk.start_byte] == b"\n"
        assert chunk.end_byte == len(raw) or raw[chunk.end_byte : chunk.end_byte + 1] == b"\n"
    for left, right in pairwise(chunks):
        assert left.end_byte <= right.start_byte
