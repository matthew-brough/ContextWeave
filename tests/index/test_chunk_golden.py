import json
import os
from pathlib import Path

import pytest

from contextweave.core.config import ChunkConfig
from contextweave.core.refs import ByteAnchor, PartAnchor, SymbolAnchor
from contextweave.index.chunk import chunk_anchor
from contextweave.index.records import Chunk, FileStamp
from contextweave.index.store import Store
from contextweave.lang.parse import parse_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lang"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "chunk"
STORAGE_SUFFIX = ".txt"

type Json = dict[str, str | int | bool | None]


def fixture_names() -> list[str]:
    return sorted(
        path.relative_to(FIXTURES).as_posix().removesuffix(STORAGE_SUFFIX)
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name != "ruff.toml"
    )


def read_fixture(name: str) -> bytes:
    plain = FIXTURES / name
    return (plain if plain.is_file() else plain.with_name(plain.name + STORAGE_SUFFIX)).read_bytes()


def golden_name(name: str) -> str:
    return name.replace("/", "__") + ".json"


def anchor_text(chunk: Chunk) -> str:
    anchor = chunk_anchor(chunk)
    match anchor:
        case SymbolAnchor(name):
            return name
        case PartAnchor(name, part):
            return f"{name}~{part}"
        case ByteAnchor(start, end):
            return f"b{start}-{end}"
    raise AssertionError(anchor)


def snapshot(chunk: Chunk) -> Json:
    return {
        "start_byte": chunk.start_byte,
        "end_byte": chunk.end_byte,
        "kind": chunk.kind,
        "body_mode": chunk.body_mode,
        "oversize": chunk.oversize,
        "part": chunk.part,
        "anchor": anchor_text(chunk),
    }


def chunk_fixture(tmp_path: Path, name: str) -> list[Json]:
    source = read_fixture(name)
    with Store.open(tmp_path, chunk_config=ChunkConfig()) as store:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
        store.upsert_file(name, parse_bytes(name, source), source, FileStamp(1, len(source)))
        return [snapshot(chunk) for chunk in store.chunks_in_file(name)]


@pytest.mark.parametrize("name", fixture_names())
def test_chunks_match_golden(tmp_path: Path, name: str) -> None:
    actual = chunk_fixture(tmp_path, name)
    path = GOLDEN / golden_name(name)
    if os.environ.get("CW_UPDATE_GOLDEN"):
        GOLDEN.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=2) + "\n", encoding="utf-8")
    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_no_orphan_chunk_goldens() -> None:
    expected = {golden_name(name) for name in fixture_names()}
    assert {p.name for p in GOLDEN.glob("*.json")} == expected
