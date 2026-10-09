from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from retrieve_support import ingest

from contextweave.index.chunk import chunk_anchor
from contextweave.index.store import Store
from contextweave.retrieve.lexical import lexical_search

HOSTILE = [
    '"',
    '""',
    "'",
    "(",
    ")",
    "*",
    "-",
    "NEAR",
    "NEAR(a b)",
    "a AND",
    "OR OR",
    "NOT x",
    "col:val",
    "^start",
    "foo*bar",
    "\x00",
    "ünïcode ²",
    "{a b}",
]


def seed(store: Store) -> None:
    ingest(store, "pkg/names.py", 'def retry_backoff(delay):\n    """Wait."""\n    return delay\n')
    ingest(
        store,
        "pkg/body.py",
        "def unrelated(value):\n    # retry_backoff retry_backoff retry_backoff\n    return value\n",
    )
    ingest(store, "docs/guide.py", "def helper():\n    return 1\n")


def paths(store: Store, query: str) -> list[str]:
    hits = lexical_search(store, query, scope=None, limit=100)
    return [hit.path for hit in hits]


def test_name_hit_outranks_body_hit(store: Store) -> None:
    seed(store)
    hits = lexical_search(store, "retry_backoff", scope=None, limit=100)
    assert [hit.path for hit in hits][:2] == ["pkg/names.py", "pkg/body.py"]
    assert [hit.ranks.lexical for hit in hits] == list(range(1, len(hits) + 1))


def test_camel_query_matches_snake_symbol(store: Store) -> None:
    seed(store)
    assert paths(store, "retryBackoff")[0] == "pkg/names.py"


def test_natural_language_query(store: Store) -> None:
    seed(store)
    assert paths(store, "where is the retry backoff")[0] == "pkg/names.py"


def test_candidate_fields_and_anchor(store: Store) -> None:
    seed(store)
    hit = lexical_search(store, "retry_backoff", scope=None, limit=5)[0]
    chunk = store.chunks_in_file("pkg/names.py")[0]
    assert hit.chunk_id == chunk.id
    assert hit.anchor == chunk_anchor(chunk)
    assert (hit.start_byte, hit.end_byte) == (chunk.start_byte, chunk.end_byte)
    assert hit.qualified_name == "retry_backoff"
    assert hit.profile == "symbols"
    assert hit.kind == "function"


def test_scope_prefix(store: Store) -> None:
    seed(store)
    assert set(paths(store, "retry_backoff helper")) == {
        "pkg/names.py",
        "pkg/body.py",
        "docs/guide.py",
    }
    scoped = lexical_search(store, "retry_backoff helper", scope="pkg", limit=100)
    assert {hit.path for hit in scoped} == {"pkg/names.py", "pkg/body.py"}
    assert lexical_search(store, "retry_backoff", scope="pk", limit=100) == ()


def test_scope_glob(store: Store) -> None:
    seed(store)
    scoped = lexical_search(store, "retry_backoff helper", scope="docs/**/*.py", limit=100)
    assert {hit.path for hit in scoped} == {"docs/guide.py"}
    scoped = lexical_search(store, "retry_backoff helper", scope="**/names.py", limit=100)
    assert {hit.path for hit in scoped} == {"pkg/names.py"}


def test_limit_caps(store: Store) -> None:
    for index in range(6):
        ingest(store, f"m{index}.py", f"def shared_{index}():\n    return 'shared'\n")
    assert len(lexical_search(store, "shared", scope=None, limit=3)) == 3
    assert lexical_search(store, "shared", scope=None, limit=0) == ()


def test_ties_break_by_path_then_start_byte(store: Store) -> None:
    body = "def same_name():\n    return 1\n\n\ndef same_name_two():\n    return 1\n"
    ingest(store, "z/dup.py", body)
    ingest(store, "a/dup.py", body)
    ingest(store, "m/dup.py", body)
    first = [
        (h.path, h.start_byte) for h in lexical_search(store, "same_name", scope=None, limit=50)
    ]
    assert first == sorted(first)
    again = [
        (h.path, h.start_byte) for h in lexical_search(store, "same_name", scope=None, limit=50)
    ]
    assert first == again


@pytest.mark.parametrize("query", HOSTILE)
def test_hostile_queries_never_raise(store: Store, query: str) -> None:
    seed(store)
    lexical_search(store, query, scope=None, limit=10)


@pytest.fixture(scope="module")
def shared_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    root: Path = tmp_path_factory.mktemp("hostile")
    with Store.open(root) as opened:
        seed(opened)
        yield opened


@settings(max_examples=300, deadline=None)
@given(st.text())
def test_arbitrary_text_never_raises(shared_store: Store, text: str) -> None:
    lexical_search(shared_store, text, scope=None, limit=10)
    lexical_search(shared_store, text, scope=text, limit=10)
