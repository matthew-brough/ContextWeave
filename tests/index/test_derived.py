import sqlite3
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import pytest

from contextweave.index.derived import (
    DerivedAnnotation,
    DerivedCache,
    cached_or_compute,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def cache(tmp_path: Path, clock: Clock) -> DerivedCache:
    return DerivedCache(tmp_path / "cache" / "contextweave" / "derived.db", clock)


def test_wal_and_tables(cache: DerivedCache, tmp_path: Path) -> None:
    path = tmp_path / "cache" / "contextweave" / "derived.db"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone() == ("wal",)


def test_annotation_roundtrip(cache: DerivedCache) -> None:
    note = DerivedAnnotation("parses ref", ("ref", "parse"))
    cache.put_annotation("h1", "m", "p1", note)
    assert cache.get_annotation("h1", "m", "p1") == note


def test_annotation_key_completeness(cache: DerivedCache) -> None:
    cache.put_annotation("h1", "m", "p1", DerivedAnnotation("x", ()))
    assert cache.get_annotation("h1", "m2", "p1") is None
    assert cache.get_annotation("h1", "m", "p2") is None
    assert cache.get_annotation("h2", "m", "p1") is None


def test_vector_roundtrip_float32(cache: DerivedCache) -> None:
    cache.put_vector("h1", "code", "m", 3, [0.5, -1.0, 2.0])
    assert cache.get_vectors(["h1"], "code", "m", 3) == {"h1": (0.5, -1.0, 2.0)}


def test_vector_key_completeness(cache: DerivedCache) -> None:
    cache.put_vector("h1", "code", "m", 2, [1.0, 2.0])
    assert cache.get_vectors(["h1"], "code", "m2", 2) == {}
    assert cache.get_vectors(["h1"], "code", "m", 4) == {}
    assert cache.get_vectors(["h1"], "doc", "m", 2) == {}


def test_vectors_batch_only_exact_hashes(cache: DerivedCache) -> None:
    for name in ("a", "b", "c"):
        cache.put_vector(name, "s", "m", 1, [1.0])
    assert set(cache.get_vectors(["a", "c", "zz"], "s", "m", 1)) == {"a", "c"}
    assert cache.get_vectors([], "s", "m", 1) == {}


def test_large_batch(cache: DerivedCache) -> None:
    hashes = [f"h{i}" for i in range(1000)]
    for h in hashes:
        cache.put_vector(h, "s", "m", 1, [1.0])
    assert len(cache.get_vectors(hashes, "s", "m", 1)) == 1000


def test_put_vector_dim_mismatch(cache: DerivedCache) -> None:
    with pytest.raises(ValueError):
        cache.put_vector("h", "s", "m", 3, [1.0])


def test_no_project_column(cache: DerivedCache, tmp_path: Path) -> None:
    path = tmp_path / "cache" / "contextweave" / "derived.db"
    with sqlite3.connect(path) as db:
        for table in ("derived_annotations", "derived_vectors"):
            columns = [row[1] for row in db.execute(f"PRAGMA table_info({table})")]
            assert "project" not in columns and "key" not in columns


def test_gc_by_registered_hashes(cache: DerivedCache) -> None:
    for h in ("keep", "drop"):
        cache.put_annotation(h, "m", "p", DerivedAnnotation("x", ()))
        cache.put_vector(h, "s", "m", 1, [1.0])
    result = cache.gc(lambda: iter(["keep"]))
    assert (result.annotations, result.vectors) == (1, 1)
    assert cache.get_annotation("keep", "m", "p") is not None
    assert cache.get_annotation("drop", "m", "p") is None
    assert set(cache.get_vectors(["keep", "drop"], "s", "m", 1)) == {"keep"}


def test_gc_by_age(cache: DerivedCache, clock: Clock) -> None:
    cache.put_vector("old", "s", "m", 1, [1.0])
    clock.now += 10 * 86400
    cache.put_vector("new", "s", "m", 1, [1.0])
    result = cache.gc(lambda: ["old", "new"], timedelta(days=5))
    assert result.vectors == 1
    assert set(cache.get_vectors(["old", "new"], "s", "m", 1)) == {"new"}


def test_hit_refreshes_last_used_so_gc_keeps(cache: DerivedCache, clock: Clock) -> None:
    cache.put_annotation("h", "m", "p", DerivedAnnotation("x", ()))
    clock.now += 10 * 86400
    assert cache.get_annotation("h", "m", "p") is not None
    assert cache.gc(lambda: ["h"], timedelta(days=5)).annotations == 0


def test_gc_empty_registered_clears_all(cache: DerivedCache) -> None:
    cache.put_vector("a", "s", "m", 1, [1.0])
    assert cache.gc(lambda: []).vectors == 1


def test_cached_or_compute_avoids_compute(cache: DerivedCache) -> None:
    calls: list[list[str]] = []

    def compute(keys: Sequence[str]) -> dict[str, tuple[float, ...]]:
        calls.append(list(keys))
        return {k: (float(len(k)),) for k in keys}

    def lookup(keys: Sequence[str]) -> dict[str, tuple[float, ...]]:
        return cache.get_vectors(keys, "s", "m", 1)

    def store(key: str, value: tuple[float, ...]) -> None:
        cache.put_vector(key, "s", "m", 1, value)

    first = cached_or_compute(["a", "bb"], lookup, compute, store)
    second = cached_or_compute(["a", "bb", "ccc"], lookup, compute, store)
    assert first == {"a": (1.0,), "bb": (2.0,)}
    assert second == {"a": (1.0,), "bb": (2.0,), "ccc": (3.0,)}
    assert calls == [["a", "bb"], ["ccc"]]
    cached_or_compute(["a", "bb", "ccc"], lookup, compute, store)
    assert len(calls) == 2
