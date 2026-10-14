from collections.abc import Iterator
from pathlib import Path

import pytest
from retrieve_support import ingest, make_ctx

from contextweave.core.config import ChunkConfig
from contextweave.index.store import Store
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.freshness import freshen
from contextweave.retrieve.lexical import lexical_search
from contextweave.retrieve.retrieve_stage import RetrieveStage


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path, chunk_config=ChunkConfig(max_tokens=14, min_tokens=1)) as opened:
        yield opened


ORIGINAL = "def alpha():\n    return 1\n\n\ndef beta_target():\n    return 2\n"


def seed(store: Store) -> None:
    ingest(store, "mod.py", ORIGINAL)


def test_fresh_files_untouched(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "beta_target")
    found = CandidateSet.of(lexical_search(store, "beta_target", scope=None, limit=10))
    assert freshen(store, found, ctx.envelope, "retrieve") == found
    assert ctx.envelope.degraded == []


def test_edited_file_remaps_to_new_chunk(store: Store) -> None:
    seed(store)
    found = lexical_search(store, "beta_target", scope=None, limit=10)
    before = found[0]
    (store.root / "mod.py").write_text("# header comment\n\n" + ORIGINAL, encoding="utf-8")
    ctx = make_ctx(store, "beta_target")
    remapped = freshen(store, CandidateSet.of(found), ctx.envelope, "retrieve")
    assert len(remapped) == 1
    after = remapped[0]
    assert after.anchor == before.anchor
    assert after.start_byte > before.start_byte
    assert after.ranks == before.ranks
    source = (store.root / "mod.py").read_bytes()
    assert b"beta_target" in source[after.start_byte : after.end_byte]


def test_removed_anchor_dropped(store: Store) -> None:
    seed(store)
    found = lexical_search(store, "beta_target", scope=None, limit=10)
    (store.root / "mod.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    ctx = make_ctx(store, "beta_target")
    assert len(freshen(store, CandidateSet.of(found), ctx.envelope, "retrieve")) == 0


def test_deleted_file_dropped(store: Store) -> None:
    seed(store)
    found = lexical_search(store, "beta_target", scope=None, limit=10)
    (store.root / "mod.py").unlink()
    ctx = make_ctx(store, "beta_target")
    assert len(freshen(store, CandidateSet.of(found), ctx.envelope, "retrieve")) == 0
    assert store.get_file("mod.py") is None


def test_retrieve_stage_refreshes_before_returning(store: Store) -> None:
    seed(store)
    (store.root / "mod.py").write_text(ORIGINAL.replace("beta_target", "gamma"), encoding="utf-8")
    ctx = make_ctx(store, "beta_target", mode="lexical")
    assert len(RetrieveStage().run(ctx, CandidateSet())) == 0


def test_readonly_store_degrades_on_stale(tmp_path: Path) -> None:
    with Store.open(tmp_path) as writer:
        seed(writer)
        found = lexical_search(writer, "beta_target", scope=None, limit=10)
    (tmp_path / "mod.py").write_text("# x\n" + ORIGINAL, encoding="utf-8")
    with Store.open(tmp_path, readonly=True) as reader:
        ctx = make_ctx(reader, "beta_target")
        assert len(freshen(reader, CandidateSet.of(found), ctx.envelope, "retrieve")) == 0
        assert ctx.envelope.degraded == [{"stage": "retrieve", "reason": "stale_readonly"}]
