from dataclasses import replace
from pathlib import Path

import pytest
from fake_embed import FakeEmbed, config_for, embed_config, slots_for
from retrieve_support import ingest, make_ctx

from contextweave.core.config import Config
from contextweave.index.derived import DerivedCache
from contextweave.index.embed import embed_project
from contextweave.index.store import Store
from contextweave.index.vectors import (
    CODE,
    VectorInput,
    VectorPart,
    ensure_space,
    vector_id,
    write_vectors,
)
from contextweave.models.slots import ModelSlots
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.context import Mode
from contextweave.retrieve.retrieve_stage import RetrieveStage
from contextweave.retrieve.stages import QueryContext
from contextweave.retrieve.vector import knn

AUTH = (
    "def check_credentials(user, password):\n    return verify_hash(user.password_hash, password)\n"
)
CHART = "def draw_chart(series):\n    return render_axes(series)\n"
PARSER = "def parse_config(text):\n    return load_toml(text)\n"


@pytest.fixture
def cache(tmp_path: Path) -> DerivedCache:
    return DerivedCache(tmp_path / "derived" / "derived.db")


@pytest.fixture
def fake() -> FakeEmbed:
    return FakeEmbed(embed_config())


@pytest.fixture
def config(fake: FakeEmbed) -> Config:
    return config_for(fake.config)


@pytest.fixture
def slots(fake: FakeEmbed, config: Config) -> ModelSlots:
    return slots_for(fake, config)


@pytest.fixture
def embedded(store: Store, slots: ModelSlots, cache: DerivedCache, config: Config) -> Store:
    ingest(store, "auth.py", AUTH)
    ingest(store, "chart.py", CHART)
    ingest(store, "conf.py", PARSER)
    embed_project(store, slots, cache, config)
    return store


def retrieve(
    store: Store,
    query: str,
    mode: Mode,
    config: Config,
    slots: ModelSlots | None,
    scope: str | None = None,
) -> tuple[QueryContext, CandidateSet]:
    ctx = make_ctx(store, query, config=config, mode=mode, models=slots, scope=scope)
    return ctx, RetrieveStage().run(ctx, CandidateSet())


def test_nearest_vector_ranks_matching_chunk_first(
    embedded: Store, config: Config, slots: ModelSlots
) -> None:
    ctx, cands = retrieve(embedded, "verify the user password", "semantic", config, slots)
    assert ctx.envelope.degraded == []
    assert cands[0].path == "auth.py"
    assert cands[0].ranks.semantic == 1
    assert [c.ranks.semantic for c in cands] == list(range(1, len(cands) + 1))


def test_query_embedded_with_query_kind(
    embedded: Store, fake: FakeEmbed, config: Config, slots: ModelSlots
) -> None:
    fake.calls.clear()
    retrieve(embedded, "draw a chart", "semantic", config, slots)
    assert fake.texts == ["draw a chart"]


def test_auto_merges_semantic_with_lexical(
    embedded: Store, config: Config, slots: ModelSlots
) -> None:
    _, cands = retrieve(embedded, "password", "auto", config, slots)
    top = cands[0]
    assert top.path == "auth.py"
    assert top.ranks.lexical is not None and top.ranks.semantic is not None


def test_scope_restricts_vector_hits(embedded: Store, config: Config, slots: ModelSlots) -> None:
    _, cands = retrieve(embedded, "verify the user password", "semantic", config, slots, "chart.py")
    assert [c.path for c in cands] == ["chart.py"]


def test_down_slot_skips_vec_and_lexical_still_answers(
    embedded: Store, fake: FakeEmbed, config: Config, slots: ModelSlots
) -> None:
    fake.down = True
    ctx, cands = retrieve(embedded, "password", "auto", config, slots)
    assert [entry["stage"] for entry in ctx.envelope.degraded] == ["retrieve"]
    assert ctx.envelope.degraded[0]["reason"].startswith("skip_vec")
    assert cands[0].path == "auth.py"
    assert all(c.ranks.semantic is None for c in cands)
    assert cands[0].ranks.lexical is not None


def test_semantic_mode_with_down_slot_returns_nothing_without_raising(
    embedded: Store, fake: FakeEmbed, config: Config, slots: ModelSlots
) -> None:
    fake.down = True
    ctx, cands = retrieve(embedded, "password", "semantic", config, slots)
    assert len(cands) == 0
    assert ctx.envelope.errors == []
    assert ctx.envelope.degraded[0]["reason"].startswith("skip_vec")


def test_model_switch_skips_vec_while_rebuilding(embedded: Store) -> None:
    other = FakeEmbed(embed_config("fake-b"))
    config = config_for(other.config)
    ctx, cands = retrieve(embedded, "password", "auto", config, slots_for(other, config))
    assert {"stage": "retrieve", "reason": "vec_rebuilding"} in ctx.envelope.degraded
    assert cands[0].ranks.lexical is not None and cands[0].ranks.semantic is None
    assert other.calls == []


def test_unembedded_project_reports_vec_unavailable(
    store: Store, config: Config, slots: ModelSlots
) -> None:
    ingest(store, "auth.py", AUTH)
    ctx, cands = retrieve(store, "password", "auto", config, slots)
    assert {"stage": "retrieve", "reason": "vec_unavailable"} in ctx.envelope.degraded
    assert len(cands) > 0


def test_no_models_is_silent_in_auto(store: Store, config: Config) -> None:
    ingest(store, "auth.py", AUTH)
    ctx, cands = retrieve(store, "password", "auto", replace(config), None)
    assert ctx.envelope.degraded == []
    assert len(cands) > 0


def chunk_ids(store: Store) -> dict[str, int]:
    rows = store.query("SELECT f.path, c.id FROM chunks c JOIN files f ON f.id = c.file_id")
    return {str(path): int(str(chunk_id)) for path, chunk_id in rows}


def test_knn_ranks_chunk_by_closest_part_and_reports_it(store: Store) -> None:
    ingest(store, "auth.py", AUTH)
    ingest(store, "chart.py", CHART)
    ids = chunk_ids(store)
    space = ensure_space(store, CODE, "m", 2)
    far, near, mid = [0.0, 1.0], [1.0, 0.0], [0.7, 0.7]
    write_vectors(
        store,
        space,
        [
            (
                VectorInput(ids["auth.py"], "a", "x" * 20),
                (VectorPart(0, 10, far), VectorPart(10, 20, near)),
            ),
            (VectorInput(ids["chart.py"], "c", "y"), (VectorPart(0, 1, mid),)),
        ],
    )
    hits = knn(store, space, [1.0, 0.0], scope=None, limit=2)
    assert [(chunk_id, part) for chunk_id, _, part in hits] == [
        (ids["auth.py"], 1),
        (ids["chart.py"], 0),
    ]
    assert hits[0][1] == pytest.approx(0.0, abs=1e-6)


def test_knn_refetches_until_enough_distinct_chunks(store: Store) -> None:
    ingest(store, "auth.py", AUTH)
    ingest(store, "chart.py", CHART)
    ids = chunk_ids(store)
    space = ensure_space(store, CODE, "m", 2)
    many = tuple(VectorPart(index, index + 1, [1.0, 0.001 * index]) for index in range(16))
    write_vectors(
        store,
        space,
        [
            (VectorInput(ids["auth.py"], "a", "x" * 16), many),
            (VectorInput(ids["chart.py"], "c", "y"), (VectorPart(0, 1, [0.0, 1.0]),)),
        ],
    )
    hits = knn(store, space, [1.0, 0.0], scope=None, limit=2)
    assert [chunk_id for chunk_id, _, _ in hits] == [ids["auth.py"], ids["chart.py"]]


def test_rewrite_drops_stale_parts(store: Store) -> None:
    ingest(store, "auth.py", AUTH)
    chunk_id = chunk_ids(store)["auth.py"]
    space = ensure_space(store, CODE, "m", 2)
    item = VectorInput(chunk_id, "a", "x" * 3)
    parts = tuple(VectorPart(index, index + 1, [1.0, float(index)]) for index in range(3))
    write_vectors(store, space, [(item, parts)])
    write_vectors(store, space, [(item, (VectorPart(0, 3, [1.0, 0.0]),))])
    assert store.query(f"SELECT rowid FROM {space.name}") == [(vector_id(chunk_id, 0),)]
    assert store.query("SELECT count(*) FROM vector_parts") == [(0,)]


def test_semantic_hit_on_a_later_part_carries_its_section(
    embedded: Store, config: Config, slots: ModelSlots, cache: DerivedCache
) -> None:
    embed_project(embedded, slots, cache, config, max_input_tokens=8)
    ctx, cands = retrieve(embedded, "verify the user password", "semantic", config, slots)
    assert ctx.envelope.degraded == []
    sections = [cand.section for cand in cands]
    assert all(section is not None for section in sections)
    assert all(start < end for start, end in filter(None, sections))
