import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st
from retrieve_support import ingest, make_ctx

from contextweave.core.budget import measure_payload
from contextweave.core.config import Config, GateConfig
from contextweave.index.edges import resolve_edges
from contextweave.index.store import Store
from contextweave.retrieve.assemble import MIN_MAX_TOKENS
from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.context import Plan, QueryContext
from contextweave.retrieve.expand import Expanded, ExpandStage, proximity
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int
from contextweave.retrieve.search import SearchOptions, context_search, search

FILES = {
    "pkg/a.py": "from pkg.b import middle\n\n\ndef top() -> int:\n    return middle()\n",
    "pkg/b.py": "from pkg.c import leaf\n\n\ndef middle() -> int:\n    return leaf()\n",
    "pkg/c.py": "def leaf() -> int:\n    return 1\n\n\ndef lone() -> int:\n    return 2\n",
}


def build(store: Store, config: Config | None = None) -> QueryContext:
    for path, text in FILES.items():
        ingest(store, path, text)
    resolve_edges(store, ())
    ctx = make_ctx(store, "top", config=config)
    ctx.plan.value = Plan("debug", rels=("callees",))
    return ctx


def chunk_for(store: Store, path: str) -> int:
    rows = store.query(
        "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?"
        " ORDER BY c.start_byte",
        (path,),
    )
    return as_int(rows[0][0])


def seeds(store: Store, gate: float = 0.9) -> CandidateSet:
    loaded = load_candidates(store, [chunk_for(store, "pkg/a.py")])
    return CandidateSet.of(
        cand.with_scores(replace(cand.scores, gate=gate)) for cand in loaded.values()
    )


def by_path(cands: CandidateSet) -> dict[str, Candidate]:
    return {cand.path: cand for cand in cands}


def test_walk_adds_callees_within_max_depth(store: Store) -> None:
    ctx = build(store)
    out = by_path(ExpandStage().run(ctx, seeds(store)))
    assert set(out) == {"pkg/a.py", "pkg/b.py", "pkg/c.py"}


def test_graph_proximity_is_one_over_one_plus_hops(store: Store) -> None:
    ctx = build(store)
    out = by_path(ExpandStage().run(ctx, seeds(store)))
    assert out["pkg/a.py"].scores.graph_proximity == 1.0
    assert out["pkg/b.py"].scores.graph_proximity == proximity(1) == 0.5
    assert out["pkg/c.py"].scores.graph_proximity == proximity(2)


def test_fallback_gate_is_half_seed_gate_over_hops(store: Store) -> None:
    ctx = build(store)
    out = by_path(ExpandStage().run(ctx, seeds(store, 0.8)))
    assert out["pkg/b.py"].scores.gate == 0.5 * 0.8
    assert out["pkg/c.py"].scores.gate == 0.5 * 0.8 / 2
    assert out["pkg/b.py"].why == "callees of top"


def test_edges_link_served_chunks(store: Store) -> None:
    ctx = build(store)
    out = by_path(ExpandStage().run(ctx, seeds(store)))
    (edge,) = out["pkg/a.py"].edges
    assert (edge.rel, edge.resolution) == ("callees", "exact")
    assert edge.to_ref.startswith(f"cw://{ctx.project_key}/pkg/b.py#")


def test_below_min_score_is_not_a_seed(store: Store) -> None:
    ctx = build(store)
    before = seeds(store, 0.3)
    after = ExpandStage().run(ctx, before)
    assert [cand.chunk_id for cand in after] == [cand.chunk_id for cand in before]
    assert all(not cand.edges for cand in after)


def test_no_rels_leaves_input_unchanged(store: Store) -> None:
    ctx = build(store)
    ctx.plan.value = Plan("lookup")
    before = seeds(store)
    assert ExpandStage().run(ctx, before) is before


def test_expand_window_caps_new_candidates(store: Store) -> None:
    config = replace(Config(), gate=GateConfig(expand_window=1))
    ctx = build(store, config)
    out = ExpandStage().run(ctx, seeds(store))
    assert len(out) == 2


def test_gate_pass_two_hook_is_pluggable(store: Store) -> None:
    ctx = build(store)
    calls: list[int] = []

    def gate_pass(ctx: QueryContext, expanded: Sequence[Expanded]) -> list[Candidate]:
        calls.append(len(expanded))
        return [
            replace(item.cand, scores=replace(item.cand.scores, gate=0.123)) for item in expanded
        ]

    out = by_path(ExpandStage(gate_pass).run(ctx, seeds(store)))
    assert calls == [2]
    assert out["pkg/b.py"].scores.gate == 0.123


@given(st.integers(min_value=0, max_value=50))
def test_proximity_in_unit_interval_and_decreasing(hops: int) -> None:
    assert 0 < proximity(hops) <= 1
    assert proximity(hops + 1) < proximity(hops)


@given(budget=st.integers(MIN_MAX_TOKENS, 1500))
@settings(max_examples=25, deadline=None)
def test_pack_with_edges_respects_ceiling(budget: int) -> None:
    with tempfile.TemporaryDirectory() as directory, Store.open(Path(directory)) as store:
        for path, text in FILES.items():
            ingest(store, path, text)
        resolve_edges(store, ())
        result = search(store, Config(), "top middle leaf", SearchOptions(max_tokens=budget))
    assert measure_payload(result.pack, result.text) <= budget
    assert result.pack["budget"]["served_tokens"] <= budget


def test_search_pack_carries_edges_when_intent_walks_the_graph(store: Store) -> None:
    for path, text in FILES.items():
        ingest(store, path, text)
    resolve_edges(store, ())
    pack = context_search(store, Config(), "top", SearchOptions(intent="debug", max_tokens=8000))
    served = {item["ref"] for item in pack["items"]}
    edges = [edge for item in pack["items"] for edge in item.get("edges", [])]
    assert all(edge["to_ref"] in served for edge in edges)
