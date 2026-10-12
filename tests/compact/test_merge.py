from dataclasses import replace
from pathlib import Path

from compact_support import ingest, make_ctx, resolve_ref, span

from contextweave.compact.rules import compact_candidates
from contextweave.compact.sources import SourceCache
from contextweave.core.refs import ByteAnchor, parse_ref
from contextweave.index.store import Store
from contextweave.retrieve.assemble import AssembleStage
from contextweave.retrieve.candidates import CandidateEdge, CandidateSet, Scores
from contextweave.retrieve.hydrate import load_file_candidates

TWO = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
BODY = "\n".join(f"    if x == {n}:\n        y = {n} * x + {n}\n        x = y" for n in range(12))
BIG = f"def long_fn(x: int) -> int:\n    y = 0\n{BODY}\n    return y\n"


def test_blank_line_separated_symbols_stay_separate(store: Store) -> None:
    ingest(store, "m.py", TWO)
    cands = load_file_candidates(store, "m.py")
    out = compact_candidates(SourceCache(store.root), cands)
    assert [c.order_key for c in out] == [
        c.order_key for c in sorted(cands, key=lambda c: c.order_key)
    ]


def test_parts_of_a_split_symbol_merge_into_one_byte_range(small_store: Store) -> None:
    ingest(small_store, "big.py", BIG)
    cands = load_file_candidates(small_store, "big.py")
    assert len(cands) >= 2
    out = compact_candidates(SourceCache(small_store.root), cands)
    assert len(out) == 1
    merged = out[0]
    assert merged.anchor == ByteAnchor(merged.start_byte, merged.end_byte)
    assert merged.qualified_name == "long_fn"
    bounds = (min(c.start_byte for c in cands), max(c.end_byte for c in cands))
    assert (merged.start_byte, merged.end_byte) == bounds
    assert "merged" in merged.why


def test_overlap_merges_to_union(store: Store) -> None:
    ingest(store, "m.py", TWO)
    base = load_file_candidates(store, "m.py")[0]
    out = compact_candidates(SourceCache(store.root), (span(base, 0, 12), span(base, 8, 20, 0.9)))
    assert [(c.start_byte, c.end_byte) for c in out] == [(0, 20)]
    assert out[0].scores == Scores(final=0.9)


def test_identical_ranges_dedupe_to_best_ranked(store: Store) -> None:
    ingest(store, "m.py", TWO)
    base = load_file_candidates(store, "m.py")[0]
    low, high = span(base, 0, 8, 0.1), span(base, 0, 8, 0.7)
    out = compact_candidates(SourceCache(store.root), (low, high))
    assert out == (high,)


def test_missing_source_leaves_candidates_untouched(store: Store, tmp_path: Path) -> None:
    ingest(store, "m.py", TWO)
    base = load_file_candidates(store, "m.py")[0]
    (store.root / "m.py").unlink()
    cands = (span(base, 0, 8), span(base, 8, 12))
    out = compact_candidates(SourceCache(store.root), cands)
    assert sorted(out, key=lambda c: c.order_key) == list(cands)


def test_merged_item_is_exact_slice_with_resolvable_ref(small_store: Store) -> None:
    ingest(small_store, "big.py", BIG)
    cands = compact_candidates(
        SourceCache(small_store.root), load_file_candidates(small_store, "big.py")
    )
    result = AssembleStage().assemble(make_ctx(small_store), CandidateSet.of(cands))
    (item,) = result.pack["items"]
    (merged,) = cands
    body = BIG.encode()[merged.start_byte : merged.end_byte]
    assert item["text"] == body.decode()
    assert resolve_ref(small_store, item["ref"]) == body
    assert parse_ref(item["ref"]).anchor == ByteAnchor(merged.start_byte, merged.end_byte)


def test_merge_keeps_union_of_edges(store: Store) -> None:
    ingest(store, "m.py", TWO)
    base = load_file_candidates(store, "m.py")[0]
    first = CandidateEdge("calls", "cw://x", "exact")
    second = CandidateEdge("imports", "cw://y", "import")
    left = replace(span(base, 0, 12), edges=(first,))
    right = replace(span(base, 8, 20), edges=(first, second))
    (merged,) = compact_candidates(SourceCache(store.root), (left, right))
    assert merged.edges == (first, second)


def test_merged_group_keeps_member_symbols(store: Store) -> None:
    ingest(store, "m.py", TWO)
    base = load_file_candidates(store, "m.py")[0]
    first = replace(span(base, 0, 12), qualified_name="a", symbols=("a",))
    second = replace(span(base, 8, 20), qualified_name="b", symbols=("b", "a"))
    out = compact_candidates(SourceCache(store.root), (first, second))
    assert [(c.kind, c.qualified_name, c.symbols) for c in out] == [("group", None, ("a", "b"))]


def test_hydrate_carries_every_chunk_symbol(store: Store) -> None:
    ingest(store, "m.py", TWO)
    names = {name for cand in load_file_candidates(store, "m.py") for name in cand.symbols}
    assert names == {"a", "b"}
