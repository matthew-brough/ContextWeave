from pack_support import make_cand
from retrieve_support import make_ctx

from contextweave.index.store import Store
from contextweave.retrieve.candidates import CandidateSet, Scores
from contextweave.retrieve.dedupe import DedupeStage, dedupe, overlaps


def test_overlap_is_same_file_and_intersecting() -> None:
    assert overlaps(make_cand(1, "a.py", 0, 10), make_cand(2, "a.py", 9, 20))
    assert not overlaps(make_cand(1, "a.py", 0, 10), make_cand(2, "a.py", 10, 20))
    assert not overlaps(make_cand(1, "a.py", 0, 10), make_cand(2, "b.py", 0, 10))


def test_overlap_keeps_highest_gate_representative() -> None:
    low = make_cand(1, "a.py", 0, 10, scores=Scores(gate=0.3))
    high = make_cand(2, "a.py", 5, 15, scores=Scores(gate=0.9))
    assert [c.chunk_id for c in dedupe(CandidateSet.of([low, high]))] == [2]


def test_symbol_identity_collapses_remapped_duplicates() -> None:
    first = make_cand(1, "a.py", 0, 10, name="f", scores=Scores(gate=0.4))
    second = make_cand(2, "a.py", 0, 10, name="f", scores=Scores(gate=0.5))
    assert [c.chunk_id for c in dedupe(CandidateSet.of([first, second]))] == [2]


def test_stage_computes_final_and_orders_by_it(store: Store) -> None:
    low = make_cand(1, "a.py", 0, 10, scores=Scores(gate=0.2, graph_proximity=1.0))
    high = make_cand(2, "b.py", 0, 10, scores=Scores(gate=0.8, graph_proximity=1.0))
    out = DedupeStage().run(make_ctx(store, "x"), CandidateSet.of([low, high]))
    assert out.chunk_ids() == (2, 1)
    assert out[0].scores.final == 0.9 * 0.8 + 0.1
    assert out[0].scores.redundancy == 0.0
