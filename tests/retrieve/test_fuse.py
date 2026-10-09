from pack_support import make_cand

from contextweave.retrieve.candidates import CandidateSet, Ranks
from contextweave.retrieve.context import RrfWeights
from contextweave.retrieve.fuse import RRF_K, fuse, rrf_score


def test_rrf_sums_weighted_reciprocal_ranks() -> None:
    cand = make_cand(1, ranks=Ranks(lexical=1, structural=3))
    weights = RrfWeights(lexical=2.0, structural=0.5)
    assert rrf_score(cand, weights) == 2.0 / (RRF_K + 1) + 0.5 / (RRF_K + 3)


def test_chunk_in_two_lists_outranks_single_list_leader() -> None:
    both = make_cand(1, "b.py", ranks=Ranks(lexical=2, structural=2))
    single = make_cand(2, "a.py", ranks=Ranks(lexical=1))
    fused = fuse(CandidateSet.of([single, both]), RrfWeights(), keep=10)
    assert fused.chunk_ids() == (1, 2)


def test_weights_change_order() -> None:
    lexical = make_cand(1, "a.py", ranks=Ranks(lexical=1))
    structural = make_cand(2, "b.py", ranks=Ranks(structural=1))
    cands = CandidateSet.of([lexical, structural])
    assert fuse(cands, RrfWeights(structural=3.0), 10).chunk_ids() == (2, 1)
    assert fuse(cands, RrfWeights(lexical=3.0), 10).chunk_ids() == (1, 2)


def test_ties_break_by_path_and_start_byte() -> None:
    first = make_cand(1, "a.py", 5, 9, ranks=Ranks(lexical=1))
    second = make_cand(2, "a.py", 0, 4, ranks=Ranks(structural=1))
    third = make_cand(3, "0.py", 0, 4, ranks=Ranks(semantic=1))
    fused = fuse(CandidateSet.of([first, second, third]), RrfWeights(), 10)
    assert fused.chunk_ids() == (3, 2, 1)


def test_keep_truncates_and_fused_norm_is_linear_in_rank() -> None:
    cands = CandidateSet.of(make_cand(i, f"{i}.py", ranks=Ranks(lexical=i)) for i in range(1, 9))
    fused = fuse(cands, RrfWeights(), keep=4)
    assert fused.chunk_ids() == (1, 2, 3, 4)
    assert [c.fused for c in fused] == [1.0, 0.75, 0.5, 0.25]


def test_zero_keep_is_empty() -> None:
    assert len(fuse(CandidateSet.of([make_cand(1, ranks=Ranks(lexical=1))]), RrfWeights(), 0)) == 0
