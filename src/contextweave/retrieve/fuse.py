from dataclasses import replace

from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.context import QueryContext, RrfWeights

RRF_K = 60
STAGE_NAME = "fuse"


def rrf_score(cand: Candidate, weights: RrfWeights) -> float:
    """Weighted reciprocal rank fusion over the strategies that returned the chunk."""
    ranks = cand.ranks
    total = 0.0
    for rank, weight in (
        (ranks.lexical, weights.lexical),
        (ranks.semantic, weights.semantic),
        (ranks.structural, weights.structural),
    ):
        if rank is not None:
            total += weight / (RRF_K + rank)
    return total


def fuse(cands: CandidateSet, weights: RrfWeights, keep: int) -> CandidateSet:
    """Top `keep` by RRF, ties by (path, start_byte); fused = 1 - (rank-1)/keep."""
    if keep <= 0:
        return CandidateSet()
    ordered = sorted(cands, key=lambda cand: (-rrf_score(cand, weights), *cand.order_key))
    return CandidateSet.of(
        replace(cand, fused=1 - (position - 1) / keep)
        for position, cand in enumerate(ordered[:keep], start=1)
    )


class FuseStage:
    """RRF fusion of the retrieval strategies' ranks."""

    name = STAGE_NAME

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        plan = ctx.plan.value
        weights = plan.rrf_weights if plan else RrfWeights()
        return fuse(cands, weights, ctx.config.fuse.keep)
