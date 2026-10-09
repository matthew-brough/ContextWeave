from contextweave.core.refs import Anchor
from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.score import with_final

STAGE_NAME = "dedupe"


def overlaps(first: Candidate, second: Candidate) -> bool:
    return (
        first.path == second.path
        and first.start_byte < second.end_byte
        and second.start_byte < first.end_byte
    )


def dedupe(cands: CandidateSet) -> tuple[Candidate, ...]:
    """Keep the highest-gate representative of each symbol identity and byte-range overlap."""
    ordered = sorted(cands, key=lambda cand: (-cand.scores.gate, *cand.order_key))
    kept: list[Candidate] = []
    identities: set[tuple[str, Anchor]] = set()
    for cand in ordered:
        identity = (cand.path, cand.anchor)
        if identity in identities or any(overlaps(cand, other) for other in kept):
            continue
        identities.add(identity)
        kept.append(cand)
    return tuple(kept)


class DedupeStage:
    """Collapses duplicates, then computes final scores and orders by them."""

    name = STAGE_NAME

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        weights = ctx.config.score
        scored = [with_final(cand, weights) for cand in dedupe(cands)]
        scored.sort(key=lambda cand: (-cand.scores.final, *cand.order_key))
        return CandidateSet.of(scored)
