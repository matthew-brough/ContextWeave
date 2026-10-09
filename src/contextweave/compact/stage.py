from contextweave.compact.rules import compact_candidates
from contextweave.compact.sources import SourceCache
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.context import QueryContext

STAGE_NAME = "compact"


class CompactStage:
    """Applies the lossless candidate rules; item rules run when the pack is assembled."""

    name = STAGE_NAME

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        sources = SourceCache(ctx.root)
        return CandidateSet.of(compact_candidates(sources, tuple(cands)))
