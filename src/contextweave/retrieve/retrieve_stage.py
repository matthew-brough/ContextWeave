from collections.abc import Callable
from dataclasses import replace

from contextweave.core.errors import DegradedEntry
from contextweave.retrieve.candidates import Candidate, CandidateSet, Ranks
from contextweave.retrieve.context import Mode, QueryContext
from contextweave.retrieve.freshness import freshen
from contextweave.retrieve.lexical import lexical_search
from contextweave.retrieve.structural import structural_search
from contextweave.retrieve.vector import vector_search

type Strategy = Callable[[QueryContext, str | None, int], tuple[Candidate, ...]]

STAGE_NAME = "retrieve"


def _lexical(ctx: QueryContext, scope: str | None, limit: int) -> tuple[Candidate, ...]:
    return lexical_search(ctx.store, ctx.query, scope=scope, limit=limit)


def _structural(ctx: QueryContext, scope: str | None, limit: int) -> tuple[Candidate, ...]:
    return structural_search(ctx.store, ctx.query, scope=scope, limit=limit)


STRATEGIES: dict[str, Strategy] = {
    "lexical": _lexical,
    "structural": _structural,
    "semantic": vector_search,
}

MODE_STRATEGIES: dict[Mode, tuple[str, ...]] = {
    "auto": ("lexical", "semantic", "structural"),
    "lexical": ("lexical",),
    "semantic": ("semantic",),
    "structural": ("structural",),
}


def merge_candidates(groups: list[tuple[Candidate, ...]]) -> CandidateSet:
    """Union by chunk id, ranks combined, ordered by best rank then (path, start_byte)."""
    merged: dict[int, Candidate] = {}
    for group in groups:
        for cand in group:
            known = merged.get(cand.chunk_id)
            if known is None:
                merged[cand.chunk_id] = cand
                continue
            ranks = Ranks(
                lexical=known.ranks.lexical or cand.ranks.lexical,
                semantic=known.ranks.semantic or cand.ranks.semantic,
                structural=known.ranks.structural or cand.ranks.structural,
            )
            merged[cand.chunk_id] = replace(known, ranks=ranks)
    ordered = sorted(merged.values(), key=lambda c: (c.ranks.best() or 0, *c.order_key))
    return CandidateSet.of(ordered)


class RetrieveStage:
    """Runs the strategies the mode allows and returns stat-checked candidates."""

    name = STAGE_NAME

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        plan = ctx.plan.value
        scope = ctx.scope or (plan.scope if plan else None)
        limit = ctx.config.fuse.keep
        groups: list[tuple[Candidate, ...]] = []
        for name in MODE_STRATEGIES[ctx.mode]:
            strategy = STRATEGIES.get(name)
            if strategy is None:
                ctx.envelope.add_degraded(
                    DegradedEntry(stage=STAGE_NAME, reason=f"{name}_unavailable")
                )
                continue
            with ctx.envelope.guard(STAGE_NAME):
                groups.append(strategy(ctx, scope, limit))
        return freshen(ctx.store, merge_candidates(groups), ctx.envelope, STAGE_NAME)
