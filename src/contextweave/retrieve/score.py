import re
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Literal

from contextweave.core.config import Config, ScoreConfig
from contextweave.retrieve.candidates import Candidate, CandidateSet, Scores
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.structural import structural_terms

type Role = Literal["test", "config", "implementation"]

STAGE_NAME = "gate"
_NAME_SEPARATORS = re.compile(r"[.:#]+")


def rank_norm(rank: int | None, keep: int) -> float:
    """1 - (rank-1)/keep for a present rank, else 0."""
    if rank is None or keep <= 0:
        return 0.0
    return max(0.0, 1 - (rank - 1) / keep)


def role_of(cand: Candidate, test_globs: tuple[str, ...]) -> Role:
    """Path rules: test glob is test, schema profile is config, else implementation."""
    path = PurePosixPath(cand.path)
    if any(path.full_match(glob) for glob in test_globs):
        return "test"
    return "config" if cand.profile == "schema" else "implementation"


def exact_name_hit(cand: Candidate, terms: tuple[str, ...]) -> bool:
    """True when a query term equals the chunk's qualified or simple symbol name."""
    name = cand.qualified_name
    if name is None:
        return False
    simple = _NAME_SEPARATORS.split(name)[-1]
    return name in terms or simple in terms


def final_score(scores: Scores, weights: ScoreConfig) -> float:
    """final = w_gate*gate + w_graph*graph_proximity + w_redundancy*redundancy."""
    return (
        weights.final_gate * scores.gate
        + weights.final_graph * scores.graph_proximity
        + weights.final_redundancy * scores.redundancy
    )


def with_final(cand: Candidate, weights: ScoreConfig) -> Candidate:
    scores = replace(cand.scores, final=final_score(cand.scores, weights))
    return cand.with_scores(scores)


def mark_exact(cand: Candidate, why: str) -> Candidate:
    """Pin a candidate as an exact hit: gate 1.0, critical path."""
    scores = replace(cand.scores, gate=1.0, graph_proximity=1.0)
    return replace(cand, scores=scores, critical=True, why=why, fused=max(cand.fused, 1.0))


def _why(cand: Candidate) -> str:
    ranks = cand.ranks
    parts = [
        f"{name} #{rank}"
        for name, rank in (
            ("lexical", ranks.lexical),
            ("semantic", ranks.semantic),
            ("structural", ranks.structural),
        )
        if rank is not None
    ]
    return ", ".join(parts)


def fallback_scores(cand: Candidate, config: Config, terms: tuple[str, ...]) -> Candidate:
    """Classify-skipped scoring: gate = fused_norm, exact name hit pins gate 1.0."""
    keep = config.fuse.keep
    role = role_of(cand, config.index.test_globs)
    scores = replace(
        cand.scores,
        gate=cand.fused,
        lexical_match=rank_norm(cand.ranks.lexical, keep),
        structural_value=rank_norm(cand.ranks.structural, keep),
        implementation=1.0 if role == "implementation" else 0.0,
        graph_proximity=1.0,
    )
    scored = replace(cand, scores=scores, why=_why(cand))
    return mark_exact(scored, "exact name match") if exact_name_hit(cand, terms) else scored


class FallbackScoreStage:
    """Gate slot without a model: fallback scores, then the top gate.window by gate."""

    name = STAGE_NAME

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        terms = structural_terms(ctx.query)
        scored = [fallback_scores(cand, ctx.config, terms) for cand in cands]
        scored.sort(key=lambda cand: (-cand.scores.gate, *cand.order_key))
        window = ctx.config.gate.window
        return CandidateSet.of(scored[:window])
