from collections.abc import Sequence
from dataclasses import replace

from contextweave.compact.sources import SourceCache
from contextweave.core.config import ScoreConfig
from contextweave.models.decision import Answers
from contextweave.retrieve.ask import ask_stage
from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.card import cached_purposes, card_of
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.expand import Expanded, fallback_gate_pass
from contextweave.retrieve.score import FallbackScoreStage, exact_name_hit, mark_exact
from contextweave.retrieve.structural import structural_terms

STAGE_NAME = "gate"
QUESTION_SET = "gate"
EXPAND_STAGE = "expand"


def gate_state(query: str, card: str) -> str:
    return f"query: {query}\n{card}"


def gate_states(ctx: QueryContext, cands: Sequence[Candidate]) -> list[str]:
    """One decision state per candidate: the query and the candidate card."""
    annotate = ctx.config.models.annotate
    purposes = cached_purposes(
        ctx.store, annotate.model, annotate.prompt_version, [cand.chunk_id for cand in cands]
    )
    sources = SourceCache(ctx.root)
    tokens = ctx.config.gate.card_tokens
    return [gate_state(ctx.query, card_of(cand, sources, purposes, tokens)) for cand in cands]


def gate_value(answers: Answers, lexical_match: float, weights: ScoreConfig) -> float:
    """The weighted gate formula over classifier terms and the lexical rank term, normalised."""
    raw = (
        weights.gate_answers_query * answers.value("answers_query")
        + weights.gate_relevance * answers.value("relevance")
        + weights.gate_role_implementation * answers.prob("role", "implementation")
        + weights.gate_lexical_match * lexical_match
        + weights.gate_structural_value * answers.value("structural_value")
    )
    return min(max(raw / weights.gate_norm, 0.0), 1.0)


def with_answers(cand: Candidate, answers: Answers, weights: ScoreConfig) -> Candidate:
    """Candidate scored from the classifier's answers; keeps its graph and lexical terms."""
    scores = replace(
        cand.scores,
        gate=gate_value(answers, cand.scores.lexical_match, weights),
        answers_query=answers.value("answers_query"),
        relevance=answers.value("relevance"),
        implementation=answers.prob("role", "implementation"),
        structural_value=answers.value("structural_value"),
    )
    return replace(cand, scores=scores)


def model_scored(
    ctx: QueryContext, cand: Candidate, answers: Answers, terms: tuple[str, ...]
) -> Candidate:
    scored = with_answers(cand, answers, ctx.config.score)
    if exact_name_hit(cand, terms):
        return mark_exact(scored, "exact name match")
    critical = scored.scores.answers_query >= ctx.config.gate.critical
    detail = f"answers_query {scored.scores.answers_query:.2f}"
    return replace(scored, critical=cand.critical or critical, why=f"{cand.why}; {detail}")


class GateStage:
    """Classifier gate over the top gate.window candidates; fallback scoring when classify is skipped."""

    name = STAGE_NAME

    def __init__(self) -> None:
        self._fallback = FallbackScoreStage()

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        windowed = self._fallback.run(ctx, cands)
        answers = ask_stage(ctx, QUESTION_SET, gate_states(ctx, windowed.items), stage=STAGE_NAME)
        if answers is None:
            return windowed
        terms = structural_terms(ctx.query)
        scored = [
            model_scored(ctx, cand, row, terms) for cand, row in zip(windowed, answers, strict=True)
        ]
        scored.sort(key=lambda cand: (-cand.scores.gate, *cand.order_key))
        return CandidateSet.of(scored)


def model_gate_pass(ctx: QueryContext, expanded: Sequence[Expanded]) -> list[Candidate]:
    """Gate pass 2: classifier-scored reached chunks, hop-discounted fallback when skipped."""
    fallback = fallback_gate_pass(ctx, expanded)
    answers = ask_stage(ctx, QUESTION_SET, gate_states(ctx, fallback), stage=EXPAND_STAGE)
    if answers is None:
        return fallback
    return [
        with_answers(cand, row, ctx.config.score)
        for cand, row in zip(fallback, answers, strict=True)
    ]
