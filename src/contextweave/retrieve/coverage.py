from collections.abc import Sequence

from contextweave.compact.stage import CompactStage
from contextweave.retrieve.ask import ask_stage
from contextweave.retrieve.candidates import Candidate, CandidateSet
from contextweave.retrieve.context import CoverageResult, QueryContext
from contextweave.retrieve.dedupe import DedupeStage
from contextweave.retrieve.expand import ExpandStage, GatePassTwo, final_of
from contextweave.retrieve.gate import model_gate_pass

STAGE_NAME = "coverage"
QUESTION_SET = "coverage"
NO_MISSING = "none"
SUFFICIENT_MIN_PROBABILITY = 0.5
OUTLINE_LIMIT = 60
HOP = 1


def outline_line(cand: Candidate) -> str:
    rels = ", ".join(sorted({edge.rel for edge in cand.edges})) or "-"
    name = cand.qualified_name or "-"
    return f"{cand.path} :: {name} [{cand.kind or 'chunk'}] rep=source rels={rels}"


def coverage_state(query: str, cands: Sequence[Candidate]) -> str:
    """The query and a pack outline of names, relations and representations."""
    lines = [outline_line(cand) for cand in cands[:OUTLINE_LIMIT]]
    return "query: " + query + "\npack outline:\n" + "\n".join(lines)


class CoverageStage:
    """Asks whether the pack suffices; grows it one hop over the missing relation, at most max_iterations."""

    name = STAGE_NAME

    def __init__(self, gate_pass: GatePassTwo = model_gate_pass) -> None:
        self._expand = ExpandStage(gate_pass)

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        if ctx.models is None:
            return cands
        current = cands
        passes = 0
        while True:
            answers = ask_stage(
                ctx, QUESTION_SET, [coverage_state(ctx.query, current.items)], stage=STAGE_NAME
            )
            if answers is None:
                return current
            row = answers[0]
            score = row.value("sufficient")
            missing = row.top("missing_rel")
            sufficient = score >= SUFFICIENT_MIN_PROBABILITY
            wanted = () if missing == NO_MISSING else (missing,)
            ctx.coverage.value = CoverageResult(score, sufficient, passes, wanted)
            if sufficient or not wanted or passes >= ctx.config.coverage.max_iterations:
                return current
            grown = self._grow(ctx, current, wanted)
            if len(grown) == len(current):
                return current
            current = grown
            passes += 1

    def _grow(self, ctx: QueryContext, cands: CandidateSet, rels: tuple[str, ...]) -> CandidateSet:
        reached = self._expand.expand(ctx, cands, rels, HOP, final_of)
        if len(reached) == len(cands):
            return cands
        pipeline = ctx.config.pipeline
        grown = DedupeStage().run(ctx, reached) if pipeline.dedupe else reached
        return CompactStage().run(ctx, grown) if pipeline.compact else grown
