from collections.abc import Mapping, Sequence

from contextweave.core.errors import DegradedEntry
from contextweave.models.decision import Answers
from contextweave.models.questions import QuestionSetError, load_question_set
from contextweave.models.slots import Done
from contextweave.retrieve.context import QueryContext


def ask_stage(
    ctx: QueryContext,
    question_set: str,
    states: Sequence[str],
    *,
    stage: str,
    dynamic: Mapping[str, Sequence[str]] | None = None,
) -> list[Answers] | None:
    """Answers per state, or None when the classify slot is absent or skipped (skip is recorded)."""
    models = ctx.models
    if models is None or not states:
        return None
    version = ctx.config.models.classify.question_set_version
    try:
        questions = load_question_set(question_set, version).build(dynamic)
    except QuestionSetError as exc:
        ctx.envelope.add_degraded(DegradedEntry(stage=stage, reason=str(exc)))
        return None
    outcome = models.ask(states, questions, stage=stage)
    if isinstance(outcome, Done):
        return outcome.value
    ctx.envelope.add_degraded(outcome.degraded)
    return None
