from contextweave.models.decision import DecisionClient, QuestionKind
from contextweave.models.decision_http import DecisionWire, HttpDecisionClient
from contextweave.models.providers import BuildContext

CLEF_WIRE = DecisionWire(
    path="/classify",
    kind_names={
        QuestionKind.CHOICE: "choice",
        QuestionKind.BINARY: "binary",
        QuestionKind.SCORE: "score",
    },
    native_score=True,
)


def build_clef(ctx: BuildContext) -> DecisionClient:
    """Factory for the clef classify provider."""
    return HttpDecisionClient(CLEF_WIRE, ctx)
