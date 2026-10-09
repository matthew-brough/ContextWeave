from contextweave.models.decision import DecisionClient, QuestionKind
from contextweave.models.decision_http import DecisionWire, HttpDecisionClient
from contextweave.models.providers import BuildContext

KEV_WIRE = DecisionWire(
    path="/v1/decide",
    kind_names={
        QuestionKind.CHOICE: "choice",
        QuestionKind.BINARY: "choice",
        QuestionKind.SCORE: "choice",
    },
    native_score=False,
)


def build_kev(ctx: BuildContext) -> DecisionClient:
    """Factory for the kev classify provider."""
    return HttpDecisionClient(KEV_WIRE, ctx)
