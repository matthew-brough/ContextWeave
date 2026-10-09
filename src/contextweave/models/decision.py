from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from contextweave.models.calibration import apply_temperature
from contextweave.models.errors import ModelDisabledError, ModelResponseError, ProviderSlotError

CHAT_ATTRIBUTES = ("annotate", "embed", "chat")
YES = "yes"
NO = "no"
BINARY_OPTIONS = (YES, NO)
SCORE_OPTIONS = ("0", "0.25", "0.5", "0.75", "1")
SCORE_VALUES = tuple(float(option) for option in SCORE_OPTIONS)


class QuestionKind(StrEnum):
    CHOICE = "choice"
    SCORE = "score"
    BINARY = "binary"


@dataclass(frozen=True, slots=True)
class Question:
    name: str
    text: str
    kind: QuestionKind
    options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decision:
    name: str
    probs: Mapping[str, float]
    top: str
    confidence: float


def kind_options(kind: QuestionKind, declared: Sequence[str]) -> tuple[str, ...]:
    """Fixed options of binary and score questions; choice options are as declared."""
    match kind:
        case QuestionKind.BINARY:
            return BINARY_OPTIONS
        case QuestionKind.SCORE:
            return SCORE_OPTIONS
        case QuestionKind.CHOICE:
            return tuple(declared)


def decision_of(question: Question, probs: Mapping[str, float]) -> Decision:
    """Restrict raw probabilities to the question's options and renormalise to sum 1."""
    kept = {option: max(probs.get(option, 0.0), 0.0) for option in question.options}
    total = sum(kept.values())
    if total <= 0:
        raise ModelResponseError(f"question '{question.name}': no probability mass on options")
    normalised = {option: value / total for option, value in kept.items()}
    top = max(
        question.options, key=lambda option: (normalised[option], -question.options.index(option))
    )
    return Decision(question.name, normalised, top, normalised[top])


def score_distribution(value: float) -> dict[str, float]:
    """Two-point distribution over the score grid whose expected value is value."""
    clipped = min(max(value, 0.0), 1.0)
    last = len(SCORE_VALUES) - 1
    low = min(int(clipped * last), last - 1)
    span = SCORE_VALUES[low + 1] - SCORE_VALUES[low]
    upper = (clipped - SCORE_VALUES[low]) / span
    return {
        option: ({low: 1 - upper, low + 1: upper}).get(index, 0.0)
        for index, option in enumerate(SCORE_OPTIONS)
    }


@dataclass(frozen=True, slots=True)
class Answers:
    """One state's decisions keyed by question, with calibrated scalar reads."""

    questions: Mapping[str, Question]
    decisions: Mapping[str, Decision]
    temperatures: Mapping[str, float]

    def value(self, name: str) -> float:
        """Binary yes-probability or score expected value, temperature-calibrated."""
        question, decision = self.questions[name], self.decisions[name]
        match question.kind:
            case QuestionKind.BINARY:
                raw = decision.probs[YES]
            case QuestionKind.SCORE:
                raw = sum(
                    decision.probs[o] * v for o, v in zip(SCORE_OPTIONS, SCORE_VALUES, strict=True)
                )
            case QuestionKind.CHOICE:
                raise ValueError(f"question '{name}' is a choice; use prob() or top()")
        return apply_temperature(raw, self.temperatures.get(name, 1.0))

    def prob(self, name: str, option: str) -> float:
        return self.decisions[name].probs.get(option, 0.0)

    def top(self, name: str) -> str:
        return self.decisions[name].top

    def confidence(self, name: str) -> float:
        return self.decisions[name].confidence


def answers_from(
    rows: Sequence[Sequence[Decision]],
    states: int,
    questions: Sequence[Question],
    temperatures: Mapping[str, float],
) -> list[Answers]:
    """Validate a decide() result against the request shape and wrap it per state."""
    if len(rows) != states:
        raise ModelResponseError(f"expected {states} decision rows, got {len(rows)}")
    by_name = {question.name: question for question in questions}
    wrapped: list[Answers] = []
    for row in rows:
        if [decision.name for decision in row] != [question.name for question in questions]:
            raise ModelResponseError("decision row does not match the question order")
        wrapped.append(
            Answers(by_name, {decision.name: decision for decision in row}, temperatures)
        )
    return wrapped


@runtime_checkable
class DecisionClient(Protocol):
    def decide(
        self, states: Sequence[str], questions: Sequence[Question]
    ) -> list[list[Decision]]: ...


class NullDecisionClient:
    """Classify provider none: every decision is disabled."""

    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        raise ModelDisabledError("classify provider is none")


def ensure_decision_client(client: object) -> DecisionClient:
    """Reject anything that is not a decision model, or that also speaks chat or embed."""
    if not isinstance(client, DecisionClient):
        raise ProviderSlotError("classify provider does not implement decide()")
    for name in CHAT_ATTRIBUTES:
        if callable(getattr(client, name, None)):
            raise ProviderSlotError(f"classify provider exposes chat-style '{name}()'")
    return client
