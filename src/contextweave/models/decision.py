from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from contextweave.models.calibration import apply_temperature
from contextweave.models.errors import ModelDisabledError, ModelResponseError, ProviderSlotError

CHAT_ATTRIBUTES = ("annotate", "embed", "chat")
YES = "yes"
NO = "no"
NOUL_OPTIONS = (YES, NO)
CHOICE_OPTION_LIMIT = 26
SCORE_LEVELS = (2, 10)


class QuestionKind(StrEnum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True, slots=True)
class Question:
    """One typed question; descriptions run parallel to options and become the wire criteria."""

    name: str
    text: str
    kind: QuestionKind
    options: tuple[str, ...]
    descriptions: tuple[str, ...] = ()

    def description(self, index: int) -> str:
        """The option's description, or its label when none is declared."""
        return self.descriptions[index] if index < len(self.descriptions) else self.options[index]

    def level_values(self) -> tuple[float, ...]:
        """Score level values spread evenly over [0, 1], lowest level first."""
        last = len(self.options) - 1
        return tuple(index / last for index in range(len(self.options)))


@dataclass(frozen=True, slots=True)
class Decision:
    name: str
    probs: Mapping[str, float]
    top: str
    confidence: float


def kind_options(kind: QuestionKind, declared: Sequence[str]) -> tuple[str, ...]:
    """Fixed options of noul questions; choice options and score levels are as declared."""
    return NOUL_OPTIONS if kind is QuestionKind.NOUL else tuple(declared)


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


def score_distribution(levels: Sequence[str], value: float) -> dict[str, float]:
    """Two-point distribution over evenly spaced levels whose expected value is value."""
    clipped = min(max(value, 0.0), 1.0)
    last = len(levels) - 1
    low = min(int(clipped * last), last - 1)
    upper = clipped * last - low
    return {
        level: ({low: 1 - upper, low + 1: upper}).get(index, 0.0)
        for index, level in enumerate(levels)
    }


def noul_distribution(yes: float) -> dict[str, float]:
    """Yes/no distribution from a noul probability."""
    clipped = min(max(yes, 0.0), 1.0)
    return {YES: clipped, NO: 1 - clipped}


@dataclass(frozen=True, slots=True)
class Answers:
    """One state's decisions keyed by question, with calibrated scalar reads."""

    questions: Mapping[str, Question]
    decisions: Mapping[str, Decision]
    temperatures: Mapping[str, float]

    def value(self, name: str) -> float:
        """Noul yes-probability or score expected value, temperature-calibrated."""
        question, decision = self.questions[name], self.decisions[name]
        match question.kind:
            case QuestionKind.NOUL:
                raw = decision.probs[YES]
            case QuestionKind.SCORE:
                raw = sum(
                    decision.probs[level] * value
                    for level, value in zip(question.options, question.level_values(), strict=True)
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
