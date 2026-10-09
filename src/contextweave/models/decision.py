from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from contextweave.models.errors import ModelDisabledError, ProviderSlotError

CHAT_ATTRIBUTES = ("annotate", "embed", "chat")


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
