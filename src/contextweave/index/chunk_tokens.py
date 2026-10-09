from dataclasses import dataclass
from typing import Protocol

from contextweave.core.budget import DEFAULT_ESTIMATOR, Estimator, estimators
from contextweave.core.registry import Registry


class ChunkTokenizer(Protocol):
    """Counts the tokens chunk sizing sees; the embed model's tokenizer plugs in here."""

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EstimatorTokenizer:
    """A ChunkTokenizer backed by a budget estimator."""

    estimator: Estimator

    def count(self, text: str) -> int:
        return self.estimator(text)


DEFAULT_TOKENIZER = "bytes35"

tokenizers: Registry[str, ChunkTokenizer] = Registry()
tokenizers.register(DEFAULT_TOKENIZER, EstimatorTokenizer(estimators.get(DEFAULT_ESTIMATOR)))


def default_tokenizer() -> ChunkTokenizer:
    """The registered tokenizer chunk sizing uses unless given another."""
    return tokenizers.get(DEFAULT_TOKENIZER)
