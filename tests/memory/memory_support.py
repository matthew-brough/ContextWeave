import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from contextweave.core.errors import DegradedEntry
from contextweave.models.providers import EmbedKind
from contextweave.models.slots import Done, Outcome, Skipped

CONCEPTS: dict[str, int] = {
    "car": 0,
    "automobile": 0,
    "vehicle": 0,
    "database": 1,
    "sqlite": 1,
    "storage": 1,
    "network": 2,
    "socket": 2,
    "latency": 3,
    "speed": 3,
    "cache": 4,
    "memoization": 4,
}
DIM = 6


@dataclass(slots=True)
class FakeEmbedder:
    model_id: str = "fake-embed"
    skip: bool = False
    calls: list[tuple[EmbedKind, tuple[str, ...]]] = field(
        default_factory=list[tuple[EmbedKind, tuple[str, ...]]]
    )

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> Outcome[list[list[float]]]:
        self.calls.append((kind, tuple(texts)))
        if self.skip:
            return Skipped(DegradedEntry(stage="memory_embed", reason="embed down"))
        return Done([self._vector(text) for text in texts])

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIM
        vector[DIM - 1] = 0.01
        for word in re.findall(r"\w+", text.lower()):
            axis = CONCEPTS.get(word)
            if axis is not None:
                vector[axis] += 1.0
        return vector
