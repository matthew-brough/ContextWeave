import math
import re
from collections.abc import Sequence

from contextweave.core.errors import DegradedEntry
from contextweave.memory.embedder import TextEmbedder
from contextweave.memory.store import SummaryInput
from contextweave.models.providers import EmbedKind
from contextweave.models.slots import Done

DEFAULT_TOP_K = 3
TRUNCATE_CHARS = 200
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split on sentence punctuation and newlines."""
    pieces: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            pieces.extend(part.strip() for part in _SENTENCE_SPLIT.split(stripped) if part.strip())
    return pieces


def truncation_summary(text: str, max_chars: int = TRUNCATE_CHARS) -> str:
    """Cut text to max_chars with an ellipsis."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[: max_chars - 1].rstrip() + "…"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm > 0 else 0.0


def _central(sentences: list[str], vectors: list[list[float]], top_k: int) -> str:
    dim = len(vectors[0])
    centroid = [sum(vector[i] for vector in vectors) / len(vectors) for i in range(dim)]
    scored = sorted(
        ((_cosine(vector, centroid), index) for index, vector in enumerate(vectors)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    chosen = sorted(index for _, index in scored[:top_k])
    return " ".join(sentences[i] for i in chosen)


def extractive_summary(
    text: str, embedder: TextEmbedder | None, top_k: int = DEFAULT_TOP_K
) -> tuple[SummaryInput, DegradedEntry | None]:
    """Pick the top_k sentences nearest the centroid, in source order; truncate when embedding is unavailable."""
    sentences = split_sentences(text)
    if len(sentences) <= top_k:
        return SummaryInput(" ".join(sentences) if sentences else text.strip(), "extractive"), None
    if embedder is None:
        degraded = DegradedEntry(stage="memory_summary", reason="no embedder")
        return SummaryInput(truncation_summary(text), "truncated"), degraded
    outcome = embedder.embed(sentences, EmbedKind.MEMORY)
    if not isinstance(outcome, Done):
        return SummaryInput(truncation_summary(text), "truncated"), outcome.degraded
    vectors = outcome.value
    if len(vectors) != len(sentences) or not vectors[0]:
        degraded = DegradedEntry(stage="memory_summary", reason="bad embed response")
        return SummaryInput(truncation_summary(text), "truncated"), degraded
    return SummaryInput(_central(sentences, vectors, top_k), "extractive"), None
