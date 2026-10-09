from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from contextweave.models.providers import EmbedKind
from contextweave.models.slots import ModelSlots, Outcome

EMBED_STAGE = "memory_embed"


class TextEmbedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> Outcome[list[list[float]]]: ...


@dataclass(frozen=True, slots=True)
class SlotEmbedder:
    """TextEmbedder backed by the embed model slot."""

    slots: ModelSlots
    model_id: str

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> Outcome[list[list[float]]]:
        """Embed texts through the embed slot, reporting skips as degraded."""
        return self.slots.embed(texts, kind=kind, stage=EMBED_STAGE)
