from collections.abc import Mapping, Sequence

from contextweave.core.types import JSON
from contextweave.models.errors import ModelDisabledError
from contextweave.models.providers import (
    AnnotateClient,
    BuildContext,
    Device,
    EmbedClient,
    EmbedKind,
    Message,
)

NONE_CHAIN = (Device.NONE,)


class NullEmbed:
    """Embed provider none."""

    @property
    def devices(self) -> tuple[Device, ...]:
        return NONE_CHAIN

    def set_device(self, device: Device) -> None:
        return None

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        raise ModelDisabledError("embed provider is none")


class NullAnnotate:
    """Annotate provider none."""

    @property
    def devices(self) -> tuple[Device, ...]:
        return NONE_CHAIN

    def set_device(self, device: Device) -> None:
        return None

    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]:
        raise ModelDisabledError("annotate provider is none")


def build_null_embed(ctx: BuildContext) -> EmbedClient:
    """Factory for the embed none provider."""
    return NullEmbed()


def build_null_annotate(ctx: BuildContext) -> AnnotateClient:
    """Factory for the annotate none provider."""
    return NullAnnotate()
