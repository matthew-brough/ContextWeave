import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from contextweave.core.config import (
    AnnotateConfig,
    ClassifyConfig,
    Config,
    EmbedConfig,
    ModelsConfig,
)
from contextweave.models.errors import ModelTransportError
from contextweave.models.providers import Device, EmbedKind, embed_inputs
from contextweave.models.slots import ModelSlots, default_registries

WORD = re.compile(r"[a-z0-9]+")


def hashed_vector(text: str, dim: int) -> list[float]:
    """Deterministic unit-length bag-of-words vector."""
    values = [0.0] * dim
    for word in WORD.findall(text.lower()):
        values[int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % dim] += 1.0
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


@dataclass(slots=True)
class FakeEmbed:
    """Embed provider hashing words; records every call and can be switched off."""

    config: EmbedConfig
    calls: list[list[str]] = field(default_factory=list[list[str]])
    down: bool = False

    @property
    def devices(self) -> tuple[Device, ...]:
        return (Device.CPU,)

    def set_device(self, device: Device) -> None:
        return None

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        if self.down:
            raise ModelTransportError("fake embed is down")
        inputs = embed_inputs(self.config, texts, kind)
        self.calls.append(inputs)
        return [hashed_vector(text, self.config.dim) for text in inputs]

    @property
    def texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def embed_config(model: str = "fake-a", dim: int = 128) -> EmbedConfig:
    return EmbedConfig(provider="fake", model=model, dim=dim, query_prefix="", doc_prefix="")


def config_for(embed: EmbedConfig) -> Config:
    models = ModelsConfig(
        embed=embed,
        annotate=AnnotateConfig(provider="none"),
        classify=ClassifyConfig(provider="none"),
    )
    return replace(Config(), models=models)


def slots_for(fake: FakeEmbed, config: Config) -> ModelSlots:
    registries = default_registries()
    registries.embed.register("fake", lambda ctx: fake)
    return ModelSlots.from_config(
        config.models, registries=registries, clock=lambda: 0.0, submit=lambda job: job()
    )
