import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from fake_embed import FakeEmbed

from contextweave.core.config import AnnotateConfig, ClassifyConfig, Config, ModelsConfig
from contextweave.core.types import JSON
from contextweave.models.errors import ModelContextError, ModelTransportError
from contextweave.models.providers import Device, Message
from contextweave.models.slots import ModelSlots, default_registries

NAME_LINE = re.compile(r"^Name: (.*)$", re.MULTILINE)


@dataclass(slots=True)
class FakeAnnotate:
    """Annotate provider deriving purpose and tags from the chunk name; records every prompt."""

    prompts: list[str] = field(default_factory=list[str])
    down: bool = False
    garbage: Mapping[str, JSON] | None = None
    garbage_for: str | None = None
    context_chars: int | None = None

    @property
    def devices(self) -> tuple[Device, ...]:
        return (Device.CPU,)

    def set_device(self, device: Device) -> None:
        return None

    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]:
        if self.down:
            raise ModelTransportError("fake annotate is down")
        user = prompt_messages[-1].content
        self.prompts.append(user)
        if self.context_chars is not None and len(user) > self.context_chars:
            raise ModelContextError(
                "fake context overflow", needed=len(user), available=self.context_chars
            )
        if self.garbage is not None and (self.garbage_for is None or self.garbage_for in user):
            return self.garbage
        match = NAME_LINE.search(user)
        name = match.group(1) if match else "unknown"
        words = name.replace("_", " ")
        return {"purpose": f"Does {words} work.", "tags": [name.lower().replace(".", "-"), "fake"]}

    @property
    def names(self) -> list[str]:
        return [match.group(1) for text in self.prompts if (match := NAME_LINE.search(text))]


def annotate_config(model: str = "fake-annotate", version: int = 1) -> AnnotateConfig:
    return AnnotateConfig(provider="fake", model=model, prompt_version=version)


def config_with(embed: FakeEmbed, annotate: AnnotateConfig) -> Config:
    models = ModelsConfig(
        embed=embed.config, annotate=annotate, classify=ClassifyConfig(provider="none")
    )
    return replace(Config(), models=models)


def slots_with(embed: FakeEmbed, annotate: FakeAnnotate, config: Config) -> ModelSlots:
    registries = default_registries()
    registries.embed.register("fake", lambda ctx: embed)
    registries.annotate.register("fake", lambda ctx: annotate)
    return ModelSlots.from_config(
        config.models, registries=registries, clock=lambda: 0.0, submit=lambda job: job()
    )
