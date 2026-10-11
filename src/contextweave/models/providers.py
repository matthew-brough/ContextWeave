import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

import httpx

from contextweave.core.config import DeviceEndpointConfig, EmbedConfig, ModelsConfig
from contextweave.core.registry import Registry
from contextweave.core.types import JSON
from contextweave.models.decision import DecisionClient
from contextweave.models.errors import ModelParseError, ModelResponseError
from contextweave.models.wire import to_json

DEFAULT_SEED = 7


class Slot(StrEnum):
    EMBED = "embed"
    ANNOTATE = "annotate"
    CLASSIFY = "classify"


class Device(StrEnum):
    GPU = "gpu"
    CPU = "cpu"
    REMOTE = "remote"
    NONE = "none"


class EmbedKind(StrEnum):
    QUERY = "query"
    DOC = "doc"
    MEMORY = "memory"


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Endpoints:
    openai_base_url: str = "http://127.0.0.1:8000"
    openai_api_key: str = ""


@dataclass(frozen=True, slots=True)
class BuildContext:
    models: ModelsConfig
    endpoints: Endpoints = field(default_factory=Endpoints)
    transport: httpx.BaseTransport | None = None

    @property
    def deadline_s(self) -> float:
        return self.models.deadline_ms / 1000

    @property
    def annotate_deadline_s(self) -> float:
        return self.models.annotate.deadline_ms / 1000

    @property
    def embed_batch_deadline_s(self) -> float:
        return self.models.embed.batch_deadline_ms / 1000

    def embed_deadline_s(self, kind: EmbedKind) -> float:
        """Hard cap of one embed call: document batches are index-time, the rest query-time."""
        return self.embed_batch_deadline_s if kind is EmbedKind.DOC else self.deadline_s


@runtime_checkable
class Provider(Protocol):
    @property
    def devices(self) -> tuple[Device, ...]: ...

    def set_device(self, device: Device) -> None: ...


@runtime_checkable
class Supervised(Protocol):
    """A provider backed by local servers, one endpoint per device, that residency may launch."""

    @property
    def servers(self) -> Mapping[Device, DeviceEndpointConfig]: ...


@runtime_checkable
class EmbedClient(Provider, Protocol):
    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]: ...


@runtime_checkable
class AnnotateClient(Provider, Protocol):
    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]: ...


type EmbedFactory = Callable[[BuildContext], EmbedClient]
type AnnotateFactory = Callable[[BuildContext], AnnotateClient]
type DecisionFactory = Callable[[BuildContext], DecisionClient]


class ProviderRegistries:
    """Per-slot provider registries; a provider can only be registered for the slot it speaks."""

    def __init__(self) -> None:
        self.embed: Registry[str, EmbedFactory] = Registry()
        self.annotate: Registry[str, AnnotateFactory] = Registry()
        self.classify: Registry[str, DecisionFactory] = Registry()


def embed_inputs(config: EmbedConfig, texts: Sequence[str], kind: EmbedKind) -> list[str]:
    """Apply the configured prefix for the embedding kind."""
    prefix = {
        EmbedKind.QUERY: config.query_prefix,
        EmbedKind.DOC: config.doc_prefix,
        EmbedKind.MEMORY: config.memory_prefix,
    }[kind]
    return [f"{prefix}{text}" for text in texts]


def shape_vectors(
    config: EmbedConfig, vectors: Sequence[Sequence[float]], expected: int
) -> list[list[float]]:
    """Truncate to the configured dim and renormalise when truncation happened."""
    if len(vectors) != expected:
        raise ModelResponseError(f"expected {expected} embeddings, got {len(vectors)}")
    shaped: list[list[float]] = []
    for vector in vectors:
        if len(vector) < config.dim:
            raise ModelResponseError(f"embedding dim {len(vector)} below configured {config.dim}")
        head = list(vector[: config.dim])
        if len(vector) > config.dim:
            norm = math.sqrt(sum(value * value for value in head))
            if norm > 0:
                head = [value / norm for value in head]
        shaped.append(head)
    return shaped


def message_payload(messages: Sequence[Message]) -> list[JSON]:
    """Serialise chat messages for the wire."""
    return [{"role": message.role, "content": message.content} for message in messages]


def parse_structured(content: str, schema: Mapping[str, JSON]) -> dict[str, JSON]:
    """Decode generated text as a JSON object containing the schema's required keys."""
    try:
        decoded: object = json.loads(content)
        value = to_json(decoded)
    except (ValueError, ModelResponseError) as exc:
        raise ModelParseError(f"structured output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelParseError("structured output is not a JSON object")
    required = schema.get("required", [])
    if isinstance(required, list):
        missing = [key for key in required if isinstance(key, str) and key not in value]
        if missing:
            raise ModelParseError(f"structured output missing keys: {', '.join(missing)}")
    return value
