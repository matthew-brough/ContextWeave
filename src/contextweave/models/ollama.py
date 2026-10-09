from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contextweave.core.types import JSON
from contextweave.models.errors import ModelResponseError, ProviderSlotError
from contextweave.models.http import JsonClient
from contextweave.models.providers import (
    DEFAULT_SEED,
    AnnotateClient,
    BuildContext,
    Device,
    EmbedClient,
    EmbedKind,
    Message,
    embed_inputs,
    message_payload,
    parse_structured,
    shape_vectors,
)
from contextweave.models.wire import as_list, as_object, as_str, as_vector, field

CHAIN = (Device.GPU, Device.CPU)
EMBED_PATH = "/api/embed"
CHAT_PATH = "/api/chat"


@dataclass(frozen=True, slots=True)
class OllamaSettings:
    seed: int = DEFAULT_SEED
    gpu_layers: int = 999
    keep_alive: str = "10m"
    load_timeout_s: float = 120.0


class _OllamaClient(ABC):
    def __init__(self, ctx: BuildContext, model: str, settings: OllamaSettings) -> None:
        self._http = JsonClient(ctx.endpoints.ollama, transport=ctx.transport)
        self._deadline_s = ctx.deadline_s
        self._model = model
        self._settings = settings
        self._device = CHAIN[0]

    @property
    def devices(self) -> tuple[Device, ...]:
        return CHAIN

    @property
    def device(self) -> Device:
        return self._device

    def set_device(self, device: Device) -> None:
        """Move residency: unload with keep_alive 0, then reload with the new num_gpu."""
        if device not in CHAIN:
            raise ProviderSlotError(f"ollama has no device '{device}'")
        if device == self._device:
            return
        self._unload()
        self._device = device
        self._load()

    def _num_gpu(self) -> int:
        return self._settings.gpu_layers if self._device is Device.GPU else 0

    @abstractmethod
    def _unload(self) -> None: ...

    @abstractmethod
    def _load(self) -> None: ...


class OllamaEmbed(_OllamaClient):
    """Embeddings via Ollama /api/embed."""

    def __init__(self, ctx: BuildContext, settings: OllamaSettings | None = None) -> None:
        super().__init__(ctx, ctx.models.embed.model, settings or OllamaSettings())
        self._config = ctx.models.embed

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        inputs = embed_inputs(self._config, texts, kind)
        body: dict[str, JSON] = {
            "model": self._model,
            "input": list(inputs),
            "options": {"num_gpu": self._num_gpu()},
        }
        reply = self._http.post(EMBED_PATH, body, timeout_s=self._deadline_s)
        rows = as_list(field(reply, "embeddings", EMBED_PATH), "embeddings")
        vectors = [as_vector(row, "embedding") for row in rows]
        return shape_vectors(self._config, vectors, len(inputs))

    def _unload(self) -> None:
        body: dict[str, JSON] = {"model": self._model, "input": "", "keep_alive": 0}
        self._http.post(EMBED_PATH, body, timeout_s=self._settings.load_timeout_s)

    def _load(self) -> None:
        body: dict[str, JSON] = {
            "model": self._model,
            "input": "",
            "keep_alive": self._settings.keep_alive,
            "options": {"num_gpu": self._num_gpu()},
        }
        self._http.post(EMBED_PATH, body, timeout_s=self._settings.load_timeout_s)


class OllamaAnnotate(_OllamaClient):
    """Structured generation via Ollama /api/chat with format=<schema>."""

    def __init__(self, ctx: BuildContext, settings: OllamaSettings | None = None) -> None:
        super().__init__(ctx, ctx.models.annotate.model, settings or OllamaSettings())
        self._think = ctx.models.annotate.think

    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]:
        body: dict[str, JSON] = {
            "model": self._model,
            "messages": message_payload(prompt_messages),
            "stream": False,
            "format": dict(json_schema),
            "think": self._think,
            "options": {
                "temperature": 0,
                "seed": self._settings.seed,
                "num_gpu": self._num_gpu(),
            },
        }
        reply = self._http.post(CHAT_PATH, body, timeout_s=self._deadline_s)
        message = as_object(field(reply, "message", CHAT_PATH), "message")
        content = as_str(field(message, "content", "message"), "content")
        if not content:
            raise ModelResponseError("empty message content")
        return parse_structured(content, json_schema)

    def _unload(self) -> None:
        body: dict[str, JSON] = {"model": self._model, "messages": [], "keep_alive": 0}
        self._http.post(CHAT_PATH, body, timeout_s=self._settings.load_timeout_s)

    def _load(self) -> None:
        body: dict[str, JSON] = {
            "model": self._model,
            "messages": [],
            "keep_alive": self._settings.keep_alive,
            "options": {"num_gpu": self._num_gpu()},
        }
        self._http.post(CHAT_PATH, body, timeout_s=self._settings.load_timeout_s)


def build_ollama_embed(ctx: BuildContext) -> EmbedClient:
    """Factory for the ollama embed provider."""
    return OllamaEmbed(ctx)


def build_ollama_annotate(ctx: BuildContext) -> AnnotateClient:
    """Factory for the ollama annotate provider."""
    return OllamaAnnotate(ctx)
