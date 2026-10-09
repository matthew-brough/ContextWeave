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

CHAIN = (Device.REMOTE,)
EMBED_PATH = "/v1/embeddings"
CHAT_PATH = "/v1/chat/completions"
SCHEMA_NAME = "annotation"


@dataclass(frozen=True, slots=True)
class OpenAISettings:
    seed: int = DEFAULT_SEED


def _http(ctx: BuildContext) -> JsonClient:
    key = ctx.endpoints.openai_api_key
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return JsonClient(ctx.endpoints.openai_base_url, transport=ctx.transport, headers=headers)


class _RemoteDevice:
    @property
    def devices(self) -> tuple[Device, ...]:
        return CHAIN

    def set_device(self, device: Device) -> None:
        if device not in CHAIN:
            raise ProviderSlotError(f"openai_compat has only the remote device, not '{device}'")


class OpenAIEmbed(_RemoteDevice):
    """Embeddings via /v1/embeddings on one remote device."""

    def __init__(self, ctx: BuildContext) -> None:
        self._http = _http(ctx)
        self._config = ctx.models.embed
        self._deadline_s = ctx.deadline_s

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        if not texts:
            return []
        inputs = embed_inputs(self._config, texts, kind)
        body: dict[str, JSON] = {"model": self._config.model, "input": list(inputs)}
        reply = self._http.post(EMBED_PATH, body, timeout_s=self._deadline_s)
        rows = as_list(field(reply, "data", EMBED_PATH), "data")
        ordered: dict[int, list[float]] = {}
        for position, row in enumerate(rows):
            item = as_object(row, "data item")
            index = item.get("index", position)
            if isinstance(index, bool) or not isinstance(index, int):
                raise ModelResponseError("data item index is not an integer")
            ordered[index] = as_vector(field(item, "embedding", "data item"), "embedding")
        vectors = [ordered[index] for index in sorted(ordered)]
        return shape_vectors(self._config, vectors, len(inputs))


class OpenAIAnnotate(_RemoteDevice):
    """Structured generation via /v1/chat/completions with response_format json_schema."""

    def __init__(self, ctx: BuildContext, settings: OpenAISettings | None = None) -> None:
        self._http = _http(ctx)
        self._model = ctx.models.annotate.model
        self._deadline_s = ctx.deadline_s
        self._settings = settings or OpenAISettings()

    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]:
        body: dict[str, JSON] = {
            "model": self._model,
            "messages": message_payload(prompt_messages),
            "temperature": 0,
            "seed": self._settings.seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": SCHEMA_NAME, "strict": True, "schema": dict(json_schema)},
            },
        }
        reply = self._http.post(CHAT_PATH, body, timeout_s=self._deadline_s)
        choices = as_list(field(reply, "choices", CHAT_PATH), "choices")
        if not choices:
            raise ModelResponseError("no choices returned")
        first = as_object(choices[0], "choice")
        message = as_object(field(first, "message", "choice"), "message")
        content = as_str(field(message, "content", "message"), "content")
        return parse_structured(content, json_schema)


def build_openai_embed(ctx: BuildContext) -> EmbedClient:
    """Factory for the openai_compat embed provider."""
    return OpenAIEmbed(ctx)


def build_openai_annotate(ctx: BuildContext) -> AnnotateClient:
    """Factory for the openai_compat annotate provider."""
    return OpenAIAnnotate(ctx)
