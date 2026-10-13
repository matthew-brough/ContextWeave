import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import PurePath

from contextweave.core.config import EmbedConfig
from contextweave.core.types import JSON
from contextweave.models.device_http import DeviceHttp, device_table
from contextweave.models.errors import ModelTransportError
from contextweave.models.openai_compat import CHAT_PATH, chat_body, embed_over, parse_chat
from contextweave.models.providers import (
    DEFAULT_SEED,
    AnnotateClient,
    BuildContext,
    EmbedClient,
    EmbedKind,
    Message,
)

SERVER_BINARY = "llama-server"
UBATCH_FLAGS = ("-ub", "--ubatch-size")
BATCH_FLAGS = ("-b", "--batch-size")
DEFAULT_UBATCH = 512
DEFAULT_BATCH = 2048


def _flag(argv: Sequence[str], names: Sequence[str], default: int) -> int:
    for flag, value in pairwise(argv):
        if flag in names and value.isdigit():
            return int(value)
    return default


def embed_launch_values(config: EmbedConfig) -> Mapping[str, str]:
    """Launch placeholders that tie the embed server's sizing to the client's input cap."""
    return {
        "max_input_tokens": str(config.max_input_tokens),
        "ctx": str(config.max_input_tokens * max(config.parallel, 1)),
    }


def guard_ubatch(expand: Callable[[str], str], config: EmbedConfig) -> Callable[[str], str]:
    """Refuse a llama-server embed launch whose physical batch cannot hold one max_input_tokens input."""

    def guarded(command: str) -> str:
        expanded = expand(command)
        argv = shlex.split(expanded)
        if not any(PurePath(arg).name == SERVER_BINARY for arg in argv):
            return expanded
        ubatch = min(
            _flag(argv, UBATCH_FLAGS, DEFAULT_UBATCH), _flag(argv, BATCH_FLAGS, DEFAULT_BATCH)
        )
        if ubatch < config.max_input_tokens:
            raise ModelTransportError(
                f"embed launch batch {ubatch} < models.embed.max_input_tokens "
                f"{config.max_input_tokens}; use -b {{max_input_tokens}} -ub {{max_input_tokens}}"
            )
        return expanded

    return guarded


@dataclass(frozen=True, slots=True)
class LlamaSettings:
    seed: int = DEFAULT_SEED


class LlamaEmbed(DeviceHttp):
    """Embeddings via llama-server /v1/embeddings, one server per device."""

    def __init__(self, ctx: BuildContext) -> None:
        config = ctx.models.embed
        super().__init__(
            "embed", config.devices, device_table(config.gpu, config.cpu), ctx.transport
        )
        self._config = config
        self._deadline_s = ctx.embed_deadline_s

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        return embed_over(
            self.active(), self._config, texts, kind, timeout_s=self._deadline_s(kind)
        )


class LlamaAnnotate(DeviceHttp):
    """Structured generation via llama-server /v1/chat/completions with a JSON schema."""

    def __init__(self, ctx: BuildContext, settings: LlamaSettings | None = None) -> None:
        config = ctx.models.annotate
        super().__init__(
            "annotate", config.devices, device_table(config.gpu, config.cpu), ctx.transport
        )
        self._model = config.model
        self._think = config.think
        self._max_tokens = config.max_output_tokens
        self._deadline_s = ctx.annotate_deadline_s
        self._settings = settings or LlamaSettings()

    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]:
        body = chat_body(
            self._model,
            prompt_messages,
            json_schema,
            seed=self._settings.seed,
            max_tokens=self._max_tokens,
        )
        body["chat_template_kwargs"] = {"enable_thinking": self._think}
        reply = self.active().post(CHAT_PATH, body, timeout_s=self._deadline_s)
        return parse_chat(reply, json_schema)


def build_llama_embed(ctx: BuildContext) -> EmbedClient:
    """Factory for the llama_server embed provider."""
    return LlamaEmbed(ctx)


def build_llama_annotate(ctx: BuildContext) -> AnnotateClient:
    """Factory for the llama_server annotate provider."""
    return LlamaAnnotate(ctx)
