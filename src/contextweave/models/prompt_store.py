import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Literal, cast

from contextweave.core.types import JSON
from contextweave.models.errors import ModelResponseError
from contextweave.models.providers import Message
from contextweave.models.wire import to_json

PROMPTS_DIR = Path(__file__).parent / "prompts"
ROLES: tuple[Literal["system", "user", "assistant"], ...] = ("system", "user", "assistant")


class PromptError(Exception):
    """A prompt file is missing, malformed, or at a different version than requested."""


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    version: int
    messages: tuple[Message, ...]
    schema: Mapping[str, JSON]

    def render(self, **fields: str) -> tuple[Message, ...]:
        """Substitute $name placeholders in every message."""
        return tuple(
            Message(message.role, Template(message.content).safe_substitute(fields))
            for message in self.messages
        )


def load_prompt(name: str, version: int, directory: Path = PROMPTS_DIR) -> PromptTemplate:
    """Load prompts/<name>.toml, requiring prompt_version to equal version."""
    path = directory / f"{name}.toml"
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PromptError(f"cannot load prompt '{name}': {exc}") from exc
    found = raw.get("prompt_version")
    if found != version:
        raise PromptError(f"prompt '{name}' is version {found}, config wants {version}")
    messages = tuple(_message(item) for item in _items(raw.get("messages")))
    try:
        schema = to_json(raw.get("schema"))
    except ModelResponseError as exc:
        raise PromptError(f"prompt '{name}' schema invalid: {exc}") from exc
    if not isinstance(schema, dict):
        raise PromptError(f"prompt '{name}' has no [schema] table")
    return PromptTemplate(version, messages, schema)


def _items(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise PromptError("prompt has no [[messages]] entries")
    rows = cast(list[object], value)  # TOML array of tables, each checked below
    items: list[Mapping[str, object]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PromptError("prompt message is not a table")
        items.append(cast(Mapping[str, object], row))  # TOML tables always have str keys
    return items


def _message(item: Mapping[str, object]) -> Message:
    role = item.get("role")
    content = item.get("content")
    for allowed in ROLES:
        if role == allowed and isinstance(content, str):
            return Message(allowed, content)
    raise PromptError("prompt message needs role and content")
