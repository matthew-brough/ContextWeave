import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tokenizers import Tokenizer

from contextweave.core.budget import DEFAULT_ESTIMATOR, Estimator, estimators
from contextweave.core.paths import user_config_dir
from contextweave.core.registry import Registry


class ChunkTokenizer(Protocol):
    """Counts the tokens chunk sizing sees; the embed model's tokenizer plugs in here."""

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class EstimatorTokenizer:
    """A ChunkTokenizer backed by a budget estimator."""

    estimator: Estimator

    def count(self, text: str) -> int:
        return self.estimator(text)


DEFAULT_TOKENIZER = "bytes35"

tokenizers: Registry[str, ChunkTokenizer] = Registry()
tokenizers.register(DEFAULT_TOKENIZER, EstimatorTokenizer(estimators.get(DEFAULT_ESTIMATOR)))


def default_tokenizer() -> ChunkTokenizer:
    """The registered tokenizer chunk sizing uses unless given another."""
    return tokenizers.get(DEFAULT_TOKENIZER)


@dataclass(frozen=True, slots=True)
class FileTokenizer:
    """A ChunkTokenizer backed by a local tokenizer.json."""

    tokenizer: Tokenizer

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def model_slug(model: str) -> str:
    """Filesystem-safe form of a model name."""
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def tokenizer_file(model: str, env: Mapping[str, str]) -> Path:
    """Where a model's tokenizer.json is looked up: <config>/tokenizers/<slug>/tokenizer.json."""
    return user_config_dir(env) / "tokenizers" / model_slug(model) / "tokenizer.json"


def register_file_tokenizer(name: str, path: Path) -> ChunkTokenizer | None:
    """Register the tokenizer.json at path under name; None when absent or unreadable."""
    if name in tokenizers:
        return tokenizers.get(name)
    if not path.is_file():
        return None
    try:
        loaded = Tokenizer.from_file(str(path))
    except Exception:
        return None
    return tokenizers.register(name, FileTokenizer(loaded))


def embed_tokenizer(model: str, env: Mapping[str, str]) -> ChunkTokenizer:
    """The embed model's local tokenizer when one is present, else the estimator."""
    found = register_file_tokenizer(f"file:{model_slug(model)}", tokenizer_file(model, env))
    return found or default_tokenizer()
