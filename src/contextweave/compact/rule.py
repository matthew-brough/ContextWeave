from collections.abc import Callable
from dataclasses import dataclass

from contextweave.compact.sources import SourceCache
from contextweave.retrieve.pack import PackItem


@dataclass(frozen=True, slots=True)
class Rule[T]:
    """One compaction rule: a name, whether it removes no source bytes, and the rewrite."""

    name: str
    lossless: bool
    apply: Callable[[SourceCache, T], T]


@dataclass(frozen=True, slots=True)
class Rung:
    """One lossy ladder step: where it applies and what it turns an item into (None = drop to ref)."""

    name: str
    applies: Callable[[PackItem, bool], bool]
    apply: Callable[[PackItem], PackItem | None]
