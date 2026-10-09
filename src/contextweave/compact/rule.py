from collections.abc import Callable
from dataclasses import dataclass

from contextweave.compact.sources import SourceCache


@dataclass(frozen=True, slots=True)
class Rule[T]:
    """One compaction rule: a name, whether it removes no source bytes, and the rewrite."""

    name: str
    lossless: bool
    apply: Callable[[SourceCache, T], T]
