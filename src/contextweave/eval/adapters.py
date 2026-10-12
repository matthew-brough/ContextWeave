import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from contextweave.core.budget import bytes35
from contextweave.eval.symbols import SymbolMap, build_symbol_map, indexed_symbol_map


@dataclass(frozen=True, slots=True)
class AdapterContext:
    """What an adapter needs to serve queries against one checkout."""

    root: Path
    profile: str = "nomodel"
    ablate: str | None = None
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    expected: Mapping[str, Sequence[str]] = field(default_factory=dict[str, Sequence[str]])
    live: bool = False


@dataclass(frozen=True, slots=True)
class ScoreSample:
    """A classifier probability for one question and whether the served item was expected."""

    question: str
    prob: float
    label: float


@dataclass(frozen=True, slots=True)
class Served:
    """What a system returned for one query; error marks a degraded result."""

    items: tuple[str, ...] = ()
    tokens: int = 0
    latency_ms: float = 0.0
    model_calls: int = 0
    verdict_covered: bool | None = None
    error: str | None = None
    samples: tuple[ScoreSample, ...] = ()
    skips: int = 0


class Adapter(Protocol):
    """A system under test."""

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]: ...


type AdapterFactory = Callable[[AdapterContext], Adapter]


def degraded(count: int, error: str) -> tuple[Served, ...]:
    """The same error result for every query."""
    return tuple(Served(error=error) for _ in range(count))


def _symbol_map(context: AdapterContext) -> SymbolMap:
    if not context.live:
        return build_symbol_map(context.root)
    names = {name for expected in context.expected.values() for name in expected}
    return indexed_symbol_map(context.root, names)


class ReadAdapter:
    """Baseline that serves whole files of the expected symbols' paths."""

    def __init__(self, context: AdapterContext, symbols: SymbolMap | None = None) -> None:
        self._context = context
        self._symbols = symbols if symbols is not None else _symbol_map(context)

    def _serve(self, query: str) -> Served:
        started = time.perf_counter()
        paths: dict[str, None] = {}
        missing: list[str] = []
        for name in self._context.expected.get(query, ()):
            found = self._symbols.paths_of(name)
            if not found:
                missing.append(name)
            paths.update(dict.fromkeys(found))
        items: dict[str, None] = {}
        tokens = 0
        for path in paths:
            text = (self._context.root / path).read_text(encoding="utf-8", errors="replace")
            tokens += bytes35(text)
            items.update(dict.fromkeys(self._symbols.by_path.get(path, ())))
        error = f"unresolved: {', '.join(missing)}" if missing else None
        elapsed = (time.perf_counter() - started) * 1000
        return Served(tuple(items), tokens, elapsed, error=error)

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]:
        return tuple(self._serve(query) for query in queries)
