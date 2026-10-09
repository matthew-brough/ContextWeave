from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace

from contextweave.core.refs import Anchor
from contextweave.index.records import FileProfile
from contextweave.lang.registry import Group


@dataclass(frozen=True, slots=True)
class Ranks:
    """1-based rank per retrieval strategy; None when the strategy did not return the chunk."""

    lexical: int | None = None
    semantic: int | None = None
    structural: int | None = None

    def best(self) -> int | None:
        present = [rank for rank in (self.lexical, self.semantic, self.structural) if rank]
        return min(present) if present else None


@dataclass(frozen=True, slots=True)
class Scores:
    """ContextPack score terms; unset terms are 0.0."""

    final: float = 0.0
    gate: float = 0.0
    relevance: float = 0.0
    answers_query: float = 0.0
    implementation: float = 0.0
    structural_value: float = 0.0
    lexical_match: float = 0.0
    graph_proximity: float = 0.0
    redundancy: float = 0.0


@dataclass(frozen=True, slots=True)
class Candidate:
    """One chunk moving through the pipeline."""

    chunk_id: int
    path: str
    start_byte: int
    end_byte: int
    anchor: Anchor
    profile: FileProfile
    group: Group | None
    qualified_name: str | None = None
    kind: str | None = None
    ranks: Ranks = field(default_factory=Ranks)
    scores: Scores = field(default_factory=Scores)

    @property
    def order_key(self) -> tuple[str, int, int]:
        """The (path, start_byte) tie-break of V12, then end_byte."""
        return (self.path, self.start_byte, self.end_byte)

    def with_ranks(self, ranks: Ranks) -> Candidate:
        return replace(self, ranks=ranks)

    def with_scores(self, scores: Scores) -> Candidate:
        return replace(self, scores=scores)


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """An ordered, immutable sequence of candidates."""

    items: tuple[Candidate, ...] = ()

    @classmethod
    def of(cls, items: Iterable[Candidate]) -> CandidateSet:
        return cls(tuple(items))

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Candidate:
        return self.items[index]

    def top(self, count: int) -> CandidateSet:
        return CandidateSet(self.items[:count])

    def filter(self, keep: Callable[[Candidate], bool]) -> CandidateSet:
        return CandidateSet(tuple(item for item in self.items if keep(item)))

    def map(self, change: Callable[[Candidate], Candidate]) -> CandidateSet:
        return CandidateSet(tuple(change(item) for item in self.items))

    def appended(self, items: Iterable[Candidate]) -> CandidateSet:
        return CandidateSet((*self.items, *items))

    def chunk_ids(self) -> tuple[int, ...]:
        return tuple(item.chunk_id for item in self.items)
