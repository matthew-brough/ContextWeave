from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict

from contextweave.core.errors import DegradedEntry, ErrorEntry, OmittedEntry, OverflowEntry
from contextweave.retrieve.candidates import Candidate

type Rep = Literal["source", "signature", "collapsed"]

LOSSLESS_REPS: frozenset[Rep] = frozenset({"source", "collapsed"})


class RrfWire(TypedDict):
    lexical: float
    semantic: float
    structural: float


class DevicesWire(TypedDict):
    embed: str
    classify: str


class PlanWire(TypedDict):
    intent: str
    scope: str | None
    rrf_weights: RrfWire
    rels: list[str]
    compaction: str
    fast_path: bool
    planner: str
    devices: DevicesWire


class CoverageWire(TypedDict):
    score: float
    sufficient: bool
    iterations: int
    missing: list[str]


class ScoresWire(TypedDict):
    final: float
    gate: float
    relevance: float
    answers_query: float
    implementation: float
    structural_value: float
    lexical_match: float
    graph_proximity: float
    redundancy: float


class EdgeWire(TypedDict):
    rel: str
    to_ref: str
    resolution: str


class ItemWire(TypedDict):
    ref: str
    rep: str
    profile: str
    group: str | None
    path: str
    start_line: int
    end_line: int
    qualified_name: str | None
    kind: str | None
    why: str
    scores: ScoresWire
    edges: list[EdgeWire]
    freshness: str
    synthesized: bool
    oversize: bool
    text: str


class BudgetWire(TypedDict):
    served_tokens: int
    ceiling: int
    truncated: bool
    hint: NotRequired[str]


class ContextPack(TypedDict):
    pack_id: str
    plan: PlanWire
    coverage: CoverageWire
    items: list[ItemWire]
    omitted: list[OmittedEntry | OverflowEntry]
    budget: BudgetWire
    degraded: list[DegradedEntry]
    errors: list[ErrorEntry]


@dataclass(frozen=True, slots=True)
class PackItem:
    """One served chunk: its candidate, ref, current representation and text."""

    cand: Candidate
    ref: str
    rep: Rep
    text: str
    signature: str | None
    start_line: int
    end_line: int
    oversize: bool

    def with_signature_rep(self) -> PackItem:
        return replace(self, rep="signature", text=self.signature or self.text)

    def to_wire(self) -> ItemWire:
        cand = self.cand
        scores = cand.scores
        return ItemWire(
            ref=self.ref,
            rep=self.rep,
            profile=cand.profile,
            group=cand.group,
            path=cand.path,
            start_line=self.start_line,
            end_line=self.end_line,
            qualified_name=cand.qualified_name,
            kind=cand.kind,
            why=cand.why,
            scores=ScoresWire(
                final=scores.final,
                gate=scores.gate,
                relevance=scores.relevance,
                answers_query=scores.answers_query,
                implementation=scores.implementation,
                structural_value=scores.structural_value,
                lexical_match=scores.lexical_match,
                graph_proximity=scores.graph_proximity,
                redundancy=scores.redundancy,
            ),
            edges=[
                EdgeWire(rel=edge.rel, to_ref=edge.to_ref, resolution=edge.resolution)
                for edge in cand.edges
            ],
            freshness="fresh",
            synthesized=False,
            oversize=self.oversize,
            text=self.text,
        )

    def to_omitted(self, reason: str) -> OmittedEntry:
        cand = self.cand
        name = cand.qualified_name or f"b{cand.start_byte}-{cand.end_byte}"
        return OmittedEntry(ref=self.ref, qualified_name=name, path=cand.path, reason=reason)


@dataclass(frozen=True, slots=True)
class PackResult:
    """The assembled pack, its text fallback and the uncapped omitted list."""

    pack: ContextPack
    text: str
    omitted: tuple[OmittedEntry, ...]
