from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from contextweave.core.config import Config
from contextweave.core.errors import Envelope
from contextweave.index.store import Store

type Mode = Literal["auto", "lexical", "semantic", "structural"]
type Planner = Literal["rules", "model"]

MODES: tuple[Mode, ...] = ("auto", "lexical", "semantic", "structural")


@dataclass(frozen=True, slots=True)
class RrfWeights:
    """Per-strategy weights of the reciprocal rank fusion."""

    lexical: float = 1.0
    semantic: float = 1.0
    structural: float = 1.0


@dataclass(frozen=True, slots=True)
class Plan:
    """The planner's decisions, filled by the plan stage."""

    intent: str
    scope: str | None = None
    rrf_weights: RrfWeights = field(default_factory=RrfWeights)
    rels: tuple[str, ...] = ()
    compaction: str = "lossless"
    fast_path: bool = False
    planner: Planner = "rules"


class PlanSlot:
    """Mutable holder the plan stage fills and later stages read."""

    def __init__(self) -> None:
        self.value: Plan | None = None


@dataclass(frozen=True, slots=True)
class QueryContext:
    """Everything a stage needs to answer one query."""

    query: str
    store: Store
    config: Config
    project_key: str
    root: Path
    envelope: Envelope = field(default_factory=Envelope)
    plan: PlanSlot = field(default_factory=PlanSlot)
    scope: str | None = None
    intent: str | None = None
    max_tokens: int = 6000
    mode: Mode = "auto"
