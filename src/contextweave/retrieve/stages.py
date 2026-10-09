import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from contextweave.core.config import PipelineConfig
from contextweave.retrieve.candidates import Candidate, CandidateSet, Ranks, Scores
from contextweave.retrieve.context import Mode, Plan, PlanSlot, QueryContext, RrfWeights
from contextweave.retrieve.retrieve_stage import RetrieveStage

__all__ = [
    "STAGE_SLOTS",
    "Candidate",
    "CandidateSet",
    "Mode",
    "PipelineRun",
    "Plan",
    "PlanSlot",
    "QueryContext",
    "Ranks",
    "RrfWeights",
    "Scores",
    "Stage",
    "StageRegistry",
    "StageTiming",
    "default_registry",
    "run_pipeline",
]


class Stage(Protocol):
    """One pipeline step: candidates in, candidates out."""

    name: str

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet: ...


@dataclass(frozen=True, slots=True)
class StageSlot:
    """A fixed position in the pipeline and the config switch that enables it."""

    name: str
    enabled: Callable[[PipelineConfig], bool]


STAGE_SLOTS: tuple[StageSlot, ...] = (
    StageSlot("plan", lambda config: config.plan),
    StageSlot("retrieve", lambda config: config.retrieve),
    StageSlot("fuse", lambda config: config.fuse),
    StageSlot("gate", lambda config: config.gate),
    StageSlot("expand", lambda config: config.expand),
    StageSlot("dedupe", lambda config: config.dedupe),
    StageSlot("compact", lambda config: config.compact),
    StageSlot("coverage", lambda config: config.coverage),
    StageSlot("assemble", lambda config: config.assemble),
)


class StageRegistry:
    """Stages keyed by name; iteration follows STAGE_SLOTS regardless of registration order."""

    def __init__(self) -> None:
        self._stages: dict[str, Stage] = {}

    def register(self, stage: Stage) -> None:
        if stage.name not in {slot.name for slot in STAGE_SLOTS}:
            raise ValueError(f"unknown pipeline stage: {stage.name}")
        self._stages[stage.name] = stage

    def __contains__(self, name: str) -> bool:
        return name in self._stages

    def __iter__(self) -> Iterator[tuple[StageSlot, Stage]]:
        for slot in STAGE_SLOTS:
            stage = self._stages.get(slot.name)
            if stage is not None:
                yield slot, stage


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Wall time of one executed stage and whether it raised."""

    stage: str
    elapsed_ms: float
    failed: bool


@dataclass(frozen=True, slots=True)
class PipelineRun:
    """Final candidates plus per-stage timings in execution order."""

    cands: CandidateSet
    timings: tuple[StageTiming, ...]


def default_registry() -> StageRegistry:
    """The registry holding every implemented stage."""
    registry = StageRegistry()
    registry.register(RetrieveStage())
    return registry


def run_pipeline(ctx: QueryContext, registry: StageRegistry) -> PipelineRun:
    """Run enabled stages in registry order; a raising stage degrades and passes input through."""
    cands = CandidateSet()
    timings: list[StageTiming] = []
    for slot, stage in registry:
        if not slot.enabled(ctx.config.pipeline):
            continue
        errors_before = len(ctx.envelope.errors)
        started = time.perf_counter()
        with ctx.envelope.guard(slot.name):
            cands = stage.run(ctx, cands)
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings.append(StageTiming(slot.name, elapsed_ms, len(ctx.envelope.errors) > errors_before))
    return PipelineRun(cands, tuple(timings))
