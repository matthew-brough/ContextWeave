from dataclasses import dataclass

from contextweave.core.config import Config
from contextweave.core.errors import OmittedEntry
from contextweave.core.refs import project_key
from contextweave.index.store import Store
from contextweave.retrieve.assemble import AssembleStage
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.context import Mode, QueryContext
from contextweave.retrieve.dedupe import DedupeStage
from contextweave.retrieve.fuse import FuseStage
from contextweave.retrieve.pack import ContextPack, PackResult
from contextweave.retrieve.pack_text import TextRender, render_pack_text
from contextweave.retrieve.plan import PlanStage
from contextweave.retrieve.score import FallbackScoreStage
from contextweave.retrieve.stages import StageRegistry, StageTiming, default_registry, run_pipeline


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """context_search arguments; max_tokens None uses budget.search_max_tokens."""

    scope: str | None = None
    intent: str | None = None
    max_tokens: int | None = None
    mode: Mode = "auto"
    render: TextRender = render_pack_text


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The ContextPack, its text fallback, uncapped omitted entries and stage timings."""

    pack: ContextPack
    text: str
    omitted: tuple[OmittedEntry, ...]
    timings: tuple[StageTiming, ...]


def full_registry(assembler: AssembleStage) -> StageRegistry:
    """Every implemented stage: plan, retrieve, fuse, fallback gate, dedupe, assemble."""
    registry = default_registry()
    registry.register(PlanStage())
    registry.register(FuseStage())
    registry.register(FallbackScoreStage())
    registry.register(DedupeStage())
    registry.register(assembler)
    return registry


def search(
    store: Store, config: Config, query: str, options: SearchOptions | None = None
) -> SearchResult:
    """Run the full pipeline; stage failures degrade into the pack, nothing raises."""
    chosen = options or SearchOptions()
    ctx = QueryContext(
        query=query,
        store=store,
        config=config,
        project_key=project_key(store.root),
        root=store.root,
        scope=chosen.scope,
        intent=chosen.intent,
        max_tokens=chosen.max_tokens or config.budget.search_max_tokens,
        mode=chosen.mode,
    )
    assembler = AssembleStage(chosen.render)
    run = run_pipeline(ctx, full_registry(assembler))
    result: PackResult = ctx.pack.value or assembler.assemble(ctx, CandidateSet())
    return SearchResult(result.pack, result.text, result.omitted, run.timings)


def context_search(
    store: Store, config: Config, query: str, options: SearchOptions | None = None
) -> ContextPack:
    """The ContextPack payload for a query."""
    return search(store, config, query, options).pack
