from dataclasses import replace

import pytest
from retrieve_support import ingest, make_ctx

from contextweave.core.config import Config, FuseConfig, PipelineConfig
from contextweave.index.store import Store
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.context import Mode, QueryContext
from contextweave.retrieve.stages import (
    STAGE_SLOTS,
    StageRegistry,
    default_registry,
    run_pipeline,
)


class Recorder:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        self.log.append(self.name)
        return cands


class Exploding:
    name = "fuse"

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        raise RuntimeError("boom")


def seed(store: Store) -> None:
    ingest(store, "pkg/widget.py", "class Widget:\n    def render(self):\n        return 1\n")
    ingest(store, "pkg/other.py", "def widget_factory():\n    return 1\n")


def test_registry_order_ignores_registration_order(store: Store) -> None:
    log: list[str] = []
    registry = StageRegistry()
    for slot in reversed(STAGE_SLOTS):
        registry.register(Recorder(slot.name, log))
    run_pipeline(make_ctx(store, "x"), registry)
    assert log == [slot.name for slot in STAGE_SLOTS]


def test_unknown_stage_rejected() -> None:
    with pytest.raises(ValueError, match="unknown pipeline stage"):
        StageRegistry().register(Recorder("rerank", []))


def test_toggles_disable_without_reordering(store: Store) -> None:
    log: list[str] = []
    registry = StageRegistry()
    for slot in STAGE_SLOTS:
        registry.register(Recorder(slot.name, log))
    config = replace(Config(), pipeline=PipelineConfig(gate=False, plan=False, assemble=False))
    run = run_pipeline(make_ctx(store, "x", config=config), registry)
    assert log == ["retrieve", "fuse", "expand", "dedupe", "compact", "coverage"]
    assert [timing.stage for timing in run.timings] == log
    assert all(timing.elapsed_ms >= 0 and not timing.failed for timing in run.timings)


def test_raising_stage_degrades_and_pipeline_continues(store: Store) -> None:
    seed(store)
    log: list[str] = []
    registry = default_registry()
    registry.register(Exploding())
    registry.register(Recorder("gate", log))
    ctx = make_ctx(store, "Widget")
    run = run_pipeline(ctx, registry)
    assert log == ["gate"]
    assert len(run.cands) > 0
    assert ctx.envelope.degraded == [{"stage": "fuse", "reason": "boom"}]
    assert ctx.envelope.errors[0].get("stage") == "fuse"
    assert [(t.stage, t.failed) for t in run.timings] == [
        ("retrieve", False),
        ("fuse", True),
        ("gate", False),
    ]


def retrieve_with(
    store: Store, mode: Mode, config: Config | None = None
) -> tuple[QueryContext, CandidateSet]:
    ctx = make_ctx(store, "Widget", mode=mode, config=config)
    return ctx, run_pipeline(ctx, default_registry()).cands


def test_mode_lexical_forces_lexical_only(store: Store) -> None:
    seed(store)
    _, cands = retrieve_with(store, "lexical")
    assert len(cands) > 0
    assert all(c.ranks.lexical is not None and c.ranks.structural is None for c in cands)


def test_mode_structural_forces_structural_only(store: Store) -> None:
    seed(store)
    _, cands = retrieve_with(store, "structural")
    assert len(cands) > 0
    assert all(c.ranks.structural is not None and c.ranks.lexical is None for c in cands)


def test_mode_auto_combines_strategies(store: Store) -> None:
    seed(store)
    _, cands = retrieve_with(store, "auto")
    assert any(c.ranks.lexical and c.ranks.structural for c in cands)
    keys = [(c.ranks.best() or 0, *c.order_key) for c in cands]
    assert keys == sorted(keys)


def test_mode_semantic_degrades_until_vectors_exist(store: Store) -> None:
    seed(store)
    ctx, cands = retrieve_with(store, "semantic")
    assert len(cands) == 0
    assert ctx.envelope.degraded == [{"stage": "retrieve", "reason": "semantic_unavailable"}]


def test_fuse_keep_caps_each_strategy(store: Store) -> None:
    for index in range(8):
        ingest(store, f"m{index}.py", f"def widget_{index}():\n    return 1\n")
    config = replace(Config(), fuse=FuseConfig(keep=3))
    _, cands = retrieve_with(store, "lexical", config)
    assert len(cands) == 3
    _, cands = retrieve_with(store, "structural", config)
    assert len(cands) == 3


def test_scope_override_and_plan_scope(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "Widget", scope="pkg/other.py")
    cands = run_pipeline(ctx, default_registry()).cands
    assert {c.path for c in cands} == {"pkg/other.py"}
