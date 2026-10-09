from dataclasses import replace

from retrieve_support import ingest, make_ctx

from contextweave.index.store import Store
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.plan import (
    INTENT_RELS,
    INTENTS,
    PlanStage,
    rules_intent,
    rules_scope,
    rules_weights,
)
from contextweave.retrieve.stages import StageRegistry, run_pipeline


class Recorder:
    def __init__(self, name: str, log: list[str]) -> None:
        self.name = name
        self.log = log

    def run(self, ctx: object, cands: CandidateSet) -> CandidateSet:
        self.log.append(self.name)
        return cands


def seed(store: Store) -> None:
    ingest(store, "pkg/widget.py", "class Widget:\n    def render(self):\n        return 1\n")
    ingest(store, "pkg/other.py", "class Other:\n    def render(self):\n        return 2\n")
    ingest(
        store,
        "pkg/dup.py",
        "def twin():\n    return 1\n\n\nclass Box:\n    def twin(self):\n        return 2\n",
    )


def test_intent_rules() -> None:
    assert rules_intent("Widget") == "lookup"
    assert rules_intent("why does the parser crash") == "debug"
    assert rules_intent("where is the config loaded") == "config"
    assert rules_intent("architecture of the indexer") == "architecture"
    assert rules_intent("who calls the parser") == "usage"
    assert rules_intent("render a widget tree") == "implementation"
    assert set(INTENT_RELS) == set(INTENTS)
    assert INTENT_RELS["lookup"] == ()


def test_weights_rules() -> None:
    assert rules_weights("Widget").structural > rules_weights("Widget").lexical
    assert rules_weights("how does the widget get rendered here").semantic > 1.0
    assert rules_weights("render widget") == rules_weights("render widget")


def test_scope_rule_requires_indexed_path(store: Store) -> None:
    seed(store)
    assert rules_scope(store, "render in pkg/other.py please") == "pkg/other.py"
    assert rules_scope(store, "look at pkg/ now") == "pkg"
    assert rules_scope(store, "look at nowhere/else") is None


def test_fast_path_single_symbol(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "Widget")
    out = PlanStage().run(ctx, CandidateSet())
    assert ctx.plan.value is not None
    assert ctx.plan.value.fast_path
    assert ctx.plan.value.intent == "lookup"
    assert ctx.plan.value.rels == ()
    assert [c.qualified_name for c in out] == ["Widget"]
    assert out[0].critical
    assert out[0].scores.gate == 1.0
    assert out[0].scores.final > 0


def test_fast_path_same_name_and_kind_returns_all(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "render")
    out = PlanStage().run(ctx, CandidateSet())
    assert ctx.plan.value is not None and ctx.plan.value.fast_path
    assert {c.path for c in out} == {"pkg/widget.py", "pkg/other.py"}


def test_ambiguous_name_kind_skips_fast_path(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "twin")
    out = PlanStage().run(ctx, CandidateSet())
    assert ctx.plan.value is not None and not ctx.plan.value.fast_path
    assert len(out) == 0


def test_multi_word_and_forced_mode_skip_fast_path(store: Store) -> None:
    seed(store)
    multi = make_ctx(store, "Widget render")
    PlanStage().run(multi, CandidateSet())
    assert multi.plan.value is not None and not multi.plan.value.fast_path
    forced = make_ctx(store, "Widget", mode="lexical")
    PlanStage().run(forced, CandidateSet())
    assert forced.plan.value is not None and not forced.plan.value.fast_path


def test_scope_override_and_intent_override(store: Store) -> None:
    seed(store)
    scoped = make_ctx(store, "render widget tree", scope="./pkg/other.py/")
    PlanStage().run(scoped, CandidateSet())
    assert scoped.plan.value is not None and scoped.plan.value.scope == "pkg/other.py"
    debug = replace(make_ctx(store, "render widget tree"), intent="debug")
    PlanStage().run(debug, CandidateSet())
    assert debug.plan.value is not None and debug.plan.value.intent == "debug"
    assert debug.plan.value.rels == INTENT_RELS["debug"]


def test_invalid_intent_reports_error_and_uses_rules(store: Store) -> None:
    ctx = replace(make_ctx(store, "render widget tree"), intent="nonsense")
    PlanStage().run(ctx, CandidateSet())
    assert ctx.plan.value is not None and ctx.plan.value.intent == "implementation"
    assert ctx.envelope.errors[0]["code"] == "invalid_intent"


def test_fast_path_skips_stages_two_to_eight(store: Store) -> None:
    seed(store)
    log: list[str] = []
    registry = StageRegistry()
    registry.register(PlanStage())
    for name in ("retrieve", "fuse", "gate", "expand", "dedupe", "compact", "coverage", "assemble"):
        registry.register(Recorder(name, log))
    run_pipeline(make_ctx(store, "Widget"), registry)
    assert log == ["assemble"]
    log.clear()
    run_pipeline(make_ctx(store, "render widget tree"), registry)
    assert log[0] == "retrieve" and log[-1] == "assemble" and len(log) == 8
