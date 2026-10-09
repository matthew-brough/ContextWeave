from dataclasses import replace

from fake_decision import FakeDecision, fake_slots
from retrieve_support import ingest, make_ctx

from contextweave.core.config import Config
from contextweave.core.errors import DegradedEntry
from contextweave.index.edges import resolve_edges
from contextweave.index.store import Store
from contextweave.retrieve.candidates import CandidateSet
from contextweave.retrieve.context import Plan
from contextweave.retrieve.coverage import CoverageStage
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int
from contextweave.retrieve.search import SearchOptions, context_search, search
from contextweave.retrieve.stages import STAGE_SLOTS, StageRegistry

PARSER = (
    "def parse_tokens(text):\n    return text.split()\n\n\ndef parse_config(text):\n    return {}\n"
)
CHAIN = {
    "pkg/a.py": "from pkg.b import middle\n\n\ndef top() -> int:\n    return middle()\n",
    "pkg/b.py": "def middle() -> int:\n    return 1\n",
}


def seed(store: Store) -> None:
    ingest(store, "pkg/parser.py", PARSER)
    ingest(store, "pkg/widget.py", "class Widget:\n    def render(self):\n        return 1\n")


def seed_chain(store: Store) -> None:
    for path, text in CHAIN.items():
        ingest(store, path, text)
    resolve_edges(store, ())


def asked_names(client: FakeDecision) -> set[str]:
    return {question.name for batch in client.asked for question in batch}


def stages_of(entries: list[DegradedEntry]) -> set[str]:
    return {entry["stage"] for entry in entries}


def test_gate_scores_come_from_the_classifier(store: Store) -> None:
    seed(store)
    client = FakeDecision(yes=lambda state: 0.9)
    pack = context_search(
        store, Config(), "parse the config text", SearchOptions(models=fake_slots(client))
    )
    assert {"answers_query", "relevance", "structural_value", "role"} <= asked_names(client)
    assert pack["items"]
    assert pack["items"][0]["scores"]["answers_query"] == 0.9
    assert pack["plan"]["devices"]["classify"] == "gpu"
    assert not stages_of(pack["degraded"]) & {"plan", "gate", "expand", "coverage"}


def test_skipped_slot_keeps_fallback_ranking(store: Store) -> None:
    seed(store)
    baseline = context_search(store, Config(), "parse the config text")
    down = FakeDecision(down=True)
    pack = context_search(
        store, Config(), "parse the config text", SearchOptions(models=fake_slots(down))
    )
    assert [i["ref"] for i in pack["items"]] == [i["ref"] for i in baseline["items"]]
    assert pack["errors"] == []
    assert "gate" in stages_of(pack["degraded"])


def test_unhealthy_slot_never_reaches_the_client_again(store: Store) -> None:
    seed(store)
    client = FakeDecision(down=True)
    slots = fake_slots(client, clock=lambda: 0.0)
    for _ in range(4):
        search(store, Config(), "parse the config text", SearchOptions(models=slots))
    client.calls.clear()
    pack = context_search(store, Config(), "parse the config text", SearchOptions(models=slots))
    assert client.calls == []
    assert pack["items"]
    assert pack["plan"]["devices"]["classify"] == "none"


def test_fast_path_skips_the_model_plan(store: Store) -> None:
    seed(store)
    client = FakeDecision()
    pack = context_search(store, Config(), "Widget", SearchOptions(models=fake_slots(client)))
    assert pack["plan"]["fast_path"] is True
    assert "intent" not in asked_names(client)


def test_model_plan_runs_when_there_is_no_fast_path(store: Store) -> None:
    seed(store)
    client = FakeDecision(choices={"intent": "debug"}, yes=lambda state: 0.9)
    pack = context_search(
        store, Config(), "parse the config text", SearchOptions(models=fake_slots(client))
    )
    assert pack["plan"]["planner"] == "model"
    assert pack["plan"]["intent"] == "debug"
    assert "callers" in pack["plan"]["rels"]


def test_skipped_plan_keeps_rules_plan(store: Store) -> None:
    seed(store)
    rules = context_search(store, Config(), "parse the config text")["plan"]
    pack = context_search(
        store,
        Config(),
        "parse the config text",
        SearchOptions(models=fake_slots(FakeDecision(down=True))),
    )
    assert pack["plan"]["intent"] == rules["intent"]
    assert pack["plan"]["planner"] == rules["planner"]
    assert "plan" in stages_of(pack["degraded"])


def test_explicit_intent_beats_model_intent(store: Store) -> None:
    seed(store)
    client = FakeDecision(choices={"intent": "debug"})
    pack = context_search(
        store,
        Config(),
        "parse the config text",
        SearchOptions(intent="config", models=fake_slots(client)),
    )
    assert pack["plan"]["intent"] == "config"


def test_coverage_reports_sufficient_pack(store: Store) -> None:
    seed(store)
    client = FakeDecision(yes=lambda state: 0.9, choices={"missing_rel": "none"})
    pack = context_search(
        store, Config(), "parse the config text", SearchOptions(models=fake_slots(client))
    )
    assert pack["coverage"]["sufficient"] is True
    assert pack["coverage"]["iterations"] == 0


def test_coverage_expands_over_the_missing_relation_up_to_the_cap(store: Store) -> None:
    seed_chain(store)

    def yes(state: str) -> float:
        return 0.1 if "pack outline" in state else 0.9

    client = FakeDecision(yes=yes, choices={"missing_rel": "callees"})
    slots = fake_slots(client)
    ctx = make_ctx(store, "top", models=slots)
    ctx.plan.value = Plan("debug", rels=("callees",))
    out = CoverageStage().run(ctx, chain_seed(store))
    assert {cand.path for cand in out} >= {"pkg/a.py", "pkg/b.py"}
    result = ctx.coverage.value
    assert result is not None
    assert not result.sufficient
    assert result.iterations <= ctx.config.coverage.max_iterations


def chain_seed(store: Store) -> CandidateSet:
    rows = store.query(
        "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?"
        " ORDER BY c.start_byte",
        ("pkg/a.py",),
    )
    loaded = load_candidates(store, [as_int(rows[0][0])])
    return CandidateSet.of(
        cand.with_scores(replace(cand.scores, gate=0.9, final=0.9)) for cand in loaded.values()
    )


def test_coverage_without_models_is_identity(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "parse")
    cands = CandidateSet.of([])
    assert CoverageStage().run(ctx, cands) is cands
    assert ctx.coverage.value is None


def test_coverage_skip_degrades_and_returns_pack_unchanged(store: Store) -> None:
    seed(store)
    ctx = make_ctx(store, "parse", models=fake_slots(FakeDecision(down=True)))
    cands = CandidateSet.of([])
    assert CoverageStage().run(ctx, cands) is cands
    assert stages_of(ctx.envelope.degraded) == {"coverage"}


def test_stage_registry_orders_by_slot_whatever_the_registration_order(store: Store) -> None:
    class Named:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, ctx: object, cands: CandidateSet) -> CandidateSet:
            return cands

    registry = StageRegistry()
    for slot in reversed(STAGE_SLOTS):
        registry.register(Named(slot.name))
    assert [slot.name for slot, _ in registry] == [slot.name for slot in STAGE_SLOTS]
    assert [slot.name for slot in STAGE_SLOTS][-2:] == ["coverage", "assemble"]


def test_coverage_toggle_disables_the_stage(store: Store) -> None:
    seed(store)
    config = replace(Config(), pipeline=replace(Config().pipeline, coverage=False))
    result = search(
        store, config, "parse the config text", SearchOptions(models=fake_slots(FakeDecision()))
    )
    assert "coverage" not in [t.stage for t in result.timings]
