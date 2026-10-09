from pathlib import Path

from conftest import KEY_A, KEY_B
from memory_support import FakeEmbedder

from contextweave.core.config import EmbedConfig, ModelsConfig
from contextweave.memory.capture import (
    record_prompt,
    record_tool_event,
    remember,
    store_synthesized_summary,
)
from contextweave.memory.embedder import SlotEmbedder
from contextweave.memory.recall import recall
from contextweave.memory.savings import full_file_tokens, record_savings, savings_summary
from contextweave.memory.store import MemoryStore, memory_db_path
from contextweave.models.providers import EmbedKind
from contextweave.models.slots import ModelSlots, Skipped


def test_project_isolation_same_db_different_keys(state_env: dict[str, str]) -> None:
    path = memory_db_path(KEY_A, state_env)
    a = MemoryStore.open(path, KEY_A)
    b = MemoryStore.open(path, KEY_B)
    remember(a, "decision", "use the secret plan", "because")
    assert len(recall(a, "secret")["hits"]) == 1
    assert recall(b, "secret")["hits"] == []
    assert recall(b, session_id="x")["errors"][0]["code"] == "not_found"
    a.close()
    b.close()


def test_project_isolation_separate_dbs_and_global_scope(
    state_env: dict[str, str], store_a: MemoryStore, store_b: MemoryStore, embedder: FakeEmbedder
) -> None:
    remember(store_a, "note", "alpha database note", embedder=embedder)
    remember(store_b, "note", "beta database note", embedder=embedder)
    only_a = recall(store_a, "database", embedder=embedder)
    assert [h["project"] for h in only_a["hits"]] == [KEY_A]
    assert [h["project"] for h in recall(store_b, "database")["hits"]] == [KEY_B]
    global_hits = recall(store_a, "database", scope="global", embedder=embedder, env=state_env)
    assert sorted(h["project"] for h in global_hits["hits"]) == [KEY_A, KEY_B]
    assert global_hits["scope"] == "global"


def test_global_reports_unreadable_project_and_continues(
    state_env: dict[str, str], store_a: MemoryStore
) -> None:
    remember(store_a, "note", "findable thing")
    bad = memory_db_path("cccccccccccc", state_env)
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not a database at all, definitely not sqlite" * 10)
    result = recall(store_a, "findable", scope="global", env=state_env)
    assert [h["project"] for h in result["hits"]] == [KEY_A]
    assert result["errors"][0]["code"] == "project_unreadable"
    assert result["errors"][0].get("detail") == {"project": "cccccccccccc"}


def test_global_rejects_event_and_session_ids(
    store_a: MemoryStore, state_env: dict[str, str]
) -> None:
    result = recall(store_a, scope="global", event_id=1, env=state_env)
    assert result["errors"][0]["code"] == "invalid_arg"
    assert result["event"] is None


def test_invalid_scope_kind_and_missing_topic(store_a: MemoryStore) -> None:
    assert recall(store_a, "x", scope="everywhere")["errors"][0]["code"] == "invalid_arg"
    assert recall(store_a, "x", kinds=["bogus"])["errors"][0]["code"] == "invalid_arg"
    assert recall(store_a, "  ")["errors"][0]["code"] == "missing_arg"


def test_precedence_event_over_session_over_topic(store_a: MemoryStore) -> None:
    remember(store_a, "note", "banana topic")
    store_synthesized_summary(store_a, "s1", 1, "turn summary", model="m", prompt_version=1)
    event_id = record_tool_event(
        store_a, "s1", 1, "Bash", summary="ran", raw_input="in", raw_output="out"
    )

    by_event = recall(store_a, "banana", session_id="s1", event_id=event_id)
    assert by_event["mode"] == "event"
    assert by_event["event"] is not None
    assert by_event["timeline"] is None and by_event["hits"] == []

    by_session = recall(store_a, "banana", session_id="s1")
    assert by_session["mode"] == "timeline"
    assert by_session["timeline"] is not None
    assert by_session["hits"] == []

    by_topic = recall(store_a, "banana")
    assert by_topic["mode"] == "topic"
    assert len(by_topic["hits"]) == 1


def test_timeline_lists_turns_and_event_ids(store_a: MemoryStore) -> None:
    record_prompt(store_a, "s1", "first")
    record_prompt(store_a, "s1", "second")
    e1 = record_tool_event(store_a, "s1", 1, "Read", summary="read a")
    e2 = record_tool_event(store_a, "s1", 1, "Edit", summary=None)
    store_synthesized_summary(store_a, "s1", 2, "second turn", model="m", prompt_version=1)
    store_synthesized_summary(store_a, "s1", 1, "first turn", model="m", prompt_version=1)
    timeline = recall(store_a, session_id="s1")["timeline"]
    assert timeline is not None
    assert [t["prompt_number"] for t in timeline["turns"]] == [1, 2]
    assert [e["id"] for e in timeline["turns"][0]["events"]] == [e1, e2]
    assert timeline["turns"][1]["events"] == []
    assert timeline["prompt_count"] == 2
    assert timeline["project"] == KEY_A


def test_unknown_event_and_session_not_found(store_a: MemoryStore) -> None:
    assert recall(store_a, event_id=42)["errors"][0]["code"] == "not_found"
    assert recall(store_a, session_id="nope")["errors"][0]["code"] == "not_found"


def test_event_payload_capped_at_4kb(store_a: MemoryStore) -> None:
    big = "x" * 10_000
    multi = "é" * 3_000
    event_id = record_tool_event(store_a, "s1", 1, "Read", raw_input=big, raw_output=multi)
    event = recall(store_a, event_id=event_id)["event"]
    assert event is not None
    assert event["raw_input"] == "x" * 4096
    assert event["raw_output"] is not None
    assert len(event["raw_output"].encode()) <= 4096
    assert event["truncated"] is True
    assert event["payload"] == "present"
    small = record_tool_event(store_a, "s1", 1, "Read", raw_input="tiny", raw_output="ok")
    detail = recall(store_a, event_id=small)["event"]
    assert detail is not None
    assert (detail["raw_input"], detail["raw_output"], detail["truncated"]) == ("tiny", "ok", False)


def test_event_without_payload_and_aged_out(store_a: MemoryStore) -> None:
    bare = record_tool_event(store_a, "s1", 1, "Glob")
    event = recall(store_a, event_id=bare)["event"]
    assert event is not None and event["payload"] == "none" and event["raw_input"] is None
    aged = record_tool_event(store_a, "s1", 1, "Read", raw_input="data", raw_output="more")
    with store_a.conn:
        store_a.conn.execute("UPDATE tool_event_payloads SET raw_input='', raw_output=NULL")
    detail = recall(store_a, event_id=aged)["event"]
    assert detail is not None and detail["payload"] == "aged_out"


def test_event_belongs_to_project(store_a: MemoryStore, store_b: MemoryStore) -> None:
    event_id = record_tool_event(store_a, "s1", 1, "Read", raw_input="a")
    assert recall(store_b, event_id=event_id)["event"] is None


def test_vector_finds_without_lexical_overlap(store_a: MemoryStore, embedder: FakeEmbedder) -> None:
    remember(store_a, "decision", "we picked sqlite", "storage in one file", embedder=embedder)
    remember(store_a, "note", "unrelated latency remark", embedder=embedder)
    with_vec = recall(store_a, "database", embedder=embedder)
    assert [h["kind"] for h in with_vec["hits"]] == ["decision"]
    assert recall(store_a, "database")["hits"] == []
    kinds = {kind for kind, _ in embedder.calls if kind is EmbedKind.QUERY}
    assert kinds == {EmbedKind.QUERY}


def test_fts_fallback_when_embed_skipped_at_query(
    store_a: MemoryStore, embedder: FakeEmbedder
) -> None:
    remember(store_a, "note", "cache warmup note", embedder=embedder)
    skipped = FakeEmbedder(skip=True)
    result = recall(store_a, "warmup", embedder=skipped)
    assert len(result["hits"]) == 1
    assert result["degraded"][0]["reason"] == "embed down"
    assert result["errors"] == []


def test_kinds_filter(store_a: MemoryStore) -> None:
    remember(store_a, "decision", "shared word", "reason shared")
    remember(store_a, "note", "shared word")
    both = recall(store_a, "shared")["hits"]
    assert {h["kind"] for h in both} == {"decision", "note"}
    only = recall(store_a, "shared", kinds=["note"])["hits"]
    assert [h["kind"] for h in only] == ["note"]


def test_deterministic_ordering_and_tie_break(store_a: MemoryStore) -> None:
    first = remember(store_a, "note", "twin text")["id"]
    second = remember(store_a, "note", "twin text")["id"]
    third = remember(store_a, "decision", "twin text", "twin text")["id"]
    runs = [recall(store_a, "twin") for _ in range(3)]
    ordered = [[(h["kind"], h["id"]) for h in run["hits"]] for run in runs]
    assert ordered[0] == ordered[1] == ordered[2]
    assert ordered[0] == [("decision", third), ("note", first), ("note", second)]


def test_rrf_scores_use_k60(store_a: MemoryStore) -> None:
    remember(store_a, "note", "solo entry")
    hit = recall(store_a, "solo")["hits"][0]
    assert hit["score"] == round(1 / 61, 6)


def test_hybrid_fusion_boosts_double_match(store_a: MemoryStore, embedder: FakeEmbedder) -> None:
    remember(store_a, "note", "car speed", embedder=embedder)
    remember(store_a, "note", "sqlite car notes", embedder=embedder)
    hits = recall(store_a, "car speed", embedder=embedder)["hits"]
    assert hits[0]["text"] == "car speed"
    assert hits[0]["score"] > hits[1]["score"]


def test_limit(store_a: MemoryStore) -> None:
    for i in range(5):
        remember(store_a, "note", f"limited item {i}")
    assert len(recall(store_a, "limited", limit=3)["hits"]) == 3


def test_savings_ledger(store_a: MemoryStore, store_b: MemoryStore, tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x" * 350)
    (tmp_path / "b.py").write_text("y" * 70)
    baseline = full_file_tokens(tmp_path, ["a.py", "b.py", "a.py", "gone.py"])
    assert baseline == 100 + 20
    record_savings(store_a, tokens_served=30, full_file_tokens=baseline, meta={"pack": "p1"})
    record_savings(store_a, tokens_served=10, full_file_tokens=80)
    record_savings(store_a, tokens_served=5, full_file_tokens=9, bucket="memory_recall")
    totals = savings_summary(store_a)
    assert totals["retrieval"] == {"baseline": 200, "served": 40, "saved": 160, "calls": 2}
    assert totals["memory_recall"]["calls"] == 1
    assert totals["progressive_disclosure"]["calls"] == 0
    assert savings_summary(store_b)["retrieval"]["calls"] == 0


def test_slot_embedder_reports_skip_for_none_provider() -> None:
    slots = ModelSlots.from_config(ModelsConfig(embed=EmbedConfig(provider="none")))
    outcome = SlotEmbedder(slots, "none").embed(["hello"], EmbedKind.QUERY)
    assert isinstance(outcome, Skipped)
    assert outcome.degraded["stage"] == "memory_embed"
