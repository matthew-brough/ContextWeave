from memory_support import FakeEmbedder

from contextweave.memory.capture import (
    backfill_vectors,
    end_session,
    enqueue_turn_summary,
    record_prompt,
    record_tool_event,
    remember,
    start_session,
    store_extractive_summary,
    store_synthesized_summary,
)
from contextweave.memory.recall import recall
from contextweave.memory.store import MemoryStore
from contextweave.models.providers import EmbedKind


def count(store: MemoryStore, table: str) -> int:
    return store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_remember_missing_args_returns_error_and_writes_nothing(
    store_a: MemoryStore, embedder: FakeEmbedder
) -> None:
    cases = [
        ("decision", "use sqlite", None, None, "reason"),
        ("decision", "use sqlite", "   ", None, "reason"),
        ("code_area", "the walker", None, None, "path"),
        ("note", "   ", None, None, "text"),
    ]
    for kind, text, reason, path, missing in cases:
        result = remember(store_a, kind, text, reason, path, embedder=embedder)
        assert not result["stored"]
        assert result["id"] is None
        assert result["errors"][0]["code"] == "missing_arg"
        assert result["errors"][0].get("detail") == {"missing": [missing]}
    for table in ("decisions", "code_areas", "notes", "vectors_meta"):
        assert count(store_a, table) == 0
    assert embedder.calls == []


def test_remember_unknown_kind_rejected(store_a: MemoryStore) -> None:
    result = remember(store_a, "fact", "x")
    assert result["errors"][0]["code"] == "invalid_arg"
    assert count(store_a, "notes") == 0


def test_round_trip_decision_code_area_note(store_a: MemoryStore, embedder: FakeEmbedder) -> None:
    decision = remember(store_a, "decision", "use sqlite", "single file storage", embedder=embedder)
    area = remember(store_a, "code_area", "walks the tree", path="src/walk.py", embedder=embedder)
    note = remember(store_a, "note", "remember the cache invalidation", embedder=embedder)
    assert decision["stored"] and area["stored"] and note["stored"]
    hits = recall(store_a, "sqlite", embedder=embedder)["hits"]
    assert [(h["kind"], h["id"]) for h in hits] == [("decision", decision["id"])]
    assert hits[0].get("reason") == "single file storage"
    assert hits[0]["text"] == "use sqlite"
    area_hit = recall(store_a, "walks", kinds=["code_area"])["hits"][0]
    assert area_hit.get("path") == "src/walk.py"
    assert area_hit["text"] == "walks the tree"
    note_hit = recall(store_a, "invalidation", kinds=["note"])["hits"][0]
    assert note_hit["text"] == "remember the cache invalidation"
    assert note_hit["session_id"] is None


def test_remember_records_session_id(store_a: MemoryStore) -> None:
    remember(store_a, "note", "tagged note", session_id="s1")
    assert recall(store_a, "tagged")["hits"][0]["session_id"] == "s1"


def test_session_prompt_event_flow(store_a: MemoryStore) -> None:
    start_session(store_a, "s1")
    first = record_prompt(store_a, "s1", "do a thing")
    second = record_prompt(store_a, "s1", "do another")
    assert (first, second) == (1, 2)
    event_id = record_tool_event(store_a, "s1", 1, "Read", summary="read file", raw_input="in")
    assert event_id == 1
    assert end_session(store_a, "s1", "completed", "stop")
    assert not end_session(store_a, "missing")
    status = store_a.conn.execute("SELECT status, prompt_count FROM sessions").fetchone()
    assert tuple(status) == ("completed", 2)


def test_prompt_on_unknown_session_creates_it(store_a: MemoryStore) -> None:
    assert record_prompt(store_a, "fresh", "hi") == 1
    assert count(store_a, "sessions") == 1


def test_extractive_summary_picks_central_sentences_in_order(
    store_a: MemoryStore, embedder: FakeEmbedder
) -> None:
    text = (
        "The car was slow. The automobile needed a vehicle check. "
        "Weather was fine. A car vehicle automobile test ran. Lunch happened."
    )
    row_id, degraded = store_extractive_summary(store_a, "s1", 1, text, embedder)
    assert degraded == []
    turn = recall(store_a, session_id="s1")["timeline"]
    assert turn is not None
    summary = turn["turns"][0]
    assert summary["tier"] == "extractive"
    assert not summary["synthesized"]
    assert row_id == 1
    assert summary["summary"] == (
        "The car was slow. The automobile needed a vehicle check. "
        "A car vehicle automobile test ran."
    )


def test_extractive_short_text_verbatim(store_a: MemoryStore, embedder: FakeEmbedder) -> None:
    store_extractive_summary(store_a, "s1", 1, "One. Two.", embedder)
    assert embedder.calls == [(EmbedKind.MEMORY, ("One. Two.",))]
    timeline = recall(store_a, session_id="s1")["timeline"]
    assert timeline is not None
    assert timeline["turns"][0]["summary"] == "One. Two."


def test_extractive_falls_back_to_truncation_when_embed_skipped(store_a: MemoryStore) -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(30))
    skipped = FakeEmbedder(skip=True)
    _, degraded = store_extractive_summary(store_a, "s1", 1, text, skipped)
    assert degraded[0]["stage"] == "memory_embed"
    timeline = recall(store_a, session_id="s1")["timeline"]
    assert timeline is not None
    turn = timeline["turns"][0]
    assert turn["tier"] == "truncated"
    assert len(turn["summary"]) <= 200
    assert not turn["synthesized"]


def test_synthesized_summary_flagged_with_model_and_prompt_version(
    store_a: MemoryStore, embedder: FakeEmbedder
) -> None:
    enqueue_turn_summary(store_a, "s1", 1)
    assert [p.prompt_number for p in store_a.pending_summaries()] == [1]
    store_synthesized_summary(
        store_a,
        "s1",
        1,
        "Explained the car design.",
        model="qwen",
        prompt_version=3,
        embedder=embedder,
    )
    assert store_a.pending_summaries() == []
    hit = recall(store_a, "car", kinds=["turn_summary"], embedder=embedder)["hits"][0]
    assert hit.get("synthesized") is True
    assert hit.get("tier") == "synthesized"
    assert hit.get("model") == "qwen"
    assert hit.get("prompt_version") == 3
    assert hit["session_id"] == "s1"
    assert hit.get("prompt_number") == 1


def test_summary_replaces_same_turn(store_a: MemoryStore) -> None:
    store_synthesized_summary(store_a, "s1", 1, "old text", model="m", prompt_version=1)
    store_synthesized_summary(store_a, "s1", 1, "new text", model="m", prompt_version=2)
    assert count(store_a, "turn_summaries") == 1
    assert recall(store_a, "old")["hits"] == []
    assert recall(store_a, "new")["hits"][0].get("prompt_version") == 2


def test_vectors_keyed_by_model_and_never_mixed(store_a: MemoryStore) -> None:
    old = FakeEmbedder(model_id="model-old")
    new = FakeEmbedder(model_id="model-new")
    remember(store_a, "note", "the car is fast", embedder=old)
    assert store_a.vector_counts() == {("model-old", 6): 1}
    query_old = recall(store_a, "automobile", embedder=old)["hits"]
    assert len(query_old) == 1
    assert recall(store_a, "automobile", embedder=new)["hits"] == []
    written, degraded = backfill_vectors(store_a, new)
    assert (written, degraded) == (1, [])
    assert store_a.vector_counts() == {("model-new", 6): 1}
    assert len(recall(store_a, "automobile", embedder=new)["hits"]) == 1
    assert recall(store_a, "automobile", embedder=old)["hits"] == []


def test_fts_only_when_embed_skipped_at_write(store_a: MemoryStore) -> None:
    skipped = FakeEmbedder(skip=True)
    result = remember(store_a, "decision", "pick the car", "speed matters", embedder=skipped)
    assert result["stored"]
    assert result["degraded"][0]["reason"] == "embed down"
    assert store_a.vector_counts() == {}
    hits = recall(store_a, "pick")["hits"]
    assert [h["id"] for h in hits] == [result["id"]]
