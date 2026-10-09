from contextweave.core.errors import DegradedEntry, ErrorEntry
from contextweave.memory.embedder import EMBED_STAGE, TextEmbedder
from contextweave.memory.extractive import extractive_summary
from contextweave.memory.store import (
    MemoryStore,
    SessionStatus,
    Source,
    SummaryInput,
    vector_text,
)
from contextweave.memory.wire import RememberResult
from contextweave.models.providers import EmbedKind
from contextweave.models.slots import Done

type Vector = tuple[str, list[float]]


def start_session(store: MemoryStore, session_id: str) -> None:
    """Create the session if it does not exist."""
    store.ensure_session(session_id)


def end_session(
    store: MemoryStore,
    session_id: str,
    status: SessionStatus = "completed",
    exit_reason: str | None = None,
) -> bool:
    """Mark the session finished; False when it is unknown."""
    return store.end_session(session_id, status, exit_reason)


def record_prompt(store: MemoryStore, session_id: str, text: str) -> int:
    """Store a user prompt and return its turn number."""
    return store.add_prompt(session_id, text)


def record_tool_event(
    store: MemoryStore,
    session_id: str,
    prompt_number: int,
    tool_name: str,
    *,
    summary: str | None = None,
    raw_input: str | None = None,
    raw_output: str | None = None,
) -> int:
    """Store a tool event with its optional raw payload and return the event id."""
    return store.add_tool_event(
        session_id, prompt_number, tool_name, summary, raw_input, raw_output
    )


def enqueue_turn_summary(store: MemoryStore, session_id: str, prompt_number: int) -> None:
    """Queue a turn for async summarisation."""
    store.enqueue_summary(session_id, prompt_number)


def embed_for_store(
    store: MemoryStore, text: str, embedder: TextEmbedder | None
) -> tuple[Vector | None, list[DegradedEntry]]:
    """Embed text for storage; any failure yields no vector plus a degraded entry."""
    if embedder is None:
        return None, []
    if not store.vec_enabled:
        return None, [DegradedEntry(stage=EMBED_STAGE, reason="sqlite-vec unavailable")]
    outcome = embedder.embed([text], EmbedKind.MEMORY)
    if not isinstance(outcome, Done):
        return None, [outcome.degraded]
    if len(outcome.value) != 1 or not outcome.value[0]:
        return None, [DegradedEntry(stage=EMBED_STAGE, reason="bad embed response")]
    return (embedder.model_id, outcome.value[0]), []


def store_turn_summary(
    store: MemoryStore,
    session_id: str,
    prompt_number: int,
    data: SummaryInput,
    embedder: TextEmbedder | None = None,
) -> tuple[int, list[DegradedEntry]]:
    """Store a turn summary (any tier) with its vector when embedding is available."""
    vector, degraded = embed_for_store(store, data.summary, embedder)
    return store.put_turn_summary(session_id, prompt_number, data, vector), degraded


def store_synthesized_summary(
    store: MemoryStore,
    session_id: str,
    prompt_number: int,
    summary: str,
    *,
    model: str,
    prompt_version: int,
    embedder: TextEmbedder | None = None,
) -> tuple[int, list[DegradedEntry]]:
    """Store a model-written summary flagged synthesized with its model and prompt version."""
    data = SummaryInput(summary, "synthesized", model, prompt_version)
    return store_turn_summary(store, session_id, prompt_number, data, embedder)


def store_extractive_summary(
    store: MemoryStore,
    session_id: str,
    prompt_number: int,
    source_text: str,
    embedder: TextEmbedder | None = None,
) -> tuple[int, list[DegradedEntry]]:
    """Summarise extractively (truncation when embedding is unavailable) and store."""
    data, summary_degraded = extractive_summary(source_text, embedder)
    row_id, store_degraded = store_turn_summary(store, session_id, prompt_number, data, embedder)
    degraded = [summary_degraded] if summary_degraded else []
    return row_id, [*degraded, *store_degraded]


def _rejected(kind: str, code: str, message: str, missing: list[str]) -> RememberResult:
    entry = ErrorEntry(code=code, message=message, stage="remember")
    if missing:
        entry["detail"] = {"missing": list(missing)}
    return RememberResult(stored=False, kind=kind, id=None, errors=[entry], degraded=[])


def remember(
    store: MemoryStore,
    kind: str,
    text: str,
    reason: str | None = None,
    path: str | None = None,
    *,
    session_id: str | None = None,
    embedder: TextEmbedder | None = None,
    source: Source = "manual",
) -> RememberResult:
    """Record a decision (needs reason), code_area (needs path) or note; validation failures write nothing."""
    body = text.strip()
    why = (reason or "").strip()
    where = (path or "").strip()
    missing = ["text"] if not body else []
    match kind:
        case "decision":
            if not why:
                missing.append("reason")
        case "code_area":
            if not where:
                missing.append("path")
        case "note":
            pass
        case _:
            return _rejected(kind, "invalid_arg", f"unknown kind: {kind!r}", [])
    if missing:
        message = f"missing required argument: {', '.join(missing)}"
        return _rejected(kind, "missing_arg", message, missing)
    match kind:
        case "decision":
            vector, degraded = embed_for_store(store, vector_text("decision", body, why), embedder)
            row_id = store.add_decision(body, why, session_id, source, vector)
        case "code_area":
            vector, degraded = embed_for_store(
                store, vector_text("code_area", body, where), embedder
            )
            row_id = store.add_code_area(where, body, session_id, source, vector)
        case _:
            vector, degraded = embed_for_store(store, body, embedder)
            row_id = store.add_note(body, session_id, source, vector)
    return RememberResult(stored=True, kind=kind, id=row_id, errors=[], degraded=degraded)


def backfill_vectors(store: MemoryStore, embedder: TextEmbedder) -> tuple[int, list[DegradedEntry]]:
    """Embed rows lacking a vector under the embedder's model, then evict other models' vectors."""
    if not store.vec_enabled:
        return 0, [DegradedEntry(stage=EMBED_STAGE, reason="sqlite-vec unavailable")]
    pending = store.rows_missing_vector(embedder.model_id)
    written = 0
    dim = 0
    for kind, row_id, text in pending:
        outcome = embedder.embed([text], EmbedKind.MEMORY)
        if not isinstance(outcome, Done):
            return written, [outcome.degraded]
        if len(outcome.value) != 1 or not outcome.value[0]:
            return written, [DegradedEntry(stage=EMBED_STAGE, reason="bad embed response")]
        store.put_vector(kind, row_id, text, embedder.model_id, outcome.value[0])
        dim = len(outcome.value[0])
        written += 1
    if not dim:
        dim = max(
            (d for (model, d) in store.vector_counts() if model == embedder.model_id), default=0
        )
    if dim:
        store.evict_other_models(embedder.model_id, dim)
    return written, []
