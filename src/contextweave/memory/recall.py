import os
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contextweave.core.errors import DegradedEntry, Envelope, ErrorEntry
from contextweave.memory.embedder import EMBED_STAGE, TextEmbedder
from contextweave.memory.store import (
    MemoryStore,
    MemoryStoreError,
    fts_match_query,
    project_memory_dbs,
)
from contextweave.memory.wire import (
    RECALL_KINDS,
    RecallHit,
    RecallKind,
    RecallMode,
    RecallResult,
    RecallScope,
)
from contextweave.models.providers import EmbedKind
from contextweave.models.slots import Done

RRF_K = 60
DEFAULT_LIMIT = 10
TIMELINE_LIMIT = 200
MAX_COSINE_DISTANCE = 0.42
STAGE = "recall"


@dataclass(frozen=True, slots=True)
class QueryVector:
    model_id: str
    values: list[float]


def _embed_query(
    topic: str, embedder: TextEmbedder | None
) -> tuple[QueryVector | None, list[DegradedEntry]]:
    if embedder is None or not topic.strip():
        return None, []
    outcome = embedder.embed([topic], EmbedKind.QUERY)
    if not isinstance(outcome, Done):
        return None, [outcome.degraded]
    if len(outcome.value) != 1 or not outcome.value[0]:
        return None, [DegradedEntry(stage=EMBED_STAGE, reason="bad embed response")]
    return QueryVector(embedder.model_id, outcome.value[0]), []


def search_store(
    store: MemoryStore,
    topic: str,
    kinds: Sequence[RecallKind],
    query: QueryVector | None,
    limit: int,
) -> list[RecallHit]:
    """Hybrid FTS5 + vector search over one project, fused with RRF."""
    pool = limit * 4
    rankings: list[list[tuple[RecallKind, int]]] = []
    match = fts_match_query(topic)
    if match is not None:
        for kind in kinds:
            ids = store.fts_ranked(kind, match, pool)
            if ids:
                rankings.append([(kind, row_id) for row_id in ids])
    if query is not None:
        matches = store.vector_ranked(
            query.model_id, query.values, kinds, pool, MAX_COSINE_DISTANCE
        )
        if matches:
            rankings.append([(m.kind, m.row_id) for m in matches])
    scores: defaultdict[tuple[RecallKind, int], float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] += 1.0 / (RRF_K + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))[:limit]
    loaded = {kind: store.hits(kind, [i for k, i in ordered if k == kind]) for kind in kinds}
    hits: list[RecallHit] = []
    for key in ordered:
        hit = loaded[key[0]].get(key[1])
        if hit is not None:
            hit["score"] = round(scores[key], 6)
            hits.append(hit)
    return hits


def _hit_order(hit: RecallHit) -> tuple[float, str, str, int]:
    return (-hit["score"], hit["project"], hit["kind"], hit["id"])


def _unreadable(key: str, exc: Exception) -> ErrorEntry:
    return ErrorEntry(
        code="project_unreadable",
        message=str(exc) or type(exc).__name__,
        stage=STAGE,
        detail={"project": key},
    )


def _global_hits(
    store: MemoryStore,
    topic: str,
    kinds: Sequence[RecallKind],
    query: QueryVector | None,
    limit: int,
    env: Mapping[str, str],
    envelope: Envelope,
) -> list[RecallHit]:
    hits: list[RecallHit] = []
    for key, path in project_memory_dbs(env):
        if key == store.project_key:
            hits.extend(search_store(store, topic, kinds, query, limit))
            continue
        try:
            other = MemoryStore.open(path, key)
        except (MemoryStoreError, sqlite3.Error) as exc:
            envelope.add_error(_unreadable(key, exc))
            continue
        try:
            hits.extend(search_store(other, topic, kinds, query, limit))
        except (MemoryStoreError, sqlite3.Error) as exc:
            envelope.add_error(_unreadable(key, exc))
        finally:
            other.close()
    hits.sort(key=_hit_order)
    return hits[:limit]


def _empty(mode: RecallMode, scope: RecallScope, envelope: Envelope) -> RecallResult:
    return RecallResult(
        mode=mode,
        scope=scope,
        hits=[],
        timeline=None,
        event=None,
        errors=envelope.errors,
        degraded=envelope.degraded,
        truncated=False,
    )


def _arg_error(code: str, message: str) -> ErrorEntry:
    return ErrorEntry(code=code, message=message, stage=STAGE)


def recall(
    store: MemoryStore,
    topic: str = "",
    *,
    kinds: Sequence[str] | None = None,
    scope: str = "project",
    session_id: str | None = None,
    event_id: int | None = None,
    embedder: TextEmbedder | None = None,
    env: Mapping[str, str] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> RecallResult:
    """Resolve event_id, else session_id timeline, else topic search; project scoped unless scope='global'."""
    envelope = Envelope()
    mode: RecallMode = (
        "event" if event_id is not None else "timeline" if session_id is not None else "topic"
    )
    if scope not in ("project", "global"):
        envelope.add_error(_arg_error("invalid_arg", f"unknown scope: {scope!r}"))
        return _empty(mode, "project", envelope)
    resolved_scope: RecallScope = "global" if scope == "global" else "project"
    result = _empty(mode, resolved_scope, envelope)
    if resolved_scope == "global" and mode != "topic":
        envelope.add_error(
            _arg_error(
                "invalid_arg", "event_id and session_id are project scoped; use scope='project'"
            )
        )
        return result
    with envelope.guard(STAGE):
        if event_id is not None:
            result["event"] = store.event(event_id)
            if result["event"] is None:
                envelope.add_error(_arg_error("not_found", f"no event with id={event_id}"))
        elif session_id is not None:
            result["timeline"] = store.timeline(session_id, TIMELINE_LIMIT)
            if result["timeline"] is None:
                envelope.add_error(_arg_error("not_found", f"no session {session_id}"))
        else:
            result["hits"] = _topic_hits(
                store, topic, kinds, resolved_scope, embedder, env, limit, envelope
            )
    return result


def _topic_hits(
    store: MemoryStore,
    topic: str,
    kinds: Sequence[str] | None,
    scope: RecallScope,
    embedder: TextEmbedder | None,
    env: Mapping[str, str] | None,
    limit: int,
    envelope: Envelope,
) -> list[RecallHit]:
    selected = _select_kinds(kinds, envelope)
    if selected is None:
        return []
    if not topic.strip():
        envelope.add_error(_arg_error("missing_arg", "missing required argument: topic"))
        return []
    query, degraded = _embed_query(topic, embedder)
    for entry in degraded:
        envelope.add_degraded(entry)
    if scope == "global":
        return _global_hits(
            store, topic, selected, query, limit, os.environ if env is None else env, envelope
        )
    return search_store(store, topic, selected, query, limit)


def _select_kinds(kinds: Sequence[str] | None, envelope: Envelope) -> list[RecallKind] | None:
    if kinds is None:
        return list(RECALL_KINDS)
    selected: list[RecallKind] = []
    for name in kinds:
        match name:
            case "decision" | "code_area" | "note" | "turn_summary":
                if name not in selected:
                    selected.append(name)
            case _:
                envelope.add_error(_arg_error("invalid_arg", f"unknown kind: {name!r}"))
                return None
    return selected or list(RECALL_KINDS)
