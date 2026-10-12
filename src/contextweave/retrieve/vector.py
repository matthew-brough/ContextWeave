import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import replace

import sqlite_vec

from contextweave.core.errors import DegradedEntry
from contextweave.index.embed import embed_model_id
from contextweave.index.store import SqlValue, Store
from contextweave.index.vectors import (
    CODE,
    KINDS,
    Space,
    find_space,
    part_spans,
    spaces,
    vec_available,
)
from contextweave.models.providers import EmbedKind
from contextweave.models.slots import Done
from contextweave.retrieve.candidates import Candidate, Ranks
from contextweave.retrieve.context import QueryContext
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.scope import scope_clause

STAGE_NAME = "retrieve"
SKIP_VEC = "skip_vec"
VEC_REBUILDING = "vec_rebuilding"
VEC_UNAVAILABLE = "vec_unavailable"
MAX_KNN = 4096
OVERFETCH = 4


def knn(
    store: Store, space: Space, vector: Sequence[float], *, scope: str | None, limit: int
) -> list[tuple[int, float, int]]:
    """Nearest chunks by their closest part's cosine distance within the path scope, with that part's id.

    Parts are over-fetched and the fetch doubles while too few distinct chunks come back; vectors
    lacking metadata drop out.
    """
    if limit <= 0:
        return []
    clause, params = scope_clause("f.path", scope)
    sql = (
        "SELECT nearest.rowid, nearest.distance, p.chunk_id,"
        f" coalesce(m.chunk_id IS NOT NULL AND {clause}, 0)"
        f" FROM (SELECT rowid, distance FROM {space.name} WHERE embedding MATCH ? AND k = ?) nearest"
        " LEFT JOIN vector_parts p ON p.id = nearest.rowid AND p.space_name = ?"
        " LEFT JOIN vectors_meta m ON m.chunk_id = p.chunk_id AND m.space_name = p.space_name"
        " LEFT JOIN chunks c ON c.id = m.chunk_id LEFT JOIN files f ON f.id = c.file_id"
        " ORDER BY nearest.distance, f.path, c.start_byte, c.end_byte, c.id"
    )
    query = sqlite_vec.serialize_float32(list(vector))
    fetch = min(limit * OVERFETCH, MAX_KNN)
    while True:
        bound: list[SqlValue] = [*params, query, fetch, space.name]
        rows = store.query(sql, bound)
        nearest: dict[int, tuple[float, int]] = {}
        for row in rows:
            if not row[3]:
                continue
            chunk_id = int(str(row[2]))
            if chunk_id not in nearest:
                nearest[chunk_id] = (float(str(row[1])), int(str(row[0])))
        if len(nearest) >= limit or len(rows) < fetch or fetch >= MAX_KNN:
            break
        fetch = min(fetch * 2, MAX_KNN)
    return [(chunk_id, distance, vector) for chunk_id, (distance, vector) in nearest.items()][
        :limit
    ]


def _sections(store: Store, best: Mapping[int, tuple[float, int]]) -> dict[int, tuple[int, int]]:
    spans = part_spans(store, [vector for _, vector in best.values()])
    return {chunk_id: spans[vector] for chunk_id, (_, vector) in best.items() if vector in spans}


def _skip(ctx: QueryContext, reason: str, device: str | None = None) -> tuple[Candidate, ...]:
    entry = DegradedEntry(stage=STAGE_NAME, reason=reason)
    if device is not None:
        entry["device_tried"] = device
    ctx.envelope.add_degraded(entry)
    return ()


def _live_spaces(ctx: QueryContext) -> tuple[Space, ...] | str:
    embed = ctx.config.models.embed
    model_id = embed_model_id(embed)
    code = find_space(ctx.store, CODE, model_id, embed.dim)
    if code is None:
        known = any(space.kind == CODE for space in spaces(ctx.store))
        return VEC_REBUILDING if known else VEC_UNAVAILABLE
    if code.status != "live":
        return VEC_REBUILDING
    found = [code]
    for kind in KINDS:
        other = find_space(ctx.store, kind, model_id, embed.dim)
        if kind != CODE and other is not None and other.status == "live":
            found.append(other)
    return tuple(found)


def vector_search(ctx: QueryContext, scope: str | None, limit: int) -> tuple[Candidate, ...]:
    """Embed the query and rank chunks by nearest vector over the live code and annot spaces."""
    if ctx.models is None:
        return _skip(ctx, VEC_UNAVAILABLE) if ctx.mode == "semantic" else ()
    if not vec_available(ctx.store):
        return _skip(ctx, VEC_UNAVAILABLE)
    live = _live_spaces(ctx)
    if isinstance(live, str):
        return _skip(ctx, live)
    outcome = ctx.models.embed([ctx.query], kind=EmbedKind.QUERY, stage=STAGE_NAME)
    if not isinstance(outcome, Done):
        entry = outcome.degraded
        return _skip(ctx, f"{SKIP_VEC}: {entry['reason']}", entry.get("device_tried"))
    if len(outcome.value) != 1:
        return _skip(ctx, SKIP_VEC)
    vector = outcome.value[0]
    best: dict[int, tuple[float, int]] = {}
    try:
        for space in live:
            if len(vector) != space.dim:
                return _skip(ctx, f"{SKIP_VEC}: dim_mismatch")
            for chunk_id, distance, part in knn(ctx.store, space, vector, scope=scope, limit=limit):
                held = best.get(chunk_id)
                if held is None or distance < held[0]:
                    best[chunk_id] = (distance, part)
        sections = _sections(ctx.store, best)
    except sqlite3.Error:
        return _skip(ctx, f"{SKIP_VEC}: vec_query_failed")
    loaded = load_candidates(ctx.store, tuple(best))
    ordered = sorted(
        (loaded[chunk_id] for chunk_id in best if chunk_id in loaded),
        key=lambda cand: (best[cand.chunk_id][0], *cand.order_key),
    )[:limit]
    return tuple(
        replace(cand, ranks=Ranks(semantic=rank), section=sections.get(cand.chunk_id))
        for rank, cand in enumerate(ordered, start=1)
    )
