import sqlite3
from collections.abc import Sequence

import sqlite_vec

from contextweave.core.errors import DegradedEntry
from contextweave.index.embed import embed_model_id
from contextweave.index.store import SqlValue, Store
from contextweave.index.vectors import (
    CODE,
    KINDS,
    Space,
    find_space,
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
) -> list[tuple[int, float]]:
    """Nearest chunks by cosine distance within the path scope; vectors lacking metadata drop out."""
    if limit <= 0:
        return []
    clause, params = scope_clause("f.path", scope)
    sql = (
        "SELECT nearest.rowid, nearest.distance FROM"
        f" (SELECT rowid, distance FROM {space.name} WHERE embedding MATCH ? AND k = ?) nearest"
        " JOIN vectors_meta m ON m.chunk_id = nearest.rowid AND m.space_name = ?"
        " JOIN chunks c ON c.id = nearest.rowid JOIN files f ON f.id = c.file_id"
        f" WHERE {clause}"
        " ORDER BY nearest.distance, f.path, c.start_byte, c.end_byte, c.id LIMIT ?"
    )
    bound: list[SqlValue] = [
        sqlite_vec.serialize_float32(list(vector)),
        min(limit * OVERFETCH, MAX_KNN),
        space.name,
        *params,
        limit,
    ]
    return [(int(str(row[0])), float(str(row[1]))) for row in store.query(sql, bound)]


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
    best: dict[int, float] = {}
    try:
        for space in live:
            if len(vector) != space.dim:
                return _skip(ctx, f"{SKIP_VEC}: dim_mismatch")
            for chunk_id, distance in knn(ctx.store, space, vector, scope=scope, limit=limit):
                best[chunk_id] = min(distance, best.get(chunk_id, distance))
    except sqlite3.Error:
        return _skip(ctx, f"{SKIP_VEC}: vec_query_failed")
    loaded = load_candidates(ctx.store, tuple(best))
    ordered = sorted(
        (loaded[chunk_id] for chunk_id in best if chunk_id in loaded),
        key=lambda cand: (best[cand.chunk_id], *cand.order_key),
    )[:limit]
    return tuple(
        cand.with_ranks(Ranks(semantic=rank)) for rank, cand in enumerate(ordered, start=1)
    )
