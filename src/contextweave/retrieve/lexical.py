import sqlite3

from contextweave.core.errors import StageError
from contextweave.index.store import Store
from contextweave.retrieve.candidates import Candidate, Ranks
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int
from contextweave.retrieve.scope import scope_clause
from contextweave.retrieve.terms import match_expression

COLUMN_WEIGHTS = (10.0, 10.0, 8.0, 4.0, 1.0, 4.0, 0.5)


def lexical_hits(store: Store, query: str, *, scope: str | None, limit: int) -> tuple[int, ...]:
    """Chunk ids best-first by weighted BM25 over fts_code, ties by (path, start_byte)."""
    expression = match_expression(query)
    if expression is None or limit <= 0:
        return ()
    clause, params = scope_clause("fts_code.path", scope)
    weights = ", ".join(str(weight) for weight in COLUMN_WEIGHTS)
    sql = (
        f"SELECT c.id, bm25(fts_code, {weights}) AS score FROM fts_code"
        " JOIN chunks c ON c.id = fts_code.rowid"
        f" WHERE fts_code MATCH ? AND {clause}"
        " ORDER BY score, fts_code.path, c.start_byte, c.end_byte, c.id LIMIT ?"
    )
    try:
        rows = store.query(sql, [expression, *params, limit])
    except sqlite3.OperationalError as exc:
        raise StageError("bad_query", f"fts match rejected: {exc}") from exc
    return tuple(as_int(row[0]) for row in rows)


def lexical_search(
    store: Store, query: str, *, scope: str | None, limit: int
) -> tuple[Candidate, ...]:
    """Candidates carrying lexical ranks 1..N."""
    ids = lexical_hits(store, query, scope=scope, limit=limit)
    loaded = load_candidates(store, ids)
    return tuple(
        loaded[chunk_id].with_ranks(Ranks(lexical=rank))
        for rank, chunk_id in enumerate(ids, start=1)
        if chunk_id in loaded
    )
