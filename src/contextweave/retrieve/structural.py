from contextweave.index.store import SqlValue, Store
from contextweave.retrieve.candidates import Candidate, Ranks
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int, as_str
from contextweave.retrieve.scope import scope_clause
from contextweave.retrieve.terms import MIN_PREFIX_LENGTH, STOPWORDS, words

MAX_STRUCTURAL_TERMS = 8
EXACT, PREFIX, PATH = 0, 1, 2

_SYMBOL_JOIN = (
    "SELECT cs.chunk_id, f.path, c.start_byte, c.end_byte FROM symbols s"
    " JOIN chunk_symbols cs ON cs.symbol_id = s.id JOIN chunks c ON c.id = cs.chunk_id"
    " JOIN files f ON f.id = s.file_id"
)
_FILE_JOIN = (
    "SELECT c.id, f.path, c.start_byte, c.end_byte FROM chunks c JOIN files f ON f.id = c.file_id"
)


def structural_terms(query: str) -> tuple[str, ...]:
    kept = [word for word in words(query) if word.lower() not in STOPWORDS]
    return tuple(dict.fromkeys(kept or words(query)))[:MAX_STRUCTURAL_TERMS]


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def structural_hits(store: Store, query: str, *, scope: str | None, limit: int) -> tuple[int, ...]:
    """Chunk ids best-first: exact symbol name, then name prefix, then path match."""
    if limit <= 0:
        return ()
    clause, scope_params = scope_clause("f.path", scope)
    best: dict[int, tuple[int, str, int, int]] = {}

    def collect(tier: int, sql: str, params: list[SqlValue]) -> None:
        for row in store.query(f"{sql} AND {clause}", [*params, *scope_params]):
            chunk_id = as_int(row[0])
            key = (tier, as_str(row[1]), as_int(row[2]), as_int(row[3]))
            if chunk_id not in best or key < best[chunk_id]:
                best[chunk_id] = key

    for term in structural_terms(query):
        collect(EXACT, f"{_SYMBOL_JOIN} WHERE (s.name = ? OR s.qualified_name = ?)", [term, term])
        if len(term) >= MIN_PREFIX_LENGTH:
            collect(
                PREFIX,
                f"{_SYMBOL_JOIN} WHERE substr(lower(s.name), 1, ?) = ?",
                [len(term), term.lower()],
            )
        escaped = _like_escape(term)
        collect(
            PATH,
            f"{_FILE_JOIN} WHERE (f.path = ? OR f.path LIKE ? ESCAPE '\\' OR f.path LIKE ? ESCAPE '\\'"
            " OR f.path LIKE ? ESCAPE '\\')",
            [term, f"%/{escaped}", f"%/{escaped}.%", f"{escaped}.%"],
        )
    ordered = sorted(best, key=lambda chunk_id: (*best[chunk_id], chunk_id))
    return tuple(ordered[:limit])


def structural_search(
    store: Store, query: str, *, scope: str | None, limit: int
) -> tuple[Candidate, ...]:
    """Candidates carrying structural ranks 1..N."""
    ids = structural_hits(store, query, scope=scope, limit=limit)
    loaded = load_candidates(store, ids)
    return tuple(
        loaded[chunk_id].with_ranks(Ranks(structural=rank))
        for rank, chunk_id in enumerate(ids, start=1)
        if chunk_id in loaded
    )
