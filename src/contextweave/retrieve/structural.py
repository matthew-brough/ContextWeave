from contextweave.index.store import SqlValue, Store
from contextweave.retrieve.candidates import Candidate, Ranks
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int, as_str
from contextweave.retrieve.scope import scope_clause
from contextweave.retrieve.terms import MIN_PREFIX_LENGTH, STOPWORDS, split_identifier, words

MAX_STRUCTURAL_TERMS = 8
MIN_SUBSTRING_LENGTH = 4
EXACT, PART, SUBSTRING, PATH = 0, 1, 2, 3

_SYMBOL_JOIN = (
    "SELECT cs.chunk_id, f.path, c.start_byte, c.end_byte, s.name, s.qualified_name FROM symbols s"
    " JOIN chunk_symbols cs ON cs.symbol_id = s.id JOIN chunks c ON c.id = cs.chunk_id"
    " JOIN files f ON f.id = s.file_id"
)
_FILE_JOIN = (
    "SELECT c.id, f.path, c.start_byte, c.end_byte FROM chunks c JOIN files f ON f.id = c.file_id"
)

type HitKey = tuple[int, int, str, int, int]


def structural_terms(query: str) -> tuple[str, ...]:
    kept = [word for word in words(query) if word.lower() not in STOPWORDS]
    return tuple(dict.fromkeys(kept or words(query)))[:MAX_STRUCTURAL_TERMS]


def stem(term: str) -> str:
    """A lowercased term with a plural suffix dropped, so `airspaces` finds `Airspace`."""
    lowered = term.lower()
    if len(lowered) > MIN_SUBSTRING_LENGTH and lowered.endswith("ies"):
        return lowered[:-3] + "y"
    if len(lowered) > MIN_SUBSTRING_LENGTH and lowered.endswith("s") and not lowered.endswith("ss"):
        return lowered[:-1]
    return lowered


def name_tier(term: str, stemmed: str, name: str, qualified_name: str) -> int | None:
    """How a query term names a symbol: whole name, a camel/snake part prefix, a substring, or not at all."""
    if term in (name, qualified_name) or stemmed == name.lower():
        return EXACT
    if len(stemmed) >= MIN_PREFIX_LENGTH and any(
        part.startswith(stemmed) for part in split_identifier(name)
    ):
        return PART
    if len(stemmed) >= MIN_SUBSTRING_LENGTH and stemmed in name.lower():
        return SUBSTRING
    return None


def _like_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def structural_hits(store: Store, query: str, *, scope: str | None, limit: int) -> tuple[int, ...]:
    """Chunk ids best-first: most distinct query terms naming a symbol, then exact, part, substring, path."""
    if limit <= 0:
        return ()
    terms = structural_terms(query)
    if not terms:
        return ()
    clause, scope_params = scope_clause("f.path", scope)
    stems = [stem(term) for term in terms]
    best: dict[int, HitKey] = {}

    def keep(chunk_id: int, key: HitKey) -> None:
        if chunk_id not in best or key < best[chunk_id]:
            best[chunk_id] = key

    probes = [
        (term, needle)
        for term, needle in zip(terms, stems, strict=True)
        if len(needle) >= MIN_PREFIX_LENGTH
    ]
    exact = [value for term in terms for value in (term, term)]
    where = " OR ".join(["s.name = ? OR s.qualified_name = ?" for _ in terms])
    where += "".join(" OR instr(lower(s.name), ?) > 0" for _ in probes)
    params: list[SqlValue] = [*exact, *(needle for _, needle in probes)]
    for row in store.query(
        f"{_SYMBOL_JOIN} WHERE ({where}) AND {clause}", [*params, *scope_params]
    ):
        name, qualified_name = as_str(row[4]), as_str(row[5])
        tiers = [
            tier
            for term, needle in zip(terms, stems, strict=True)
            if (tier := name_tier(term, needle, name, qualified_name)) is not None
        ]
        if tiers:
            location = (as_str(row[1]), as_int(row[2]), as_int(row[3]))
            keep(as_int(row[0]), (-len(tiers), min(tiers), *location))
    for term in terms:
        escaped = _like_escape(term)
        rows = store.query(
            f"{_FILE_JOIN} WHERE (f.path = ? OR f.path LIKE ? ESCAPE '\\' OR f.path LIKE ? ESCAPE '\\'"
            f" OR f.path LIKE ? ESCAPE '\\') AND {clause}",
            [term, f"%/{escaped}", f"%/{escaped}.%", f"{escaped}.%", *scope_params],
        )
        for row in rows:
            keep(as_int(row[0]), (-1, PATH, as_str(row[1]), as_int(row[2]), as_int(row[3])))
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
