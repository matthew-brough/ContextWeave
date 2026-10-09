from collections.abc import Sequence

from contextweave.core.refs import Anchor, ByteAnchor, PartAnchor, SymbolAnchor
from contextweave.index.store import Row, Store
from contextweave.retrieve.candidates import Candidate
from contextweave.retrieve.rows import (
    as_group,
    as_int,
    as_opt_int,
    as_opt_str,
    as_profile,
    as_str,
)

CHUNK_SELECT = (
    "SELECT c.id, f.path, c.start_byte, c.end_byte, c.kind, c.part, f.profile, f.grp,"
    " (SELECT s.qualified_name FROM chunk_symbols cs JOIN symbols s ON s.id = cs.symbol_id"
    "  WHERE cs.chunk_id = c.id ORDER BY cs.rowid LIMIT 1),"
    " (SELECT s.kind FROM chunk_symbols cs JOIN symbols s ON s.id = cs.symbol_id"
    "  WHERE cs.chunk_id = c.id ORDER BY cs.rowid LIMIT 1)"
    " FROM chunks c JOIN files f ON f.id = c.file_id"
)


def anchor_of(qualified_name: str | None, part: int | None, start: int, end: int) -> Anchor:
    """The ref anchor of a chunk row, matching index.chunk.chunk_anchor."""
    if qualified_name is None:
        return ByteAnchor(start, end)
    return SymbolAnchor(qualified_name) if part is None else PartAnchor(qualified_name, part)


def candidate_from_row(row: Row) -> Candidate:
    start = as_int(row[2])
    end = as_int(row[3])
    qualified_name = as_opt_str(row[8])
    symbol_kind = as_opt_str(row[9])
    return Candidate(
        chunk_id=as_int(row[0]),
        path=as_str(row[1]),
        start_byte=start,
        end_byte=end,
        anchor=anchor_of(qualified_name, as_opt_int(row[5]), start, end),
        profile=as_profile(row[6]),
        group=as_group(row[7]),
        qualified_name=qualified_name,
        kind=symbol_kind or as_str(row[4]),
    )


def load_candidates(store: Store, chunk_ids: Sequence[int]) -> dict[int, Candidate]:
    """Candidates for chunk ids, keyed by id; ids with no chunk row are absent."""
    loaded: dict[int, Candidate] = {}
    for offset in range(0, len(chunk_ids), 500):
        batch = chunk_ids[offset : offset + 500]
        marks = ", ".join("?" for _ in batch)
        for row in store.query(f"{CHUNK_SELECT} WHERE c.id IN ({marks})", list(batch)):
            candidate = candidate_from_row(row)
            loaded[candidate.chunk_id] = candidate
    return loaded


def load_file_candidates(store: Store, path: str) -> tuple[Candidate, ...]:
    """Candidates for every chunk of a file in source order."""
    rows = store.query(
        f"{CHUNK_SELECT} WHERE f.path = ? ORDER BY c.start_byte, c.end_byte, c.part, c.id", (path,)
    )
    return tuple(candidate_from_row(row) for row in rows)
