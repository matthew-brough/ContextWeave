import hashlib
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import sqlite_vec

from contextweave.index.store import Store

type SpaceKind = Literal["code", "annot"]
type SpaceStatus = Literal["building", "live", "retired"]

CODE: SpaceKind = "code"
ANNOT: SpaceKind = "annot"
KINDS: tuple[SpaceKind, ...] = (CODE, ANNOT)


@dataclass(frozen=True, slots=True)
class Space:
    """One vector space: the table holding every vector of one kind from one model at one dim."""

    name: str
    kind: SpaceKind
    model_id: str
    dim: int
    status: SpaceStatus


@dataclass(frozen=True, slots=True)
class VectorInput:
    """A chunk's text to embed and the hash that keys its vector."""

    chunk_id: int
    key: str
    text: str


def space_name(kind: SpaceKind, model_id: str, dim: int) -> str:
    """Table name vec_<kind>_<model slug>_<dim>, suffixed with a hash so slugs cannot collide."""
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")[:32]
    digest = hashlib.sha256(model_id.encode()).hexdigest()[:8]
    return f"vec_{kind}_{slug}_{digest}_{dim}"


def _kind(value: object) -> SpaceKind:
    return ANNOT if value == ANNOT else CODE


def _status(value: object) -> SpaceStatus:
    if value == "live":
        return "live"
    return "retired" if value == "retired" else "building"


def _space(row: tuple[object, ...]) -> Space:
    name, kind, model_id, dim, status = row
    if not isinstance(name, str) or not isinstance(model_id, str) or not isinstance(dim, int):
        raise ValueError(f"corrupt vec_spaces row: {row!r}")
    return Space(name, _kind(kind), model_id, dim, _status(status))


def vec_available(store: Store) -> bool:
    """Whether the sqlite-vec extension is loaded on the store connection."""
    try:
        store.query("SELECT vec_version()")
    except sqlite3.Error:
        return False
    return True


def spaces(store: Store) -> tuple[Space, ...]:
    """Every registered vector space."""
    rows = store.query("SELECT name, space, model_id, dim, status FROM vec_spaces ORDER BY name")
    return tuple(_space(row) for row in rows)


def find_space(store: Store, kind: SpaceKind, model_id: str, dim: int) -> Space | None:
    """The space for exactly (kind, model_id, dim), if registered."""
    rows = store.query(
        "SELECT name, space, model_id, dim, status FROM vec_spaces"
        " WHERE space = ? AND model_id = ? AND dim = ?",
        (kind, model_id, dim),
    )
    return _space(rows[0]) if rows else None


def ensure_space(store: Store, kind: SpaceKind, model_id: str, dim: int) -> Space:
    """Return the space for (kind, model_id, dim), creating it as building when new."""
    existing = find_space(store, kind, model_id, dim)
    if existing is not None:
        return existing
    name = space_name(kind, model_id, dim)
    with store.write_transaction():
        store.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name}"
            f" USING vec0(embedding float[{int(dim)}] distance_metric=cosine)"
        )
        store.execute(
            "INSERT INTO vec_spaces (name, space, model_id, dim, status)"
            " VALUES (?, ?, ?, ?, 'building')",
            (name, kind, model_id, dim),
        )
    return Space(name, kind, model_id, dim, "building")


def _drop(store: Store, name: str) -> None:
    store.execute(f"DROP TABLE IF EXISTS {name}")
    store.execute("DELETE FROM vec_spaces WHERE name = ?", (name,))


def mark_live(store: Store, space: Space) -> Space:
    """Swap the space in: it becomes live and every other space of its kind is evicted."""
    with store.write_transaction():
        store.execute("UPDATE vec_spaces SET status = 'live' WHERE name = ?", (space.name,))
        for other in spaces(store):
            if other.kind == space.kind and other.name != space.name:
                _drop(store, other.name)
    return Space(space.name, space.kind, space.model_id, space.dim, "live")


def indexed_keys(store: Store, space: Space) -> Mapping[int, str]:
    """chunk id to the key its stored vector was computed from."""
    rows = store.query(
        "SELECT chunk_id, content_hash FROM vectors_meta WHERE space_name = ?", (space.name,)
    )
    return {int(str(row[0])): str(row[1]) for row in rows}


def pending_inputs(
    store: Store, space: Space, inputs: Sequence[VectorInput]
) -> tuple[VectorInput, ...]:
    """Inputs whose chunk has no vector in the space, or one computed from other content."""
    held = indexed_keys(store, space)
    return tuple(item for item in inputs if held.get(item.chunk_id) != item.key)


def write_vectors(
    store: Store, space: Space, rows: Sequence[tuple[VectorInput, Sequence[float]]]
) -> None:
    """Replace the vectors of the given chunks in one transaction."""
    with store.write_transaction():
        for item, vector in rows:
            store.execute(f"DELETE FROM {space.name} WHERE rowid = ?", (item.chunk_id,))
            store.execute(
                f"INSERT INTO {space.name} (rowid, embedding) VALUES (?, ?)",
                (item.chunk_id, sqlite_vec.serialize_float32(list(vector))),
            )
            store.execute(
                "INSERT OR REPLACE INTO vectors_meta (chunk_id, space_name, content_hash)"
                " VALUES (?, ?, ?)",
                (item.chunk_id, space.name, item.key),
            )


def purge_orphans(store: Store, space: Space) -> int:
    """Delete vectors whose chunk no longer has a vectors_meta row; returns how many."""
    held = indexed_keys(store, space)
    stored = [int(str(row[0])) for row in store.query(f"SELECT rowid FROM {space.name}")]
    orphans = [rowid for rowid in stored if rowid not in held]
    if orphans:
        with store.write_transaction():
            for rowid in orphans:
                store.execute(f"DELETE FROM {space.name} WHERE rowid = ?", (rowid,))
    return len(orphans)
