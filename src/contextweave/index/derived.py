import json
import re
import sqlite3
import time
from array import array
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

_BATCH = 400
PART_SEP = "#"
WHOLE = -1
_PART_SUFFIX = re.compile(r"#(\d+):(\d+):(\d+)")
SQLITE_BUSY_S = 30.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS derived_annotations(
    content_hash TEXT NOT NULL,
    model_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    purpose TEXT NOT NULL,
    tags TEXT NOT NULL,
    last_used REAL NOT NULL,
    PRIMARY KEY (content_hash, model_id, prompt_version)
);
CREATE TABLE IF NOT EXISTS derived_vectors(
    content_hash TEXT NOT NULL,
    space TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    last_used REAL NOT NULL,
    PRIMARY KEY (content_hash, space, model_id, dim)
);
CREATE INDEX IF NOT EXISTS derived_annotations_used ON derived_annotations(last_used);
CREATE INDEX IF NOT EXISTS derived_vectors_used ON derived_vectors(last_used);
"""


@dataclass(frozen=True, slots=True)
class DerivedAnnotation:
    purpose: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GcResult:
    annotations: int
    vectors: int


def cached_or_compute[K: Hashable, V](
    keys: Sequence[K],
    lookup: Callable[[Sequence[K]], Mapping[K, V]],
    compute: Callable[[Sequence[K]], Mapping[K, V]],
    store: Callable[[K, V], None],
) -> dict[K, V]:
    """Return a value per key, computing and storing only the keys the lookup missed."""
    found = dict(lookup(keys))
    missing = [key for key in dict.fromkeys(keys) if key not in found]
    if missing:
        for key, value in compute(missing).items():
            store(key, value)
            found[key] = value
    return {key: found[key] for key in keys if key in found}


def _batches(items: Sequence[str]) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), _BATCH):
        yield items[start : start + _BATCH]


_BASE_HASH = f"substr(content_hash, 1, instr(content_hash || '{PART_SEP}', '{PART_SEP}') - 1)"


_KEY_PARTS = (
    "space=? AND model_id=? AND dim=?"
    " AND (content_hash = ? OR (content_hash > ? AND content_hash < ?))"
)


def _key_parts(key: str, space: str, model_id: str, dim: int) -> tuple[str | int, ...]:
    return (space, model_id, dim, key, key + PART_SEP, key + chr(ord(PART_SEP) + 1))


def _floats(blob: bytes) -> tuple[float, ...]:
    values = array("f")
    values.frombytes(blob)
    return tuple(values)


def _parts(
    key: str, rows: Sequence[tuple[str, bytes]]
) -> tuple[tuple[int, int, tuple[float, ...]], ...]:
    whole = [blob for content_hash, blob in rows if content_hash == key]
    if whole:
        return ((0, WHOLE, _floats(whole[0])),)
    indexed: dict[int, tuple[int, int, tuple[float, ...]]] = {}
    for content_hash, blob in rows:
        match = _PART_SUFFIX.fullmatch(content_hash, len(key))
        if match:
            indexed[int(match[1])] = (int(match[2]), int(match[3]), _floats(blob))
    if sorted(indexed) != list(range(len(indexed))):
        return ()
    return tuple(indexed[index] for index in range(len(indexed)))


class DerivedCache:
    """Content-addressed per-user cache of annotations and vectors, keyed by exact hash."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._db = sqlite3.connect(path, timeout=SQLITE_BUSY_S)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def get_annotation(
        self, content_hash: str, model_id: str, prompt_version: str
    ) -> DerivedAnnotation | None:
        """Return the cached annotation for the exact key, refreshing last_used."""
        key = (content_hash, model_id, prompt_version)
        row: tuple[str, str] | None = self._db.execute(
            "SELECT purpose, tags FROM derived_annotations"
            " WHERE content_hash=? AND model_id=? AND prompt_version=?",
            key,
        ).fetchone()
        if row is None:
            return None
        self._db.execute(
            "UPDATE derived_annotations SET last_used=?"
            " WHERE content_hash=? AND model_id=? AND prompt_version=?",
            (self._clock(), *key),
        )
        self._db.commit()
        tags: list[str] = json.loads(row[1])
        return DerivedAnnotation(row[0], tuple(tags))

    def put_annotation(
        self,
        content_hash: str,
        model_id: str,
        prompt_version: str,
        annotation: DerivedAnnotation,
    ) -> None:
        """Insert or replace an annotation."""
        self._db.execute(
            "INSERT OR REPLACE INTO derived_annotations VALUES (?, ?, ?, ?, ?, ?)",
            (
                content_hash,
                model_id,
                prompt_version,
                annotation.purpose,
                json.dumps(list(annotation.tags)),
                self._clock(),
            ),
        )
        self._db.commit()

    def get_vectors(
        self, hashes: Sequence[str], space: str, model_id: str, dim: int
    ) -> dict[str, tuple[float, ...]]:
        """Return vectors for the exact hashes held under (space, model_id, dim)."""
        found: dict[str, tuple[float, ...]] = {}
        now = self._clock()
        for batch in _batches(list(dict.fromkeys(hashes))):
            marks = ",".join("?" * len(batch))
            params = (space, model_id, dim, *batch)
            rows: list[tuple[str, bytes]] = self._db.execute(
                "SELECT content_hash, vector FROM derived_vectors"
                f" WHERE space=? AND model_id=? AND dim=? AND content_hash IN ({marks})",
                params,
            ).fetchall()
            for content_hash, blob in rows:
                found[content_hash] = _floats(blob)
            self._db.execute(
                f"UPDATE derived_vectors SET last_used=? WHERE space=? AND model_id=? AND dim=?"
                f" AND content_hash IN ({marks})",
                (now, *params),
            )
        self._db.commit()
        return found

    def put_vector(
        self, content_hash: str, space: str, model_id: str, dim: int, vector: Sequence[float]
    ) -> None:
        """Insert or replace a float32 vector; its length must equal dim."""
        self.put_vectors({content_hash: vector}, space, model_id, dim)

    def put_vectors(
        self, vectors: Mapping[str, Sequence[float]], space: str, model_id: str, dim: int
    ) -> None:
        """Insert or replace many float32 vectors in one commit; each length must equal dim."""
        for vector in vectors.values():
            if len(vector) != dim:
                raise ValueError(f"vector length {len(vector)} != dim {dim}")
        now = self._clock()
        self._db.executemany(
            "INSERT OR REPLACE INTO derived_vectors VALUES (?, ?, ?, ?, ?, ?)",
            (
                (content_hash, space, model_id, dim, array("f", vector).tobytes(), now)
                for content_hash, vector in vectors.items()
            ),
        )
        self._db.commit()

    def get_parts(
        self, keys: Sequence[str], space: str, model_id: str, dim: int
    ) -> dict[str, tuple[tuple[int, int, tuple[float, ...]], ...]]:
        """Return (start, end, vector) parts per key: one part ending at WHOLE, or every stored section."""
        found: dict[str, tuple[tuple[int, int, tuple[float, ...]], ...]] = {}
        now = self._clock()
        for key in dict.fromkeys(keys):
            params = _key_parts(key, space, model_id, dim)
            rows: list[tuple[str, bytes]] = self._db.execute(
                f"SELECT content_hash, vector FROM derived_vectors WHERE {_KEY_PARTS}", params
            ).fetchall()
            parts = _parts(key, rows)
            if parts:
                found[key] = parts
                self._db.execute(
                    f"UPDATE derived_vectors SET last_used=? WHERE {_KEY_PARTS}", (now, *params)
                )
        self._db.commit()
        return found

    def put_parts(
        self,
        parts: Mapping[str, Sequence[tuple[int, int, Sequence[float]]]],
        space: str,
        model_id: str,
        dim: int,
    ) -> None:
        """Replace every stored part of each key; a lone part is stored under the bare key."""
        rows: dict[str, Sequence[float]] = {}
        for key, sections in parts.items():
            self._db.execute(
                f"DELETE FROM derived_vectors WHERE {_KEY_PARTS}",
                _key_parts(key, space, model_id, dim),
            )
            if len(sections) == 1:
                rows[key] = sections[0][2]
                continue
            for index, (start, end, vector) in enumerate(sections):
                rows[f"{key}{PART_SEP}{index}:{start}:{end}"] = vector
        self.put_vectors(rows, space, model_id, dim)

    def gc(
        self,
        registered_hashes: Callable[[], Iterable[str]],
        older_than: timedelta | None = None,
    ) -> GcResult:
        """Drop rows whose hash is in no registered project, or unused for older_than."""
        cutoff = self._clock() - older_than.total_seconds() if older_than else None
        self._db.execute("CREATE TEMP TABLE IF NOT EXISTS live_hashes(h TEXT PRIMARY KEY)")
        self._db.execute("DELETE FROM live_hashes")
        self._db.executemany(
            "INSERT OR IGNORE INTO live_hashes VALUES (?)", ((h,) for h in registered_hashes())
        )
        counts: list[int] = []
        for table in ("derived_annotations", "derived_vectors"):
            cursor = self._db.execute(
                f"DELETE FROM {table} WHERE {_BASE_HASH} NOT IN (SELECT h FROM live_hashes)"
                " OR (? IS NOT NULL AND last_used < ?)",
                (cutoff, cutoff),
            )
            counts.append(cursor.rowcount)
        self._db.commit()
        return GcResult(counts[0], counts[1])
