import json
import sqlite3
import time
from array import array
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

_BATCH = 400

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


class DerivedCache:
    """Content-addressed per-user cache of annotations and vectors, keyed by exact hash."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._db = sqlite3.connect(path)
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
                values = array("f")
                values.frombytes(blob)
                found[content_hash] = tuple(values)
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
        if len(vector) != dim:
            raise ValueError(f"vector length {len(vector)} != dim {dim}")
        self._db.execute(
            "INSERT OR REPLACE INTO derived_vectors VALUES (?, ?, ?, ?, ?, ?)",
            (content_hash, space, model_id, dim, array("f", vector).tobytes(), self._clock()),
        )
        self._db.commit()

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
                f"DELETE FROM {table} WHERE content_hash NOT IN (SELECT h FROM live_hashes)"
                " OR (? IS NOT NULL AND last_used < ?)",
                (cutoff, cutoff),
            )
            counts.append(cursor.rowcount)
        self._db.commit()
        return GcResult(counts[0], counts[1])
