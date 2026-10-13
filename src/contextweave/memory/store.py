import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Literal

import sqlite_vec

from contextweave.core.paths import MEMORY_DB_NAME, projects_dir, user_state_dir
from contextweave.memory.wire import SavingsBucketTotals

SCHEMA_VERSION = 2

type Row = tuple[object, ...]
type SessionStatus = Literal["active", "completed", "failed"]
type SavingsBucket = Literal["retrieval", "progressive_disclosure"]

SAVINGS_BUCKETS: tuple[SavingsBucket, ...] = ("retrieval", "progressive_disclosure")

_RETIRED_SCHEMAS = (1,)
_KEPT_TABLES = frozenset({"sessions", "prompts", "savings_log", "sqlite_sequence"})

_BASE_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
      id TEXT PRIMARY KEY,
      project_key TEXT NOT NULL,
      started_at_epoch INTEGER NOT NULL,
      ended_at_epoch INTEGER,
      exit_reason TEXT,
      prompt_count INTEGER NOT NULL DEFAULT 0,
      status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','completed','failed'))
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(project_key, started_at_epoch)",
    """
    CREATE TABLE IF NOT EXISTS prompts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT NOT NULL,
      prompt_number INTEGER NOT NULL,
      prompt_text TEXT NOT NULL,
      created_at_epoch INTEGER NOT NULL,
      UNIQUE(project_key, session_id, prompt_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS savings_log (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      bucket TEXT NOT NULL,
      baseline INTEGER NOT NULL,
      served INTEGER NOT NULL,
      meta TEXT,
      ts INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_savings_bucket ON savings_log(project_key, bucket, ts)",
)


class MemoryStoreError(Exception):
    """Base for memory store failures that callers degrade on."""


class MemorySchemaError(MemoryStoreError):
    """Raised when memory.db carries an unknown schema version."""


class MemoryRowError(MemoryStoreError):
    """Raised when a stored row has an unexpected column type."""


def state_dir(env: Mapping[str, str]) -> Path:
    """Return $XDG_STATE_HOME/contextweave, falling back to ~/.local/state/contextweave."""
    return user_state_dir(env)


def memory_db_path(key: str, env: Mapping[str, str]) -> Path:
    """Return <state>/projects/<key>/memory.db."""
    return projects_dir(env) / key / MEMORY_DB_NAME


def _str(row: Row, index: int) -> str:
    value = row[index]
    if isinstance(value, str):
        return value
    raise MemoryRowError(f"column {index}: expected str, got {type(value).__name__}")


def _int(row: Row, index: int) -> int:
    value = row[index]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise MemoryRowError(f"column {index}: expected int, got {type(value).__name__}")


class MemoryStore:
    """Per-project memory.db; every statement is filtered by project_key."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        project_key: str,
        *,
        clock: Callable[[], float],
    ) -> None:
        self.conn = conn
        self.project_key = project_key
        self.clock = clock

    @classmethod
    def open(
        cls, path: Path, project_key: str, *, clock: Callable[[], float] = time.time
    ) -> MemoryStore:
        """Open or create memory.db for a project, creating the schema when the file is new."""
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 30000")
            _ensure_schema(conn)
        except BaseException:
            conn.close()
            raise
        return cls(conn, project_key, clock=clock)

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()

    def now(self) -> int:
        """Current epoch seconds from the store clock."""
        return int(self.clock())

    def user_version(self) -> int:
        """Return PRAGMA user_version."""
        return _int(self._one("PRAGMA user_version", ()), 0)

    def _rows(self, sql: str, params: Sequence[object]) -> list[Row]:
        rows: list[Row] = self.conn.execute(sql, params).fetchall()
        return rows

    def _one(self, sql: str, params: Sequence[object]) -> Row:
        row: Row | None = self.conn.execute(sql, params).fetchone()
        if row is None:
            raise MemoryRowError("expected one row")
        return row

    def _maybe(self, sql: str, params: Sequence[object]) -> Row | None:
        row: Row | None = self.conn.execute(sql, params).fetchone()
        return row

    def ensure_session(self, session_id: str) -> None:
        """Create the session row if absent."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO sessions(id, project_key, started_at_epoch) VALUES (?,?,?)",
                (session_id, self.project_key, self.now()),
            )

    def end_session(self, session_id: str, status: SessionStatus, exit_reason: str | None) -> bool:
        """Mark a session finished; returns False when the session is unknown."""
        with self.conn:
            cur = self.conn.execute(
                "UPDATE sessions SET status=?, exit_reason=?, ended_at_epoch=? "
                "WHERE project_key=? AND id=?",
                (status, exit_reason, self.now(), self.project_key, session_id),
            )
        return cur.rowcount > 0

    def add_prompt(self, session_id: str, text: str) -> int:
        """Append a prompt to the session and return its 1-based number."""
        self.ensure_session(session_id)
        with self.conn:
            number = _int(
                self._one(
                    "UPDATE sessions SET prompt_count = prompt_count + 1 "
                    "WHERE project_key=? AND id=? RETURNING prompt_count",
                    (self.project_key, session_id),
                ),
                0,
            )
            self.conn.execute(
                "INSERT INTO prompts(project_key, session_id, prompt_number, prompt_text, "
                "created_at_epoch) VALUES (?,?,?,?,?)",
                (self.project_key, session_id, number, text, self.now()),
            )
        return number

    def current_prompt_number(self, session_id: str) -> int:
        """Return the session's latest prompt number, 0 when it has none or is unknown."""
        row = self._maybe(
            "SELECT prompt_count FROM sessions WHERE project_key=? AND id=?",
            (self.project_key, session_id),
        )
        return 0 if row is None else _int(row, 0)

    def current_prompt_epoch(self, session_id: str) -> int | None:
        """Epoch seconds at which the session's latest prompt was stored, None when it has none."""
        row = self._maybe(
            "SELECT created_at_epoch FROM prompts WHERE project_key=? AND session_id=? "
            "ORDER BY prompt_number DESC LIMIT 1",
            (self.project_key, session_id),
        )
        return None if row is None else _int(row, 0)

    def add_savings(
        self, bucket: SavingsBucket, baseline: int, served: int, meta_json: str | None
    ) -> None:
        """Append one ledger row."""
        with self.conn:
            self.conn.execute(
                "INSERT INTO savings_log(project_key, bucket, baseline, served, meta, ts) "
                "VALUES (?,?,?,?,?,?)",
                (self.project_key, bucket, baseline, served, meta_json, self.now()),
            )

    def savings_totals(self) -> dict[SavingsBucket, SavingsBucketTotals]:
        """Per-bucket totals, zero-filled for every known bucket."""
        totals: dict[SavingsBucket, SavingsBucketTotals] = {
            bucket: SavingsBucketTotals(baseline=0, served=0, saved=0, calls=0)
            for bucket in SAVINGS_BUCKETS
        }
        for r in self._rows(
            "SELECT bucket, SUM(baseline), SUM(served), COUNT(*) FROM savings_log "
            "WHERE project_key=? GROUP BY bucket",
            (self.project_key,),
        ):
            bucket = _bucket(_str(r, 0))
            if bucket is None:
                continue
            baseline, served = _int(r, 1), _int(r, 2)
            totals[bucket] = SavingsBucketTotals(
                baseline=baseline, served=served, saved=baseline - served, calls=_int(r, 3)
            )
        return totals


def _bucket(value: str) -> SavingsBucket | None:
    match value:
        case "retrieval" | "progressive_disclosure":
            return value
        case _:
            return None


def _load_vec(conn: sqlite3.Connection) -> bool:
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except AttributeError, sqlite3.Error, OSError:
        return False
    return True


def _ensure_schema(conn: sqlite3.Connection) -> None:
    row: Row | None = conn.execute("PRAGMA user_version").fetchone()
    version = 0 if row is None else _int(row, 0)
    if version == SCHEMA_VERSION:
        return
    if version not in (0, *_RETIRED_SCHEMAS):
        raise MemorySchemaError(f"memory.db schema {version}, expected {SCHEMA_VERSION}")
    with conn:
        if version in _RETIRED_SCHEMAS:
            _drop_retired(conn)
        for statement in _BASE_SCHEMA:
            conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _drop_retired(conn: sqlite3.Connection) -> None:
    """Drop the tables, FTS shadows and vector spaces the removed recall memory left behind."""
    _load_vec(conn)
    rows: list[tuple[str]] = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND sql LIKE 'CREATE VIRTUAL TABLE%'"
    ).fetchall()
    rows += conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    for (name,) in dict.fromkeys(rows):
        if name not in _KEPT_TABLES:
            with suppress(sqlite3.OperationalError):
                conn.execute(f'DROP TABLE IF EXISTS "{name}"')
