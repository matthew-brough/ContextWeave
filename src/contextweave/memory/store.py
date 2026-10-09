import hashlib
import re
import sqlite3
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import sqlite_vec

from contextweave.core.paths import MEMORY_DB_NAME, projects_dir, user_state_dir
from contextweave.memory.wire import (
    EventDetail,
    PayloadState,
    RecallHit,
    RecallKind,
    SavingsBucketTotals,
    SummaryTier,
    Timeline,
    TimelineEvent,
    TimelineTurn,
)

SCHEMA_VERSION = 1
EVENT_PAYLOAD_READ_CAP = 4096

type Row = tuple[object, ...]
type SessionStatus = Literal["active", "completed", "failed"]
type Source = Literal["manual", "auto"]
type SavingsBucket = Literal[
    "retrieval", "memory_recall", "progressive_disclosure", "turn_summarization"
]

SAVINGS_BUCKETS: tuple[SavingsBucket, ...] = (
    "retrieval",
    "memory_recall",
    "progressive_disclosure",
    "turn_summarization",
)

_TABLES_WITH_FTS: dict[str, tuple[str, ...]] = {
    "decisions": ("decision", "reason"),
    "code_areas": ("file_path", "description"),
    "notes": ("text",),
    "turn_summaries": ("summary",),
}

_FTS_TABLE: dict[RecallKind, str] = {
    "decision": "decisions",
    "code_area": "code_areas",
    "note": "notes",
    "turn_summary": "turn_summaries",
}

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
    CREATE TABLE IF NOT EXISTS tool_event_payloads (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      raw_input TEXT NOT NULL,
      raw_output TEXT,
      size_bytes INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_events (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT NOT NULL,
      prompt_number INTEGER NOT NULL,
      tool_name TEXT NOT NULL,
      payload_id INTEGER REFERENCES tool_event_payloads(id) ON DELETE SET NULL,
      summary TEXT,
      created_at_epoch INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_events_turn ON tool_events(project_key, session_id, prompt_number)",
    """
    CREATE TABLE IF NOT EXISTS turn_summaries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT NOT NULL,
      prompt_number INTEGER NOT NULL,
      summary TEXT NOT NULL,
      tier TEXT NOT NULL CHECK(tier IN ('extractive','truncated','synthesized')),
      synthesized INTEGER NOT NULL,
      model TEXT,
      prompt_version INTEGER,
      created_at_epoch INTEGER NOT NULL,
      UNIQUE(project_key, session_id, prompt_number)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS decisions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT,
      decision TEXT NOT NULL,
      reason TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'manual' CHECK(source IN ('manual','auto')),
      created_at_epoch INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS code_areas (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT,
      file_path TEXT NOT NULL,
      description TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'manual' CHECK(source IN ('manual','auto')),
      created_at_epoch INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_code_areas_file ON code_areas(project_key, file_path)",
    """
    CREATE TABLE IF NOT EXISTS notes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT,
      text TEXT NOT NULL,
      source TEXT NOT NULL DEFAULT 'manual' CHECK(source IN ('manual','auto')),
      created_at_epoch INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pending_summaries (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      session_id TEXT NOT NULL,
      prompt_number INTEGER NOT NULL,
      enqueued_at_epoch INTEGER NOT NULL,
      attempts INTEGER NOT NULL DEFAULT 0,
      last_error TEXT,
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
    """
    CREATE TABLE IF NOT EXISTS vec_spaces (
      name TEXT PRIMARY KEY,
      model_id TEXT NOT NULL,
      dim INTEGER NOT NULL,
      UNIQUE(model_id, dim)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS vectors_meta (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      project_key TEXT NOT NULL,
      kind TEXT NOT NULL,
      row_id INTEGER NOT NULL,
      content_hash TEXT NOT NULL,
      model_id TEXT NOT NULL,
      dim INTEGER NOT NULL,
      UNIQUE(kind, row_id, model_id, dim)
    )
    """,
)


def _fts_statements(table: str, columns: tuple[str, ...]) -> list[str]:
    cols = ", ".join(columns)
    new_cols = ", ".join(f"new.{c}" for c in columns)
    old_cols = ", ".join(f"old.{c}" for c in columns)
    fts = f"{table}_fts"
    delete = f"INSERT INTO {fts}({fts}, rowid, {cols}) VALUES('delete', old.id, {old_cols});"
    insert = f"INSERT INTO {fts}(rowid, {cols}) VALUES (new.id, {new_cols});"
    return [
        f"CREATE VIRTUAL TABLE IF NOT EXISTS {fts} USING fts5({cols}, content='{table}', content_rowid='id')",
        f"CREATE TRIGGER IF NOT EXISTS {table}_ai AFTER INSERT ON {table} BEGIN {insert} END",
        f"CREATE TRIGGER IF NOT EXISTS {table}_ad AFTER DELETE ON {table} BEGIN {delete} END",
        f"CREATE TRIGGER IF NOT EXISTS {table}_au AFTER UPDATE ON {table} BEGIN {delete} {insert} END",
    ]


class MemoryStoreError(Exception):
    """Base for memory store failures that callers degrade on."""


class MemorySchemaError(MemoryStoreError):
    """Raised when memory.db carries an unknown schema version."""


class MemoryRowError(MemoryStoreError):
    """Raised when a stored row has an unexpected column type."""


@dataclass(frozen=True, slots=True)
class VectorMatch:
    kind: RecallKind
    row_id: int
    distance: float


@dataclass(frozen=True, slots=True)
class PendingSummary:
    session_id: str
    prompt_number: int
    attempts: int


@dataclass(frozen=True, slots=True)
class SummaryInput:
    summary: str
    tier: SummaryTier
    model: str | None = None
    prompt_version: int | None = None


def state_dir(env: Mapping[str, str]) -> Path:
    """Return $XDG_STATE_HOME/contextweave, falling back to ~/.local/state/contextweave."""
    return user_state_dir(env)


def memory_db_path(key: str, env: Mapping[str, str]) -> Path:
    """Return <state>/projects/<key>/memory.db."""
    return projects_dir(env) / key / MEMORY_DB_NAME


def project_memory_dbs(env: Mapping[str, str]) -> list[tuple[str, Path]]:
    """List (project key, memory.db path) for every project with a memory db, sorted by key."""
    root = projects_dir(env)
    if not root.is_dir():
        return []
    return sorted((path.parent.name, path) for path in root.glob(f"*/{MEMORY_DB_NAME}"))


def content_hash(text: str) -> str:
    """Return the first 16 hex chars of sha256 over the text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def vector_text(kind: RecallKind, primary: str, secondary: str | None) -> str:
    """Return the text embedded for a row of the given kind."""
    match kind:
        case "decision":
            return f"{primary} — {secondary}" if secondary else primary
        case "code_area":
            return f"{secondary}: {primary}" if secondary else primary
        case "note" | "turn_summary":
            return primary


def fts_match_query(topic: str) -> str | None:
    """Build an OR-of-quoted-terms FTS5 query, or None when the topic has no terms."""
    terms = list(dict.fromkeys(re.findall(r"\w+", topic)))
    if not terms:
        return None
    return " OR ".join(f'"{term}"' for term in terms)


def _vec_space_name(model_id: str, dim: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")[:24]
    return f"vec_mem_{slug}_{hashlib.sha1(model_id.encode()).hexdigest()[:6]}_{dim}"


def _str(row: Row, index: int) -> str:
    value = row[index]
    if isinstance(value, str):
        return value
    raise MemoryRowError(f"column {index}: expected str, got {type(value).__name__}")


def _opt_str(row: Row, index: int) -> str | None:
    value = row[index]
    if value is None:
        return None
    return _str(row, index)


def _int(row: Row, index: int) -> int:
    value = row[index]
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise MemoryRowError(f"column {index}: expected int, got {type(value).__name__}")


def _opt_int(row: Row, index: int) -> int | None:
    if row[index] is None:
        return None
    return _int(row, index)


def _tier(row: Row, index: int) -> SummaryTier:
    value = _str(row, index)
    match value:
        case "extractive" | "truncated" | "synthesized":
            return value
        case _:
            raise MemoryRowError(f"unknown summary tier: {value!r}")


def _truncate_bytes(text: str, cap: int) -> tuple[str, bool]:
    raw = text.encode()
    if len(raw) <= cap:
        return text, False
    return raw[:cap].decode(errors="ignore"), True


class MemoryStore:
    """Per-project memory.db; every statement is filtered by project_key (§V.13)."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        project_key: str,
        *,
        vec_enabled: bool,
        clock: Callable[[], float],
    ) -> None:
        self.conn = conn
        self.project_key = project_key
        self.vec_enabled = vec_enabled
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
            conn.execute("PRAGMA busy_timeout = 5000")
            vec_enabled = _load_vec(conn)
            _ensure_schema(conn)
        except BaseException:
            conn.close()
            raise
        return cls(conn, project_key, vec_enabled=vec_enabled, clock=clock)

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

    def add_tool_event(
        self,
        session_id: str,
        prompt_number: int,
        tool_name: str,
        summary: str | None,
        raw_input: str | None,
        raw_output: str | None,
    ) -> int:
        """Record a tool event with its optional raw payload and return the event id."""
        self.ensure_session(session_id)
        with self.conn:
            payload_id: int | None = None
            if raw_input is not None or raw_output is not None:
                body = raw_input or ""
                size = len(body.encode()) + len((raw_output or "").encode())
                payload_id = self.conn.execute(
                    "INSERT INTO tool_event_payloads(project_key, raw_input, raw_output, "
                    "size_bytes) VALUES (?,?,?,?)",
                    (self.project_key, body, raw_output, size),
                ).lastrowid
            cur = self.conn.execute(
                "INSERT INTO tool_events(project_key, session_id, prompt_number, tool_name, "
                "payload_id, summary, created_at_epoch) VALUES (?,?,?,?,?,?,?)",
                (
                    self.project_key,
                    session_id,
                    prompt_number,
                    tool_name,
                    payload_id,
                    summary,
                    self.now(),
                ),
            )
        return _row_id(cur.lastrowid)

    def current_prompt_number(self, session_id: str) -> int:
        """Return the session's latest prompt number, 0 when it has none or is unknown."""
        row = self._maybe(
            "SELECT prompt_count FROM sessions WHERE project_key=? AND id=?",
            (self.project_key, session_id),
        )
        return 0 if row is None else _int(row, 0)

    def recent(self, kind: Literal["decision", "code_area"], limit: int) -> list[RecallHit]:
        """Return the newest decisions or code areas of this project, newest first."""
        table = _FTS_TABLE[kind]
        rows = self._rows(
            f"SELECT id FROM {table} WHERE project_key=? ORDER BY created_at_epoch DESC, id DESC "
            "LIMIT ?",
            (self.project_key, limit),
        )
        ids = [_int(row, 0) for row in rows]
        loaded = self.hits(kind, ids)
        return [loaded[row_id] for row_id in ids if row_id in loaded]

    def prompt_text(self, session_id: str, prompt_number: int) -> str | None:
        """Return a stored prompt's text."""
        row = self._maybe(
            "SELECT prompt_text FROM prompts WHERE project_key=? AND session_id=? "
            "AND prompt_number=?",
            (self.project_key, session_id, prompt_number),
        )
        return None if row is None else _str(row, 0)

    def enqueue_summary(self, session_id: str, prompt_number: int) -> None:
        """Queue a turn for summarisation (idempotent)."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO pending_summaries(project_key, session_id, prompt_number, "
                "enqueued_at_epoch) VALUES (?,?,?,?)",
                (self.project_key, session_id, prompt_number, self.now()),
            )

    def pending_summaries(self, limit: int = 50) -> list[PendingSummary]:
        """List queued turns oldest first."""
        rows = self._rows(
            "SELECT session_id, prompt_number, attempts FROM pending_summaries "
            "WHERE project_key=? ORDER BY enqueued_at_epoch, id LIMIT ?",
            (self.project_key, limit),
        )
        return [PendingSummary(_str(r, 0), _int(r, 1), _int(r, 2)) for r in rows]

    def fail_pending(self, session_id: str, prompt_number: int, error: str) -> None:
        """Count a failed summarisation attempt against a queued turn."""
        with self.conn:
            self.conn.execute(
                "UPDATE pending_summaries SET attempts = attempts + 1, last_error=? "
                "WHERE project_key=? AND session_id=? AND prompt_number=?",
                (error, self.project_key, session_id, prompt_number),
            )

    def turn_events_text(self, session_id: str, prompt_number: int) -> list[str]:
        """Return the event summaries of a turn in order."""
        rows = self._rows(
            "SELECT summary FROM tool_events WHERE project_key=? AND session_id=? "
            "AND prompt_number=? AND summary IS NOT NULL ORDER BY id",
            (self.project_key, session_id, prompt_number),
        )
        return [_str(r, 0) for r in rows]

    def put_turn_summary(
        self,
        session_id: str,
        prompt_number: int,
        data: SummaryInput,
        vector: tuple[str, list[float]] | None,
    ) -> int:
        """Store (replace) a turn summary, its optional vector, and clear its queue entry."""
        self.ensure_session(session_id)
        with self.conn:
            self.conn.execute(
                "DELETE FROM turn_summaries WHERE project_key=? AND session_id=? "
                "AND prompt_number=?",
                (self.project_key, session_id, prompt_number),
            )
            cur = self.conn.execute(
                "INSERT INTO turn_summaries(project_key, session_id, prompt_number, summary, "
                "tier, synthesized, model, prompt_version, created_at_epoch) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    self.project_key,
                    session_id,
                    prompt_number,
                    data.summary,
                    data.tier,
                    int(data.tier == "synthesized"),
                    data.model,
                    data.prompt_version,
                    self.now(),
                ),
            )
            row_id = _row_id(cur.lastrowid)
            self.conn.execute(
                "DELETE FROM pending_summaries WHERE project_key=? AND session_id=? "
                "AND prompt_number=?",
                (self.project_key, session_id, prompt_number),
            )
            self._put_vector("turn_summary", row_id, data.summary, vector)
        return row_id

    def add_decision(
        self,
        decision: str,
        reason: str,
        session_id: str | None,
        source: Source,
        vector: tuple[str, list[float]] | None,
    ) -> int:
        """Insert a decision and its optional vector."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO decisions(project_key, session_id, decision, reason, source, "
                "created_at_epoch) VALUES (?,?,?,?,?,?)",
                (self.project_key, session_id, decision, reason, source, self.now()),
            )
            row_id = _row_id(cur.lastrowid)
            self._put_vector("decision", row_id, vector_text("decision", decision, reason), vector)
        return row_id

    def add_code_area(
        self,
        path: str,
        description: str,
        session_id: str | None,
        source: Source,
        vector: tuple[str, list[float]] | None,
    ) -> int:
        """Insert a code area and its optional vector."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO code_areas(project_key, session_id, file_path, description, source, "
                "created_at_epoch) VALUES (?,?,?,?,?,?)",
                (self.project_key, session_id, path, description, source, self.now()),
            )
            row_id = _row_id(cur.lastrowid)
            text = vector_text("code_area", description, path)
            self._put_vector("code_area", row_id, text, vector)
        return row_id

    def add_note(
        self,
        text: str,
        session_id: str | None,
        source: Source,
        vector: tuple[str, list[float]] | None,
    ) -> int:
        """Insert a note and its optional vector."""
        with self.conn:
            cur = self.conn.execute(
                "INSERT INTO notes(project_key, session_id, text, source, created_at_epoch) "
                "VALUES (?,?,?,?,?)",
                (self.project_key, session_id, text, source, self.now()),
            )
            row_id = _row_id(cur.lastrowid)
            self._put_vector("note", row_id, text, vector)
        return row_id

    def _put_vector(
        self,
        kind: RecallKind,
        row_id: int,
        text: str,
        vector: tuple[str, list[float]] | None,
    ) -> None:
        if vector is None or not self.vec_enabled:
            return
        model_id, values = vector
        dim = len(values)
        if dim == 0:
            return
        name = _vec_space_name(model_id, dim)
        self.conn.execute(
            "INSERT OR IGNORE INTO vec_spaces(name, model_id, dim) VALUES (?,?,?)",
            (name, model_id, dim),
        )
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {name} "
            f"USING vec0(embedding float[{dim}] distance_metric=cosine)"
        )
        old = self._maybe(
            "SELECT id FROM vectors_meta WHERE kind=? AND row_id=? AND model_id=? AND dim=?",
            (kind, row_id, model_id, dim),
        )
        if old is not None:
            self.conn.execute(f"DELETE FROM {name} WHERE rowid=?", (_int(old, 0),))
            self.conn.execute("DELETE FROM vectors_meta WHERE id=?", (_int(old, 0),))
        meta_id = _row_id(
            self.conn.execute(
                "INSERT INTO vectors_meta(project_key, kind, row_id, content_hash, model_id, dim) "
                "VALUES (?,?,?,?,?,?)",
                (self.project_key, kind, row_id, content_hash(text), model_id, dim),
            ).lastrowid
        )
        self.conn.execute(
            f"INSERT INTO {name}(rowid, embedding) VALUES (?, ?)",
            (meta_id, sqlite_vec.serialize_float32(values)),
        )

    def put_vector(
        self, kind: RecallKind, row_id: int, text: str, model_id: str, values: list[float]
    ) -> None:
        """Attach a vector to an existing row, replacing the same model's earlier one."""
        with self.conn:
            self._put_vector(kind, row_id, text, (model_id, values))

    def rows_missing_vector(self, model_id: str) -> list[tuple[RecallKind, int, str]]:
        """List (kind, id, embed text) of searchable rows with no vector under the model."""
        sources: list[tuple[RecallKind, str]] = [
            ("decision", "SELECT id, decision, reason FROM decisions"),
            ("code_area", "SELECT id, description, file_path FROM code_areas"),
            ("note", "SELECT id, text, NULL FROM notes"),
            ("turn_summary", "SELECT id, summary, NULL FROM turn_summaries"),
        ]
        missing: list[tuple[RecallKind, int, str]] = []
        for kind, select in sources:
            rows = self._rows(
                f"{select} AS t WHERE project_key=? AND NOT EXISTS ("
                "SELECT 1 FROM vectors_meta m WHERE m.kind=? AND m.row_id=t.id AND m.model_id=?"
                ") ORDER BY id",
                (self.project_key, kind, model_id),
            )
            missing.extend(
                (kind, _int(r, 0), vector_text(kind, _str(r, 1), _opt_str(r, 2))) for r in rows
            )
        return missing

    def evict_other_models(self, model_id: str, dim: int) -> int:
        """Drop vectors and vec tables of any space other than (model_id, dim)."""
        with self.conn:
            spaces = self._rows(
                "SELECT name FROM vec_spaces WHERE NOT (model_id=? AND dim=?)", (model_id, dim)
            )
            for row in spaces:
                self.conn.execute(f"DROP TABLE IF EXISTS {_str(row, 0)}")
            self.conn.execute(
                "DELETE FROM vec_spaces WHERE NOT (model_id=? AND dim=?)", (model_id, dim)
            )
            cur = self.conn.execute(
                "DELETE FROM vectors_meta WHERE project_key=? AND NOT (model_id=? AND dim=?)",
                (self.project_key, model_id, dim),
            )
        return cur.rowcount

    def vector_counts(self) -> dict[tuple[str, int], int]:
        """Return vector counts per (model_id, dim)."""
        rows = self._rows(
            "SELECT model_id, dim, COUNT(*) FROM vectors_meta WHERE project_key=? "
            "GROUP BY model_id, dim",
            (self.project_key,),
        )
        return {(_str(r, 0), _int(r, 1)): _int(r, 2) for r in rows}

    def fts_ranked(self, kind: RecallKind, match: str, limit: int) -> list[int]:
        """Return row ids for a kind ranked by bm25, ties by id."""
        table = _FTS_TABLE[kind]
        rows = self._rows(
            f"SELECT t.id FROM {table}_fts f JOIN {table} t ON t.id = f.rowid "
            f"WHERE {table}_fts MATCH ? AND t.project_key=? ORDER BY bm25({table}_fts), t.id LIMIT ?",
            (match, self.project_key, limit),
        )
        return [_int(r, 0) for r in rows]

    def vector_ranked(
        self,
        model_id: str,
        query: list[float],
        kinds: Sequence[RecallKind],
        limit: int,
        max_distance: float,
    ) -> list[VectorMatch]:
        """KNN over the single (model_id, dim) space matching the query; never mixes models."""
        if not self.vec_enabled or not query or not kinds:
            return []
        name = _vec_space_name(model_id, len(query))
        space = self._maybe(
            "SELECT name FROM vec_spaces WHERE model_id=? AND dim=?", (model_id, len(query))
        )
        if space is None:
            return []
        hits = self._rows(
            f"SELECT rowid, distance FROM {name} WHERE embedding MATCH ? AND k = ?",
            (sqlite_vec.serialize_float32(query), max(limit * 4, 50)),
        )
        distances = {_int(r, 0): _float(r, 1) for r in hits}
        if not distances:
            return []
        marks = ",".join("?" for _ in distances)
        kind_marks = ",".join("?" for _ in kinds)
        metas = self._rows(
            f"SELECT id, kind, row_id FROM vectors_meta WHERE project_key=? "
            f"AND model_id=? AND dim=? AND kind IN ({kind_marks}) AND id IN ({marks})",
            (self.project_key, model_id, len(query), *kinds, *distances),
        )
        matches = [
            VectorMatch(_kind(_str(r, 1)), _int(r, 2), distances[_int(r, 0)])
            for r in metas
            if distances[_int(r, 0)] <= max_distance
        ]
        matches.sort(key=lambda m: (m.distance, m.kind, m.row_id))
        return matches[:limit]

    def hits(self, kind: RecallKind, ids: Sequence[int]) -> dict[int, RecallHit]:
        """Load rows of one kind as unscored recall hits."""
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        params = (self.project_key, *ids)
        out: dict[int, RecallHit] = {}
        match kind:
            case "decision":
                for r in self._rows(
                    "SELECT id, session_id, decision, reason, created_at_epoch FROM decisions "
                    f"WHERE project_key=? AND id IN ({marks})",
                    params,
                ):
                    out[_int(r, 0)] = RecallHit(
                        kind=kind,
                        id=_int(r, 0),
                        project=self.project_key,
                        session_id=_opt_str(r, 1),
                        text=_str(r, 2),
                        reason=_str(r, 3),
                        score=0.0,
                        created_at=_int(r, 4),
                    )
            case "code_area":
                for r in self._rows(
                    "SELECT id, session_id, description, file_path, created_at_epoch "
                    f"FROM code_areas WHERE project_key=? AND id IN ({marks})",
                    params,
                ):
                    out[_int(r, 0)] = RecallHit(
                        kind=kind,
                        id=_int(r, 0),
                        project=self.project_key,
                        session_id=_opt_str(r, 1),
                        text=_str(r, 2),
                        path=_str(r, 3),
                        score=0.0,
                        created_at=_int(r, 4),
                    )
            case "note":
                for r in self._rows(
                    "SELECT id, session_id, text, created_at_epoch FROM notes "
                    f"WHERE project_key=? AND id IN ({marks})",
                    params,
                ):
                    out[_int(r, 0)] = RecallHit(
                        kind=kind,
                        id=_int(r, 0),
                        project=self.project_key,
                        session_id=_opt_str(r, 1),
                        text=_str(r, 2),
                        score=0.0,
                        created_at=_int(r, 3),
                    )
            case "turn_summary":
                for r in self._rows(
                    "SELECT id, session_id, summary, prompt_number, tier, synthesized, model, "
                    f"prompt_version, created_at_epoch FROM turn_summaries "
                    f"WHERE project_key=? AND id IN ({marks})",
                    params,
                ):
                    hit = RecallHit(
                        kind=kind,
                        id=_int(r, 0),
                        project=self.project_key,
                        session_id=_opt_str(r, 1),
                        text=_str(r, 2),
                        prompt_number=_int(r, 3),
                        tier=_tier(r, 4),
                        synthesized=_int(r, 5) == 1,
                        score=0.0,
                        created_at=_int(r, 8),
                    )
                    if (model := _opt_str(r, 6)) is not None:
                        hit["model"] = model
                    if (version := _opt_int(r, 7)) is not None:
                        hit["prompt_version"] = version
                    out[_int(r, 0)] = hit
        return out

    def timeline(self, session_id: str, limit: int) -> Timeline | None:
        """Per-turn summaries with their tool events; None when the session is unknown."""
        session = self._maybe(
            "SELECT status, started_at_epoch, ended_at_epoch, prompt_count FROM sessions "
            "WHERE project_key=? AND id=?",
            (self.project_key, session_id),
        )
        if session is None:
            return None
        events: dict[int, list[TimelineEvent]] = {}
        for r in self._rows(
            "SELECT prompt_number, id, tool_name, summary FROM tool_events "
            "WHERE project_key=? AND session_id=? ORDER BY id",
            (self.project_key, session_id),
        ):
            events.setdefault(_int(r, 0), []).append(
                TimelineEvent(id=_int(r, 1), tool_name=_str(r, 2), summary=_opt_str(r, 3))
            )
        turns: list[TimelineTurn] = []
        for r in self._rows(
            "SELECT prompt_number, summary, tier, synthesized, model, prompt_version "
            "FROM turn_summaries WHERE project_key=? AND session_id=? "
            "ORDER BY prompt_number LIMIT ?",
            (self.project_key, session_id, limit),
        ):
            turn = TimelineTurn(
                prompt_number=_int(r, 0),
                summary=_str(r, 1),
                tier=_tier(r, 2),
                synthesized=_int(r, 3) == 1,
                events=events.get(_int(r, 0), []),
            )
            if (model := _opt_str(r, 4)) is not None:
                turn["model"] = model
            if (version := _opt_int(r, 5)) is not None:
                turn["prompt_version"] = version
            turns.append(turn)
        return Timeline(
            session_id=session_id,
            project=self.project_key,
            status=_str(session, 0),
            started_at=_int(session, 1),
            ended_at=_opt_int(session, 2),
            prompt_count=_int(session, 3),
            turns=turns,
        )

    def event(self, event_id: int) -> EventDetail | None:
        """Raw event with payload capped at 4 KB per field; None when unknown."""
        r = self._maybe(
            "SELECT e.session_id, e.prompt_number, e.tool_name, e.created_at_epoch, e.summary, "
            "e.payload_id, p.raw_input, p.raw_output FROM tool_events e "
            "LEFT JOIN tool_event_payloads p ON p.id = e.payload_id "
            "WHERE e.project_key=? AND e.id=?",
            (self.project_key, event_id),
        )
        if r is None:
            return None
        raw_input = _opt_str(r, 6)
        raw_output = _opt_str(r, 7)
        state: PayloadState = "present"
        truncated = False
        if r[5] is None:
            state = "none"
        elif not raw_input and raw_output is None:
            state = "aged_out"
            raw_input = None
        else:
            if raw_input is not None:
                raw_input, cut_in = _truncate_bytes(raw_input, EVENT_PAYLOAD_READ_CAP)
                truncated = truncated or cut_in
            if raw_output is not None:
                raw_output, cut_out = _truncate_bytes(raw_output, EVENT_PAYLOAD_READ_CAP)
                truncated = truncated or cut_out
        return EventDetail(
            id=event_id,
            project=self.project_key,
            session_id=_str(r, 0),
            prompt_number=_int(r, 1),
            tool_name=_str(r, 2),
            created_at=_int(r, 3),
            summary=_opt_str(r, 4),
            payload=state,
            raw_input=raw_input,
            raw_output=raw_output,
            truncated=truncated,
        )

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
        case "retrieval" | "memory_recall" | "progressive_disclosure" | "turn_summarization":
            return value
        case _:
            return None


def _kind(value: str) -> RecallKind:
    match value:
        case "decision" | "code_area" | "note" | "turn_summary":
            return value
        case _:
            raise MemoryRowError(f"unknown kind: {value!r}")


def _float(row: Row, index: int) -> float:
    value = row[index]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise MemoryRowError(f"column {index}: expected number, got {type(value).__name__}")


def _row_id(value: int | None) -> int:
    if value is None:
        raise MemoryRowError("insert returned no rowid")
    return value


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
    if version != 0:
        raise MemorySchemaError(f"memory.db schema {version}, expected {SCHEMA_VERSION}")
    with conn:
        for statement in _BASE_SCHEMA:
            conn.execute(statement)
        for table, columns in _TABLES_WITH_FTS.items():
            for statement in _fts_statements(table, columns):
                conn.execute(statement)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
