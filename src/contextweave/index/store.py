import hashlib
import json
import os
import sqlite3
import sys
import time
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from importlib.resources import files as package_files
from pathlib import Path
from types import TracebackType
from typing import Self, cast

from contextweave.core.config import ChunkConfig
from contextweave.index.chunk import chunk_file
from contextweave.index.chunk_tokens import ChunkTokenizer
from contextweave.index.records import (
    BodyMode,
    Changed,
    Chunk,
    DocNodeRecord,
    FileData,
    FileProfile,
    FileRecord,
    FileStamp,
    Fresh,
    Gone,
    ImportRecord,
    RefreshResult,
    SchemaNodeRecord,
    StatCheck,
    SymbolRecord,
)
from contextweave.index.root import ensure_self_ignore, state_dir
from contextweave.index.vec_load import load_vec
from contextweave.index.walk import BINARY_SNIFF_BYTES
from contextweave.lang.extract_doc import extract_doc
from contextweave.lang.extract_schema import extract_schema
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.model import Import, LangError
from contextweave.lang.parse import ParseResult, parse_bytes
from contextweave.lang.registry import GROUPS, PROFILES, Group, spec_for_path

SCHEMA_VERSION = 2
DB_NAME = "index.db"
LOCK_NAME = "index.lock"
GENERATION_KEY = "generation"
LOCK_POLL_S = 0.05

type SqlValue = str | int | float | bytes | None
type Row = tuple[object, ...]


class SchemaMismatchError(Exception):
    """The database was written by another schema version; the caller rebuilds it."""

    def __init__(self, path: Path, found: int, expected: int) -> None:
        super().__init__(f"{path}: user_version {found}, expected {expected}")
        self.path = path
        self.found = found
        self.expected = expected


class ReadOnlyStoreError(Exception):
    """A write was attempted through a read-only store."""


class LockTimeoutError(Exception):
    """The single-writer advisory lock could not be taken in time."""


if sys.platform == "win32":

    def _try_lock(descriptor: int) -> bool:
        return True

    def _unlock(descriptor: int) -> None:
        return None

else:
    import fcntl

    def _try_lock(descriptor: int) -> bool:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


def hash_bytes(source: bytes) -> str:
    """Hex sha256 of bytes."""
    return hashlib.sha256(source).hexdigest()


def build_file_data(
    path: str, parse_result: ParseResult, source: bytes, stamp: FileStamp
) -> FileData:
    """Turn a parse into unpersisted records; ids are 0."""
    parsed = parse_result.parsed
    spec = parsed.spec if parsed is not None else spec_for_path(path)
    profile: FileProfile = spec.profile if spec is not None and parsed is not None else "text"
    file = FileRecord(
        id=0,
        path=path,
        lang=None if spec is None else spec.lang,
        group=None if spec is None else spec.group,
        profile=profile,
        stamp=stamp,
        content_hash=hash_bytes(source),
        indexed_at=int(time.time()),
    )
    errors: list[LangError] = list(parse_result.errors)
    symbols: list[SymbolRecord] = []
    imports: list[ImportRecord] = []
    schema_nodes: list[SchemaNodeRecord] = []
    doc_nodes: list[DocNodeRecord] = []
    if parsed is not None:
        if parsed.spec.profile in ("symbols", "defs"):
            found = extract_symbols(parsed)
            errors.extend(found.errors)
            symbols = [
                SymbolRecord(
                    0,
                    0,
                    path,
                    item.name,
                    item.qualified_name,
                    item.kind,
                    item.parent,
                    item.start_byte,
                    item.end_byte,
                    item.start_line,
                    item.end_line,
                    item.signature,
                    item.docstring,
                    hash_bytes(source[item.start_byte : item.end_byte]),
                )
                for item in found.symbols
            ]
            imports = [_import_record(path, item) for item in found.imports]
        elif parsed.spec.profile == "schema":
            tree = extract_schema(parsed)
            errors.extend(tree.errors)
            schema_nodes = [
                SchemaNodeRecord(
                    0,
                    0,
                    path,
                    item.key_path,
                    item.kind,
                    item.parent,
                    item.start_byte,
                    item.end_byte,
                    item.start_line,
                    item.end_line,
                    item.value_preview,
                    item.comment,
                    item.children_count,
                    item.truncated_subtree,
                    item.uniformity,
                )
                for item in tree.nodes
            ]
        else:
            outline = extract_doc(parsed)
            errors.extend(outline.errors)
            doc_nodes = [
                DocNodeRecord(
                    0,
                    0,
                    path,
                    item.title,
                    item.slug,
                    item.level,
                    item.kind,
                    item.parent,
                    item.start_byte,
                    item.end_byte,
                    item.start_line,
                    item.end_line,
                    item.info,
                )
                for item in outline.nodes
            ]
            imports = [_import_record(path, item) for item in outline.links]
    return FileData(
        file, tuple(symbols), tuple(imports), tuple(schema_nodes), tuple(doc_nodes), tuple(errors)
    )


class Store:
    """The project index database: typed records in, typed records out."""

    def __init__(
        self,
        root: Path,
        connection: sqlite3.Connection,
        *,
        readonly: bool,
        path: Path | None,
        chunk_config: ChunkConfig | None = None,
        tokenizer: ChunkTokenizer | None = None,
    ) -> None:
        self.root = root
        self.chunk_config = chunk_config or ChunkConfig()
        self.tokenizer = tokenizer
        self.readonly = readonly
        self.path = path
        self._connection = connection
        self._lock_depth = 0
        self._lock_descriptor: int | None = None

    @classmethod
    def open(
        cls,
        root: Path,
        *,
        readonly: bool = False,
        chunk_config: ChunkConfig | None = None,
        tokenizer: ChunkTokenizer | None = None,
    ) -> Self:
        """Open or create the project db; a version mismatch raises SchemaMismatchError."""
        if readonly:
            return cls._open_readonly(root, chunk_config, tokenizer)
        directory = ensure_self_ignore(root)
        path = directory / DB_NAME
        connection = _connect(str(path))
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        _prepare(connection, path)
        return cls(
            root,
            connection,
            readonly=False,
            path=path,
            chunk_config=chunk_config,
            tokenizer=tokenizer,
        )

    @classmethod
    def _open_readonly(
        cls, root: Path, chunk_config: ChunkConfig | None, tokenizer: ChunkTokenizer | None
    ) -> Self:
        path = state_dir(root) / DB_NAME
        if not path.is_file():
            connection = _connect(":memory:")
            _apply_schema(connection)
            return cls(
                root,
                connection,
                readonly=True,
                path=None,
                chunk_config=chunk_config,
                tokenizer=tokenizer,
            )
        connection = _connect(f"{path.as_uri()}?mode=ro", uri=True)
        _prepare(connection, path)
        return cls(
            root,
            connection,
            readonly=True,
            path=path,
            chunk_config=chunk_config,
            tokenizer=tokenizer,
        )

    def close(self) -> None:
        """Release the connection and any held lock."""
        self._connection.close()
        self._release_lock()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def generation(self) -> int:
        """Counter bumped by every write."""
        rows = self._query("SELECT value FROM meta WHERE key = ?", (GENERATION_KEY,))
        return _int(rows[0][0]) if rows else 0

    @contextmanager
    def writer_lock(self, timeout_s: float = 10.0) -> Generator[None]:
        """Advisory single-writer lock (V21); reentrant, no-op off POSIX."""
        self._require_writable()
        if self._lock_depth == 0:
            self._acquire_lock(timeout_s)
        self._lock_depth += 1
        try:
            yield
        finally:
            self._lock_depth -= 1
            if self._lock_depth == 0:
                self._release_lock()

    def upsert_file(
        self, path: str, parse_result: ParseResult, source: bytes, stamp: FileStamp
    ) -> FileData:
        """Replace every row of one file in a single transaction."""
        self._require_writable()
        data = build_file_data(path, parse_result, source, stamp)
        with self._transaction():
            existing = self._query("SELECT id FROM files WHERE path = ?", (path,))
            if existing:
                file_id = _int(existing[0][0])
                self._clear_children(file_id)
                self._connection.execute(
                    "UPDATE files SET lang = ?, grp = ?, profile = ?, mtime_ns = ?, size = ?,"
                    " content_hash = ?, indexed_at = ? WHERE id = ?",
                    (
                        data.file.lang,
                        data.file.group,
                        data.file.profile,
                        stamp.mtime_ns,
                        stamp.size,
                        data.file.content_hash,
                        data.file.indexed_at,
                        file_id,
                    ),
                )
            else:
                cursor = self._connection.execute(
                    "INSERT INTO files (path, lang, grp, profile, mtime_ns, size, content_hash,"
                    " indexed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        path,
                        data.file.lang,
                        data.file.group,
                        data.file.profile,
                        stamp.mtime_ns,
                        stamp.size,
                        data.file.content_hash,
                        data.file.indexed_at,
                    ),
                )
                file_id = _int(cursor.lastrowid)
            self._insert_children(file_id, data)
            self._write_chunks(file_id, path, parse_result, source)
            self._bump_generation()
        stored = self.get_data(path)
        if stored is None:
            raise RuntimeError(f"upsert of {path} left no row")
        return replace(stored, errors=data.errors)

    def touch_file(self, path: str, stamp: FileStamp) -> None:
        """Record a new stat identity for unchanged bytes."""
        self._require_writable()
        with self._transaction():
            self._connection.execute(
                "UPDATE files SET mtime_ns = ?, size = ? WHERE path = ?",
                (stamp.mtime_ns, stamp.size, path),
            )
            self._bump_generation()

    def delete_file(self, path: str) -> bool:
        """Remove a file and everything derived from it; False when it was not indexed."""
        self._require_writable()
        with self._transaction():
            rows = self._query("SELECT id FROM files WHERE path = ?", (path,))
            if not rows:
                return False
            file_id = _int(rows[0][0])
            self._clear_children(file_id)
            self._connection.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._bump_generation()
        return True

    def get_file(self, path: str) -> FileRecord | None:
        """The indexed row of a path."""
        rows = self._query(f"{_FILE_SELECT} WHERE path = ?", (path,))
        return _file_record(rows[0]) if rows else None

    def files(self) -> tuple[FileRecord, ...]:
        """Every indexed file sorted by path."""
        return tuple(_file_record(row) for row in self._query(f"{_FILE_SELECT} ORDER BY path"))

    def get_data(self, path: str) -> FileData | None:
        """A file with all its structural rows."""
        file = self.get_file(path)
        if file is None:
            return None
        return FileData(
            file,
            self.symbols_in_file(path),
            self._imports_in_file(file.id),
            self._schema_nodes_in_file(file.id),
            self._doc_nodes_in_file(file.id),
        )

    def chunks_in_file(self, path: str) -> tuple[Chunk, ...]:
        """Chunks of one file in source order, with served text and symbol membership."""
        rows = self._query(
            "SELECT c.id, c.file_id, c.kind, c.start_byte, c.end_byte, c.context_header,"
            " c.body_mode, c.oversize, c.content_hash, c.part,"
            " (SELECT body FROM fts_code WHERE rowid = c.id)"
            " FROM chunks c JOIN files f ON f.id = c.file_id WHERE f.path = ?"
            " ORDER BY c.start_byte, c.end_byte, c.part, c.id",
            (path,),
        )
        return tuple(self._chunk_record(row) for row in rows)

    def symbols_by_name(self, name: str) -> tuple[SymbolRecord, ...]:
        """Symbols with an exact simple name."""
        return self._symbols("WHERE s.name = ?", (name,))

    def symbols_by_qualified_name(self, qualified_name: str) -> tuple[SymbolRecord, ...]:
        """Symbols with an exact qualified name."""
        return self._symbols("WHERE s.qualified_name = ?", (qualified_name,))

    def symbols_in_file(self, path: str) -> tuple[SymbolRecord, ...]:
        """Symbols of one file in source order."""
        return self._symbols("WHERE f.path = ?", (path,))

    def query(self, sql: str, params: Sequence[SqlValue] = ()) -> list[Row]:
        """Run a read-only SELECT and return raw rows."""
        return self._query(sql, params)

    def execute(self, sql: str, params: Sequence[SqlValue] = ()) -> None:
        """Run one write statement; callers wrap multi-statement work in write_transaction."""
        self._require_writable()
        self._connection.execute(sql, params)

    @contextmanager
    def write_transaction(self) -> Generator[None]:
        """Run the block in one immediate transaction."""
        self._require_writable()
        with self._transaction():
            yield

    def stat_check(self, path: str) -> StatCheck:
        """Compare the file on disk with the index by mtime_ns and size (V3)."""
        try:
            stat = (self.root / path).stat()
        except OSError:
            return Gone()
        stamp = FileStamp(stat.st_mtime_ns, stat.st_size)
        record = self.get_file(path)
        return Fresh() if record is not None and record.stamp == stamp else Changed(stamp)

    def refresh(self, path: str) -> RefreshResult:
        """Stat-check, then reparse and upsert on change; read-only stores reparse in memory."""
        check = self.stat_check(path)
        if isinstance(check, Fresh):
            return RefreshResult(check, self.get_data(path), persisted=False)
        if isinstance(check, Gone):
            return RefreshResult(check, None, persisted=self._drop(path))
        source = self._read(path)
        if source is None:
            return RefreshResult(Gone(), None, persisted=self._drop(path))
        return self._reparse(path, check, source)

    def _reparse(self, path: str, check: Changed, source: bytes) -> RefreshResult:
        existing = self.get_data(path)
        if existing is not None and existing.file.content_hash == hash_bytes(source):
            if self.readonly:
                moved = replace(existing.file, stamp=check.stamp)
                return RefreshResult(check, replace(existing, file=moved), persisted=False)
            self.touch_file(path, check.stamp)
            return RefreshResult(check, self.get_data(path), persisted=True)
        result = parse_bytes(path, source)
        if self.readonly:
            return RefreshResult(
                check, build_file_data(path, result, source, check.stamp), persisted=False
            )
        return RefreshResult(check, self.upsert_file(path, result, source, check.stamp), True)

    def _read(self, path: str) -> bytes | None:
        try:
            source = (self.root / path).read_bytes()
        except OSError:
            return None
        return None if b"\0" in source[:BINARY_SNIFF_BYTES] else source

    def _drop(self, path: str) -> bool:
        return not self.readonly and self.delete_file(path)

    def _require_writable(self) -> None:
        if self.readonly:
            raise ReadOnlyStoreError(str(self.path or self.root))

    def _query(self, sql: str, params: Sequence[SqlValue] = ()) -> list[Row]:
        rows: list[Row] = self._connection.execute(sql, params).fetchall()
        return rows

    @contextmanager
    def _transaction(self) -> Generator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        self._connection.execute("COMMIT")

    def _bump_generation(self) -> None:
        self._connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, 1)"
            " ON CONFLICT (key) DO UPDATE SET value = value + 1",
            (GENERATION_KEY,),
        )

    def _clear_children(self, file_id: int) -> None:
        self._connection.execute(
            "DELETE FROM fts_code WHERE rowid IN (SELECT id FROM chunks WHERE file_id = ?)",
            (file_id,),
        )
        for table in ("chunks", "symbols", "imports", "schema_nodes", "doc_nodes", "file_scopes"):
            self._connection.execute(f"DELETE FROM {table} WHERE file_id = ?", (file_id,))

    def _insert_children(self, file_id: int, data: FileData) -> None:
        self._connection.executemany(
            "INSERT INTO symbols (file_id, name, qualified_name, kind, parent, start_byte,"
            " end_byte, start_line, end_line, signature, docstring, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    file_id,
                    s.name,
                    s.qualified_name,
                    s.kind,
                    s.parent,
                    s.start_byte,
                    s.end_byte,
                    s.start_line,
                    s.end_line,
                    s.signature,
                    s.docstring,
                    s.content_hash,
                )
                for s in data.symbols
            ],
        )
        self._connection.executemany(
            "INSERT INTO imports (file_id, module, names, alias, line, kind)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            [
                (file_id, i.module, json.dumps(list(i.names)), i.alias, i.line, i.kind)
                for i in data.imports
            ],
        )
        self._connection.executemany(
            "INSERT INTO schema_nodes (file_id, key_path, kind, parent, start_byte, end_byte,"
            " start_line, end_line, value_preview, comment, children_count, truncated_subtree,"
            " uniformity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    file_id,
                    n.key_path,
                    n.kind,
                    n.parent,
                    n.start_byte,
                    n.end_byte,
                    n.start_line,
                    n.end_line,
                    n.value_preview,
                    n.comment,
                    n.children_count,
                    int(n.truncated_subtree),
                    n.uniformity,
                )
                for n in data.schema_nodes
            ],
        )
        self._connection.executemany(
            "INSERT INTO doc_nodes (file_id, title, slug, level, kind, parent, start_byte,"
            " end_byte, start_line, end_line, info) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    file_id,
                    d.title,
                    d.slug,
                    d.level,
                    d.kind,
                    d.parent,
                    d.start_byte,
                    d.end_byte,
                    d.start_line,
                    d.end_line,
                    d.info,
                )
                for d in data.doc_nodes
            ],
        )

    def _write_chunks(
        self, file_id: int, path: str, parse_result: ParseResult, source: bytes
    ) -> None:
        symbols = self.symbols_in_file(path)
        chunks = chunk_file(
            source,
            parse_result.parsed,
            symbols,
            self._schema_nodes_in_file(file_id),
            self.chunk_config,
            self.tokenizer,
            file_id=file_id,
        )
        by_id = {symbol.id: symbol for symbol in symbols}
        for chunk in chunks:
            cursor = self._connection.execute(
                "INSERT INTO chunks (file_id, kind, start_byte, end_byte, context_header,"
                " body_mode, oversize, content_hash, part) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id,
                    chunk.kind,
                    chunk.start_byte,
                    chunk.end_byte,
                    chunk.context_header,
                    chunk.body_mode,
                    int(chunk.oversize),
                    chunk.content_hash,
                    chunk.part,
                ),
            )
            chunk_id = _int(cursor.lastrowid)
            self._connection.executemany(
                "INSERT INTO chunk_symbols (chunk_id, symbol_id) VALUES (?, ?)",
                [(chunk_id, symbol_id) for symbol_id in chunk.symbol_ids],
            )
            members = [by_id[symbol_id] for symbol_id in chunk.symbol_ids]
            self._connection.execute(
                "INSERT INTO fts_code (rowid, name, qualified_name, signature, docstring, body,"
                " annotation, path) VALUES (?, ?, ?, ?, ?, ?, '', ?)",
                (
                    chunk_id,
                    " ".join(m.name for m in members),
                    " ".join(m.qualified_name for m in members),
                    "\n".join(m.signature for m in members),
                    "\n".join(m.docstring for m in members if m.docstring is not None),
                    chunk.text,
                    path,
                ),
            )

    def _chunk_record(self, row: Row) -> Chunk:
        chunk_id = _int(row[0])
        members = self._query(
            "SELECT s.id, s.qualified_name FROM chunk_symbols cs"
            " JOIN symbols s ON s.id = cs.symbol_id WHERE cs.chunk_id = ? ORDER BY cs.rowid",
            (chunk_id,),
        )
        return Chunk(
            id=chunk_id,
            file_id=_int(row[1]),
            kind=_str(row[2]),
            start_byte=_int(row[3]),
            end_byte=_int(row[4]),
            context_header=_str(row[5]),
            body_mode=_body_mode(row[6]),
            oversize=_int(row[7]) != 0,
            content_hash=_str(row[8]),
            symbol_ids=tuple(_int(member[0]) for member in members),
            qualified_names=tuple(_str(member[1]) for member in members),
            part=None if row[9] is None else _int(row[9]),
            text=_str(row[10]),
        )

    def _symbols(self, where: str, params: Sequence[SqlValue]) -> tuple[SymbolRecord, ...]:
        rows = self._query(
            "SELECT s.id, s.file_id, f.path, s.name, s.qualified_name, s.kind, s.parent,"
            " s.start_byte, s.end_byte, s.start_line, s.end_line, s.signature, s.docstring,"
            " s.content_hash FROM symbols s JOIN files f ON f.id = s.file_id"
            f" {where} ORDER BY f.path, s.start_byte, s.id",
            params,
        )
        return tuple(_symbol_record(row) for row in rows)

    def _imports_in_file(self, file_id: int) -> tuple[ImportRecord, ...]:
        rows = self._query(
            "SELECT i.id, i.file_id, f.path, i.module, i.names, i.alias, i.line, i.kind"
            " FROM imports i JOIN files f ON f.id = i.file_id WHERE i.file_id = ? ORDER BY i.id",
            (file_id,),
        )
        return tuple(_import_row(row) for row in rows)

    def _schema_nodes_in_file(self, file_id: int) -> tuple[SchemaNodeRecord, ...]:
        rows = self._query(
            "SELECT n.id, n.file_id, f.path, n.key_path, n.kind, n.parent, n.start_byte,"
            " n.end_byte, n.start_line, n.end_line, n.value_preview, n.comment,"
            " n.children_count, n.truncated_subtree, n.uniformity"
            " FROM schema_nodes n JOIN files f ON f.id = n.file_id WHERE n.file_id = ?"
            " ORDER BY n.id",
            (file_id,),
        )
        return tuple(_schema_row(row) for row in rows)

    def _doc_nodes_in_file(self, file_id: int) -> tuple[DocNodeRecord, ...]:
        rows = self._query(
            "SELECT d.id, d.file_id, f.path, d.title, d.slug, d.level, d.kind, d.parent,"
            " d.start_byte, d.end_byte, d.start_line, d.end_line, d.info"
            " FROM doc_nodes d JOIN files f ON f.id = d.file_id WHERE d.file_id = ?"
            " ORDER BY d.id",
            (file_id,),
        )
        return tuple(_doc_row(row) for row in rows)

    def _acquire_lock(self, timeout_s: float) -> None:
        directory = state_dir(self.root)
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + timeout_s
        while not _try_lock(descriptor):
            if time.monotonic() >= deadline:
                os.close(descriptor)
                raise LockTimeoutError(str(directory / LOCK_NAME))
            time.sleep(LOCK_POLL_S)
        self._lock_descriptor = descriptor

    def _release_lock(self) -> None:
        descriptor, self._lock_descriptor = self._lock_descriptor, None
        if descriptor is not None:
            _unlock(descriptor)
            os.close(descriptor)


_FILE_SELECT = (
    "SELECT id, path, lang, grp, profile, mtime_ns, size, content_hash, indexed_at FROM files"
)


def _connect(target: str, *, uri: bool = False) -> sqlite3.Connection:
    connection = sqlite3.connect(target, uri=uri, isolation_level=None)
    load_vec(connection)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _prepare(connection: sqlite3.Connection, path: Path) -> None:
    tables = _int(connection.execute("SELECT count(*) FROM sqlite_master").fetchone()[0])
    if tables == 0:
        _apply_schema(connection)
        return
    found = _int(connection.execute("PRAGMA user_version").fetchone()[0])
    if found != SCHEMA_VERSION:
        connection.close()
        raise SchemaMismatchError(path, found, SCHEMA_VERSION)


def _apply_schema(connection: sqlite3.Connection) -> None:
    schema = package_files("contextweave.index").joinpath("schema.sql").read_text(encoding="utf-8")
    connection.executescript(f"BEGIN;\n{schema}\nPRAGMA user_version = {SCHEMA_VERSION};\nCOMMIT;")


def _import_record(path: str, item: Import) -> ImportRecord:
    return ImportRecord(0, 0, path, item.module, item.names, item.alias, item.line, item.kind)


def _int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    return value


def _str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def _opt_str(value: object) -> str | None:
    return None if value is None else _str(value)


def _group(value: object) -> Group | None:
    if value is None:
        return None
    text = _str(value)
    for group in GROUPS:
        if group == text:
            return group
    raise ValueError(f"unknown group {text}")


def _profile(value: object) -> FileProfile:
    text = _str(value)
    if text == "text":
        return "text"
    for profile in PROFILES:
        if profile == text:
            return profile
    raise ValueError(f"unknown profile {text}")


def _body_mode(value: object) -> BodyMode:
    text = _str(value)
    if text == "schema":
        return "schema"
    if text == "bytes":
        return "bytes"
    raise ValueError(f"unknown body mode {text}")


def _names(value: object) -> tuple[str, ...]:
    decoded: object = json.loads(_str(value))
    if not isinstance(decoded, list):
        raise TypeError("imports.names is not a list")
    items = cast(list[object], decoded)  # json.loads yields list[Unknown]; elements checked below
    return tuple(_str(item) for item in items)


def _file_record(row: Row) -> FileRecord:
    return FileRecord(
        id=_int(row[0]),
        path=_str(row[1]),
        lang=_opt_str(row[2]),
        group=_group(row[3]),
        profile=_profile(row[4]),
        stamp=FileStamp(_int(row[5]), _int(row[6])),
        content_hash=_str(row[7]),
        indexed_at=_int(row[8]),
    )


def _symbol_record(row: Row) -> SymbolRecord:
    return SymbolRecord(
        _int(row[0]),
        _int(row[1]),
        _str(row[2]),
        _str(row[3]),
        _str(row[4]),
        _str(row[5]),
        _opt_str(row[6]),
        _int(row[7]),
        _int(row[8]),
        _int(row[9]),
        _int(row[10]),
        _str(row[11]),
        _opt_str(row[12]),
        _str(row[13]),
    )


def _import_row(row: Row) -> ImportRecord:
    return ImportRecord(
        _int(row[0]),
        _int(row[1]),
        _str(row[2]),
        _str(row[3]),
        _names(row[4]),
        _opt_str(row[5]),
        _int(row[6]),
        _str(row[7]),
    )


def _schema_row(row: Row) -> SchemaNodeRecord:
    return SchemaNodeRecord(
        _int(row[0]),
        _int(row[1]),
        _str(row[2]),
        _str(row[3]),
        _str(row[4]),
        _opt_str(row[5]),
        _int(row[6]),
        _int(row[7]),
        _int(row[8]),
        _int(row[9]),
        _opt_str(row[10]),
        _opt_str(row[11]),
        _int(row[12]),
        _int(row[13]) != 0,
        _opt_str(row[14]),
    )


def _doc_row(row: Row) -> DocNodeRecord:
    return DocNodeRecord(
        _int(row[0]),
        _int(row[1]),
        _str(row[2]),
        _str(row[3]),
        _str(row[4]),
        _int(row[5]),
        _str(row[6]),
        _opt_str(row[7]),
        _int(row[8]),
        _int(row[9]),
        _int(row[10]),
        _int(row[11]),
        _opt_str(row[12]),
    )
