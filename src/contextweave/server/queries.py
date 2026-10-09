import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from contextweave.lang.registry import GROUPS, Group

type SqlValue = str | int | None

_SELECTS = (
    (
        "SELECT f.path, s.name, s.qualified_name, s.kind, s.start_byte, s.start_line, f.lang, f.grp"
        " FROM symbols s JOIN files f ON f.id = s.file_id WHERE ",
        "(instr(lower(s.name), ?1) > 0 OR instr(lower(s.qualified_name), ?1) > 0)",
        "s.kind",
    ),
    (
        "SELECT f.path, n.key_path, n.key_path, n.kind, n.start_byte, n.start_line, f.lang, f.grp"
        " FROM schema_nodes n JOIN files f ON f.id = n.file_id WHERE ",
        "instr(lower(n.key_path), ?1) > 0",
        "n.kind",
    ),
    (
        "SELECT f.path, d.title, d.slug, d.kind, d.start_byte, d.start_line, f.lang, f.grp"
        " FROM doc_nodes d JOIN files f ON f.id = d.file_id WHERE ",
        "(instr(lower(d.title), ?1) > 0 OR instr(lower(d.slug), ?1) > 0)",
        "d.kind",
    ),
)


@dataclass(frozen=True, slots=True)
class SearchRow:
    """Address of one matching node."""

    path: str
    name: str
    qualified_name: str
    kind: str
    start_byte: int
    start_line: int
    lang: str | None
    group: Group | None


@dataclass(frozen=True, slots=True)
class HashRow:
    """A symbol carrying a given content hash."""

    path: str
    qualified_name: str
    start_byte: int


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _number(value: object) -> int:
    return value if isinstance(value, int) else 0


def _optional(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _group(value: object) -> Group | None:
    return next((group for group in GROUPS if group == value), None)


class Queries:
    """Read-only SQL over the project db for cross-file lookups the Store does not expose."""

    def __init__(self, db_path: Path | None) -> None:
        self._connection: sqlite3.Connection | None = None
        if db_path is not None and db_path.is_file():
            try:
                self._connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
            except sqlite3.Error:
                self._connection = None

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def _rows(self, sql: str, params: Sequence[SqlValue]) -> list[tuple[object, ...]]:
        if self._connection is None:
            return []
        try:
            rows: list[tuple[object, ...]] = self._connection.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        return rows

    def search(
        self, needle: str, *, kind: str | None, lang: str | None, group: Group | None
    ) -> list[SearchRow]:
        """Nodes whose name or qualified name may contain needle; callers refilter exactly."""
        filters = [(lang, "f.lang"), (group, "f.grp")]
        rows: list[tuple[object, ...]] = []
        for head, match, kind_column in _SELECTS:
            checks = [(kind, kind_column), *filters]
            active = [(value, column) for value, column in checks if value is not None]
            params: list[SqlValue] = [needle.lower()]
            clauses = [match if needle.isascii() else "?1 IS NOT NULL"]
            for value, column in active:
                params.append(value)
                clauses.append(f"{column} = ?{len(params)}")
            rows.extend(self._rows(head + " AND ".join(clauses), params))
        return [
            SearchRow(
                _text(row[0]),
                _text(row[1]),
                _text(row[2]),
                _text(row[3]),
                _number(row[4]),
                _number(row[5]),
                _optional(row[6]),
                _group(row[7]),
            )
            for row in rows
        ]

    def symbols_with_hash(self, content_hash: str) -> list[HashRow]:
        """Symbols whose 12-hex content hash prefix matches."""
        rows = self._rows(
            "SELECT f.path, s.qualified_name, s.start_byte FROM symbols s"
            " JOIN files f ON f.id = s.file_id WHERE substr(s.content_hash, 1, 12) = ?"
            " ORDER BY f.path, s.start_byte",
            [content_hash],
        )
        return [HashRow(_text(row[0]), _text(row[1]), _number(row[2])) for row in rows]
