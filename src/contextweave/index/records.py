from dataclasses import dataclass
from typing import Literal

from contextweave.lang.model import LangError
from contextweave.lang.registry import Group, Profile

type FileProfile = Profile | Literal["text"]


@dataclass(frozen=True, slots=True)
class FileStamp:
    """The stat identity of a file's bytes (V3)."""

    mtime_ns: int
    size: int


@dataclass(frozen=True, slots=True)
class FileRecord:
    """An indexed file; lang and group are None when no registry row matches."""

    id: int
    path: str
    lang: str | None
    group: Group | None
    profile: FileProfile
    stamp: FileStamp
    content_hash: str
    indexed_at: int


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """A stored definition; ids are 0 when the record was never persisted."""

    id: int
    file_id: int
    path: str
    name: str
    qualified_name: str
    kind: str
    parent: str | None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    signature: str
    docstring: str | None
    content_hash: str


@dataclass(frozen=True, slots=True)
class ImportRecord:
    """A stored module reference."""

    id: int
    file_id: int
    path: str
    module: str
    names: tuple[str, ...]
    alias: str | None
    line: int
    kind: str


@dataclass(frozen=True, slots=True)
class SchemaNodeRecord:
    """A stored key path of a data file."""

    id: int
    file_id: int
    path: str
    key_path: str
    kind: str
    parent: str | None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    value_preview: str | None
    comment: str | None
    children_count: int
    truncated_subtree: bool
    uniformity: str | None


@dataclass(frozen=True, slots=True)
class DocNodeRecord:
    """A stored document structure node."""

    id: int
    file_id: int
    path: str
    title: str
    slug: str
    level: int
    kind: str
    parent: str | None
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    info: str | None


@dataclass(frozen=True, slots=True)
class FileData:
    """A file with every structural row derived from it."""

    file: FileRecord
    symbols: tuple[SymbolRecord, ...]
    imports: tuple[ImportRecord, ...]
    schema_nodes: tuple[SchemaNodeRecord, ...]
    doc_nodes: tuple[DocNodeRecord, ...]
    errors: tuple[LangError, ...] = ()


@dataclass(frozen=True, slots=True)
class Fresh:
    """Stat matches the index."""


@dataclass(frozen=True, slots=True)
class Changed:
    """Stat differs from the index, or the file is not indexed yet."""

    stamp: FileStamp


@dataclass(frozen=True, slots=True)
class Gone:
    """The file no longer exists on disk."""


type StatCheck = Fresh | Changed | Gone


@dataclass(frozen=True, slots=True)
class RefreshResult:
    """Outcome of a stat-check-and-reparse; data is None only when the file is gone."""

    check: StatCheck
    data: FileData | None
    persisted: bool
