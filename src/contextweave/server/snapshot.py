import os
from pathlib import Path

from contextweave.core.errors import Envelope, ErrorEntry
from contextweave.core.refs import ByteAnchor, Ref, RefError, SymbolAnchor, encode_ref
from contextweave.index.records import FileData, FileStamp, Fresh
from contextweave.index.store import Store, build_file_data, hash_bytes
from contextweave.lang.model import LangError
from contextweave.lang.parse import parse_bytes
from contextweave.server.nodes import Node, nodes_of, short_hash
from contextweave.server.queries import Queries

LOCK_TIMEOUT_S = 2.0


def lang_error_entry(error: LangError) -> ErrorEntry:
    """Convert a non-fatal language problem to an envelope error."""
    entry = ErrorEntry(code=error.code)
    if error.detail is not None:
        entry["message"] = error.detail
    if error.path is not None:
        entry["detail"] = {"path": error.path}
    return entry


class Snapshot:
    """One call's view of a project: every path stat-checked and refreshed before use (V3)."""

    def __init__(self, store: Store, env: Envelope, key: str) -> None:
        self.store = store
        self.env = env
        self.key = key
        self.queries = Queries(store.path)
        self._data: dict[str, FileData | None] = {}
        self._sources: dict[str, bytes] = {}
        self._nodes: dict[str, tuple[Node, ...]] = {}
        self._overlay: set[str] = set()
        self._synced_all = False

    def close(self) -> None:
        self.queries.close()

    @property
    def overlay(self) -> frozenset[str]:
        """Paths whose truth lives only in memory, so SQL rows for them are stale."""
        return frozenset(self._overlay)

    def normalize(self, raw: str) -> str | None:
        """Project-relative posix path, or None with an outside_project error."""
        root = os.path.normpath(self.store.root)
        candidate = os.path.normpath(
            raw if Path(raw).is_absolute() else os.path.join(root, raw)  # noqa: PTH118
        )
        relative = os.path.relpath(candidate, root)
        if relative == os.pardir or relative.startswith(os.pardir + os.sep) or relative == ".":
            self.env.add_error(
                ErrorEntry(code="outside_project", message=f"{raw} is not inside the project")
            )
            return None
        return Path(relative).as_posix()

    def source(self, path: str) -> bytes:
        """Current bytes of a file; empty when unreadable."""
        if path not in self._sources:
            try:
                self._sources[path] = (self.store.root / path).read_bytes()
            except OSError:
                self._sources[path] = b""
        return self._sources[path]

    def exists(self, path: str) -> bool:
        return (self.store.root / path).is_file()

    def sync(self, path: str) -> FileData | None:
        """Stat-check and refresh one path; None when gone, binary or unparseable as a file."""
        if path in self._data:
            return self._data[path]
        data = self._refresh(path)
        if data is not None:
            data = self._verified(path, data)
        self._data[path] = data
        return data

    def sync_all(self) -> None:
        """Refresh every indexed path."""
        if self._synced_all:
            return
        self._synced_all = True
        for record in self.store.files():
            self.sync(record.path)

    def nodes(self, path: str) -> tuple[Node, ...]:
        if path not in self._nodes:
            data = self.sync(path)
            self._nodes[path] = () if data is None else nodes_of(data, self.source(path))
        return self._nodes[path]

    def symbol_ref(self, node: Node) -> str:
        """Ref of a node, falling back to its byte range when the name cannot anchor."""
        try:
            return encode_ref(
                Ref(self.key, node.path, SymbolAnchor(node.qualified_name), node.content_hash)
            )
        except RefError:
            return self.byte_ref(node.path, node.start_byte, node.end_byte)

    def byte_ref(self, path: str, start: int, end: int) -> str:
        source = self.source(path)
        return encode_ref(
            Ref(self.key, path, ByteAnchor(start, end), short_hash(source[start:end]))
        )

    def _refresh(self, path: str) -> FileData | None:
        try:
            if isinstance(self.store.stat_check(path), Fresh):
                return self.store.get_data(path)
            if self.store.readonly:
                result = self.store.refresh(path)
                if not result.persisted:
                    self._overlay.add(path)
                return result.data
            with self.store.writer_lock(LOCK_TIMEOUT_S):
                return self.store.refresh(path).data
        except Exception as exc:
            self.env.add_error(
                ErrorEntry(
                    code="refresh_failed",
                    message=f"{type(exc).__name__}: {exc}",
                    stage="refresh",
                    detail={"path": path},
                )
            )
            return self._in_memory(path)

    def _in_memory(self, path: str) -> FileData | None:
        source = self.source(path)
        if not source:
            return None
        stat = (self.store.root / path).stat()
        return self._rebuild(path, source, FileStamp(stat.st_mtime_ns, stat.st_size))

    def _rebuild(self, path: str, source: bytes, stamp: FileStamp) -> FileData:
        self._overlay.add(path)
        return build_file_data(path, parse_bytes(path, source), source, stamp)

    def _verified(self, path: str, data: FileData) -> FileData:
        source = self.source(path)
        if hash_bytes(source) == data.file.content_hash:
            return data
        return self._rebuild(path, source, data.file.stamp)
