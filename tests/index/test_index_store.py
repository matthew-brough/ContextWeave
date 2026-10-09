import sqlite3
from pathlib import Path

import pytest

from contextweave.index.records import Changed, FileStamp, Fresh, Gone
from contextweave.index.root import state_dir
from contextweave.index.store import (
    SCHEMA_VERSION,
    LockTimeoutError,
    ReadOnlyStoreError,
    SchemaMismatchError,
    Store,
    hash_bytes,
)
from contextweave.lang.parse import parse_bytes

EXPECTED_TABLES = {
    "files",
    "symbols",
    "imports",
    "import_targets",
    "file_scopes",
    "file_sides",
    "handler_meta",
    "schema_nodes",
    "doc_nodes",
    "chunks",
    "chunk_symbols",
    "symbol_edges",
    "annotations",
    "vec_spaces",
    "vectors_meta",
    "fts_code",
    "packs",
    "feedback",
}


def ingest(store: Store, relative: str, text: str) -> None:
    target = store.root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    stat = target.stat()
    source = text.encode()
    store.upsert_file(
        relative, parse_bytes(relative, source), source, FileStamp(stat.st_mtime_ns, stat.st_size)
    )


def lock_blocked(store: Store) -> bool:
    try:
        with store.writer_lock(timeout_s=0.1):
            return False
    except LockTimeoutError:
        return True


def test_schema_created(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        assert store.path == state_dir(tmp_path) / "index.db"
    connection = sqlite3.connect(state_dir(tmp_path) / "index.db")
    names = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master")}
    assert names >= EXPECTED_TABLES
    assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert str(connection.execute("PRAGMA journal_mode").fetchone()[0]) == "wal"
    assert (state_dir(tmp_path) / ".gitignore").read_text(encoding="utf-8") == "*\n"
    connection.close()


def test_user_version_mismatch_signals_rebuild(tmp_path: Path) -> None:
    Store.open(tmp_path).close()
    connection = sqlite3.connect(state_dir(tmp_path) / "index.db")
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    connection.commit()
    connection.close()
    with pytest.raises(SchemaMismatchError) as caught:
        Store.open(tmp_path)
    assert caught.value.found == SCHEMA_VERSION + 1
    assert caught.value.expected == SCHEMA_VERSION
    with pytest.raises(SchemaMismatchError):
        Store.open(tmp_path, readonly=True)


def test_upsert_writes_symbols_imports_and_hashes(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "import os\n\ndef alpha():\n    return 1\n")
        data = store.get_data("m.py")
        assert data is not None
        assert data.file.profile == "symbols"
        assert data.file.lang == "python"
        assert data.file.content_hash == hash_bytes(b"import os\n\ndef alpha():\n    return 1\n")
        assert [s.name for s in data.symbols] == ["alpha"]
        symbol = data.symbols[0]
        assert symbol.content_hash == hash_bytes(b"def alpha():\n    return 1")
        assert [i.module for i in data.imports] == ["os"]
        assert store.symbols_by_name("alpha") == (symbol,)
        assert store.symbols_by_qualified_name("alpha") == (symbol,)
        assert store.symbols_in_file("m.py") == (symbol,)


def test_upsert_replaces_rows_and_bumps_generation(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "def alpha():\n    pass\n")
        first_id = store.get_file("m.py")
        generation = store.generation
        ingest(store, "m.py", "def beta():\n    pass\n\ndef gamma():\n    pass\n")
        assert store.symbols_by_name("alpha") == ()
        assert [s.name for s in store.symbols_in_file("m.py")] == ["beta", "gamma"]
        assert len(store.files()) == 1
        assert first_id is not None and store.get_file("m.py") is not None
        assert store.generation > generation


def test_schema_and_doc_nodes(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "c.json", '{"a": {"b": 1}}\n')
        ingest(store, "d.md", "# Title\n\ntext\n")
        schema = store.get_data("c.json")
        doc = store.get_data("d.md")
    assert schema is not None and doc is not None
    assert schema.file.profile == "schema"
    assert {n.key_path for n in schema.schema_nodes} >= {"a", "a.b"}
    assert doc.file.profile == "outline"
    assert [n.title for n in doc.doc_nodes] == ["Title"]


def test_unknown_extension_is_text(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "notes.xyz", "hello\n")
        record = store.get_file("notes.xyz")
    assert record is not None
    assert record.profile == "text"
    assert record.lang is None and record.group is None


def test_delete_file(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "def alpha():\n    pass\n")
        assert store.delete_file("m.py")
        assert not store.delete_file("m.py")
        assert store.get_file("m.py") is None
        assert store.symbols_by_name("alpha") == ()


def test_stat_check(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        assert isinstance(store.stat_check("m.py"), Gone)
        ingest(store, "m.py", "x = 1\n")
        assert isinstance(store.stat_check("m.py"), Fresh)
        (tmp_path / "m.py").write_text("x = 22\n", encoding="utf-8")
        assert isinstance(store.stat_check("m.py"), Changed)
        (tmp_path / "m.py").unlink()
        assert isinstance(store.stat_check("m.py"), Gone)
        (tmp_path / "new.py").write_text("y = 1\n", encoding="utf-8")
        assert isinstance(store.stat_check("new.py"), Changed)


def test_refresh_reparses_on_change(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "def alpha():\n    pass\n")
        fresh = store.refresh("m.py")
        assert isinstance(fresh.check, Fresh) and not fresh.persisted
        (tmp_path / "m.py").write_text("def omega():\n    pass\n\n# padding\n", encoding="utf-8")
        changed = store.refresh("m.py")
        assert isinstance(changed.check, Changed) and changed.persisted
        assert changed.data is not None
        assert [s.name for s in changed.data.symbols] == ["omega"]
        assert store.symbols_by_name("alpha") == ()
        assert isinstance(store.stat_check("m.py"), Fresh)


def test_refresh_same_bytes_only_touches_stamp(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "def alpha():\n    pass\n")
        before = store.get_file("m.py")
        (tmp_path / "m.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
        result = store.refresh("m.py")
        assert before is not None and result.data is not None
        assert result.data.file.content_hash == before.content_hash
        assert isinstance(store.stat_check("m.py"), Fresh)


def test_refresh_gone_deletes_and_binary_counts_as_gone(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "x = 1\n")
        ingest(store, "n.py", "y = 1\n")
        (tmp_path / "m.py").unlink()
        gone = store.refresh("m.py")
        assert isinstance(gone.check, Gone) and gone.persisted and gone.data is None
        assert store.get_file("m.py") is None
        (tmp_path / "n.py").write_bytes(b"\0\0\0binary")
        binary = store.refresh("n.py")
        assert binary.data is None and store.get_file("n.py") is None


def test_readonly_never_writes(tmp_path: Path) -> None:
    with Store.open(tmp_path) as store:
        ingest(store, "m.py", "def alpha():\n    pass\n")
    database = state_dir(tmp_path) / "index.db"
    snapshot = (database.stat().st_mtime_ns, database.read_bytes())
    (tmp_path / "m.py").write_text("def beta():\n    pass\n\n# more\n", encoding="utf-8")
    (tmp_path / "extra.py").write_text("def gamma():\n    pass\n", encoding="utf-8")
    with Store.open(tmp_path, readonly=True) as store:
        changed = store.refresh("m.py")
        assert not changed.persisted
        assert changed.data is not None
        assert [s.name for s in changed.data.symbols] == ["beta"]
        added = store.refresh("extra.py")
        assert added.data is not None and not added.persisted
        assert store.get_file("extra.py") is None
        with pytest.raises(ReadOnlyStoreError):
            store.delete_file("m.py")
        with pytest.raises(ReadOnlyStoreError), store.writer_lock():
            pass
    assert (database.stat().st_mtime_ns, database.read_bytes()) == snapshot


def test_readonly_without_db_creates_nothing(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text("def alpha():\n    pass\n", encoding="utf-8")
    with Store.open(tmp_path, readonly=True) as store:
        assert store.files() == ()
        result = store.refresh("m.py")
        assert result.data is not None and [s.name for s in result.data.symbols] == ["alpha"]
    assert not state_dir(tmp_path).exists()


def test_writer_lock_excludes_second_writer(tmp_path: Path) -> None:
    with Store.open(tmp_path) as first, Store.open(tmp_path) as second:
        with first.writer_lock(), first.writer_lock():
            assert lock_blocked(second)
        with second.writer_lock(timeout_s=0.5):
            pass
