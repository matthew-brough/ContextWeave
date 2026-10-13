import sqlite3
from pathlib import Path

import pytest
from conftest import KEY_A

from contextweave.memory.store import (
    SCHEMA_VERSION,
    MemorySchemaError,
    MemoryStore,
    memory_db_path,
    state_dir,
)

TABLES = ["sessions", "prompts", "savings_log"]


def test_state_dir_honours_xdg_and_falls_back_to_home() -> None:
    assert state_dir({"XDG_STATE_HOME": "/x"}) == Path("/x/contextweave")
    assert state_dir({}) == Path.home() / ".local" / "state" / "contextweave"
    assert memory_db_path(KEY_A, {"XDG_STATE_HOME": "/x"}) == Path(
        f"/x/contextweave/projects/{KEY_A}/memory.db"
    )


def test_schema_has_project_key_on_every_table(store_a: MemoryStore) -> None:
    for table in TABLES:
        columns = [row[1] for row in store_a.conn.execute(f"PRAGMA table_info({table})")]
        assert "project_key" in columns, table
        info = [row for row in store_a.conn.execute(f"PRAGMA table_info({table})")]
        assert next(row[3] for row in info if row[1] == "project_key") == 1, table


def test_wal_and_user_version(store_a: MemoryStore) -> None:
    mode = store_a.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"
    assert store_a.user_version() == SCHEMA_VERSION


def test_unknown_schema_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(MemorySchemaError):
        MemoryStore.open(path, KEY_A)


def test_reopen_is_idempotent(state_env: dict[str, str]) -> None:
    path = memory_db_path(KEY_A, state_env)
    first = MemoryStore.open(path, KEY_A)
    first.add_prompt("s1", "keep me")
    first.close()
    second = MemoryStore.open(path, KEY_A)
    assert second.current_prompt_number("s1") == 1
    second.close()


def test_retired_schema_drops_recall_tables(tmp_path: Path) -> None:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY, project_key TEXT)")
    conn.execute("CREATE VIRTUAL TABLE notes_fts USING fts5(text)")
    conn.execute(
        "CREATE TABLE prompts (id INTEGER PRIMARY KEY AUTOINCREMENT, project_key TEXT NOT NULL, session_id TEXT NOT NULL, prompt_number INTEGER NOT NULL, prompt_text TEXT NOT NULL, created_at_epoch INTEGER NOT NULL)"
    )
    conn.execute("INSERT INTO prompts VALUES (1, ?, 's1', 1, 'kept', 0)", (KEY_A,))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    store = MemoryStore.open(path, KEY_A)
    try:
        names = {
            row[0]
            for row in store.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert names == {"sessions", "prompts", "savings_log", "sqlite_sequence"}
        assert store.user_version() == SCHEMA_VERSION
        assert store.current_prompt_epoch("s1") == 0
    finally:
        store.close()
