import sqlite3
from pathlib import Path

import pytest
from conftest import KEY_A, KEY_B

from contextweave.memory.capture import remember
from contextweave.memory.store import (
    SCHEMA_VERSION,
    MemorySchemaError,
    MemoryStore,
    memory_db_path,
    project_memory_dbs,
    state_dir,
)

TABLES = [
    "sessions",
    "prompts",
    "tool_events",
    "tool_event_payloads",
    "turn_summaries",
    "decisions",
    "code_areas",
    "notes",
    "savings_log",
    "vectors_meta",
    "pending_summaries",
]


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
    assert store_a.vec_enabled


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
    remember(first, "note", "keep me")
    first.close()
    second = MemoryStore.open(path, KEY_A)
    assert second.fts_ranked("note", '"keep"', 5) == [1]
    second.close()


def test_project_dbs_listed_sorted(
    state_env: dict[str, str], store_a: MemoryStore, store_b: MemoryStore
) -> None:
    keys = [key for key, _ in project_memory_dbs(state_env)]
    assert keys == [KEY_A, KEY_B]
    assert project_memory_dbs({"XDG_STATE_HOME": "/nonexistent-cw"}) == []
