import json
from pathlib import Path

import pytest
from cli_support import cw

from contextweave.core.refs import project_key
from contextweave.memory.savings import record_savings
from contextweave.memory.store import MemoryStore, memory_db_path


def seed(env: dict[str, str], root: Path) -> None:
    key = project_key(root)
    store = MemoryStore.open(memory_db_path(key, env), key)
    try:
        record_savings(store, tokens_served=100, full_file_tokens=1000)
    finally:
        store.close()


def test_savings_text_json_reset(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    seed(env, initialised)
    assert "retrieval" in cw(capsys, env, initialised, "savings").out
    data = json.loads(cw(capsys, env, initialised, "savings", "--json").out)
    assert data["retrieval"]["saved"] == 900
    assert "cleared 1" in cw(capsys, env, initialised, "savings", "--reset").out
    again = json.loads(cw(capsys, env, initialised, "savings", "--json").out)
    assert again["retrieval"]["calls"] == 0
