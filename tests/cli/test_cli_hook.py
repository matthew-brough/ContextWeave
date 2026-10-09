import io
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from cli_support import cw

from contextweave.agents.cli_entry import run_hook
from contextweave.core.refs import project_key
from contextweave.memory.store import MemoryStore, memory_db_path


def no_spawn(env: Mapping[str, str]) -> None:
    return None


def feed(monkeypatch: pytest.MonkeyPatch, payload: dict[str, object]) -> None:
    raw = io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(raw))


def test_hook_with_no_daemon_exits_zero_and_silent(
    initialised: Path,
    env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("contextweave.agents.transport.spawn_detached", no_spawn)
    feed(monkeypatch, {"session_id": "s1", "cwd": str(initialised), "prompt": "hi"})
    result = cw(capsys, env, initialised, "hook", "UserPromptSubmit")
    assert result.code == 0
    assert result.out == ""


def test_hook_garbage_stdin_is_fail_open(
    initialised: Path,
    env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(b"not json")))
    assert cw(capsys, env, initialised, "hook", "Stop").code == 0


def test_default_transport_is_ipc(
    initialised: Path, env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[str] = []

    def spawn(env: Mapping[str, str]) -> None:
        spawned.append("x")

    monkeypatch.setattr("contextweave.agents.transport.spawn_detached", spawn)
    payload = json.dumps({"session_id": "s1", "cwd": str(initialised)}).encode()
    assert run_hook("SessionStart", payload, env) == (b"", 0)
    assert spawned == ["x"]
    key = project_key(initialised)
    assert not memory_db_path(key, env).exists() or _no_sessions(env, key)


def _no_sessions(env: dict[str, str], key: str) -> bool:
    store = MemoryStore.open(memory_db_path(key, env), key)
    try:
        row: tuple[int] = store.conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] == 0
    finally:
        store.close()
