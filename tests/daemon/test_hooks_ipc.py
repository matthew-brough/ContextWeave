import json
from pathlib import Path

import pytest
from daemon_support import Project, RunningDaemon, eventually

from contextweave.agents.cli_entry import run_hook
from contextweave.agents.transport import IpcTransport
from contextweave.core.config import load_config
from contextweave.core.project import ResolvedProject
from contextweave.daemon.client import IpcClient
from contextweave.daemon.lifecycle import client_for
from contextweave.memory.store import MemoryStore, memory_db_path

pytestmark = pytest.mark.daemon

SESSION = "sess-ipc"


def counts(env: dict[str, str], key: str) -> dict[str, int]:
    store = MemoryStore.open(memory_db_path(key, env), key)
    try:
        result: dict[str, int] = {}
        for table in ("sessions", "prompts"):
            row: tuple[int] = store.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            result[table] = row[0]
        return result
    finally:
        store.close()


def resolved(project: Project) -> ResolvedProject:
    return ResolvedProject(project.key, project.root, "cwd")


def test_full_session_over_ipc(daemon: RunningDaemon, project: Project) -> None:
    transport = IpcTransport(client_for(daemon.env, 5.0), daemon.env, timeout_s=5.0)
    loaded = load_config(project.root, strict=False, env=daemon.env)
    target = resolved(project)
    transport.session_start(target, SESSION, loaded)
    transport.prompt(target, SESSION, "why is alpha one")
    transport.tool_event(target, SESSION, "Edit", "edited alpha", "{}", "{}")
    transport.stop(target, SESSION)
    transport.session_end(target, SESSION, None)
    assert eventually(lambda: counts(daemon.env, project.key) == {"sessions": 1, "prompts": 1})


def test_fails_open_without_daemon(
    env: dict[str, str], project: Project, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[str] = []

    def record(env: object) -> None:
        spawned.append("x")

    monkeypatch.setattr("contextweave.agents.transport.spawn_detached", record)
    transport = IpcTransport(client_for(env), env)
    loaded = load_config(project.root, strict=False, env=env)
    target = resolved(project)
    assert transport.session_start(target, SESSION, loaded) is None
    assert spawned == ["x"]
    transport.prompt(target, SESSION, "text")
    transport.tool_event(target, SESSION, "Edit", "s", "i", "o")
    transport.stop(target, SESSION)
    transport.session_end(target, SESSION, "clear")
    assert spawned == ["x"]


def test_remote_errors_and_spawn_failure_are_swallowed(
    daemon: RunningDaemon, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(env: object) -> None:
        raise OSError("no exec")

    monkeypatch.setattr("contextweave.agents.transport.spawn_detached", refuse)
    transport = IpcTransport(client_for(daemon.env, 5.0), daemon.env, timeout_s=5.0)
    stranger = ResolvedProject("000000000000", tmp_path, "cwd")
    transport.prompt(stranger, SESSION, "text")
    transport.stop(stranger, SESSION)
    dead = IpcClient(tmp_path / "missing.sock", 0.2)
    IpcTransport(dead, daemon.env).prompt(stranger, SESSION, "text")
    assert (
        IpcTransport(dead, daemon.env).session_start(
            stranger, SESSION, load_config(None, strict=False, env=daemon.env)
        )
        is None
    )


def test_cw_hook_entrypoint_uses_ipc_by_default(daemon: RunningDaemon, project: Project) -> None:
    body = {"session_id": SESSION, "cwd": str(project.root)}
    out, code = run_hook("SessionStart", json.dumps(body).encode(), daemon.env)
    assert code == 0
    assert "additionalContext" in json.loads(out)["hookSpecificOutput"]
    prompt = json.dumps({**body, "prompt": "why is alpha one"}).encode()
    assert run_hook("UserPromptSubmit", prompt, daemon.env) == (b"", 0)
    assert eventually(lambda: counts(daemon.env, project.key) == {"sessions": 1, "prompts": 1})


def test_daemon_status_lists_model_slots(daemon: RunningDaemon) -> None:
    status = daemon.client.call("status")
    assert isinstance(status, dict)
    assert status["models"] == []
