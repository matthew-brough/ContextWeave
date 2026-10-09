import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Iterator

import anyio
import mcp_types as types
import pytest
from daemon_support import Project, eventually
from mcp import Client

from contextweave.core.config import Config
from contextweave.core.types import JSON
from contextweave.daemon.lifecycle import (
    DaemonPaths,
    client_for,
    daemon_status,
    ensure_running,
    is_running,
    read_pid,
    stop_daemon,
)
from contextweave.daemon.pidlock import lock_is_held
from contextweave.daemon.shim import shim_backend
from contextweave.index.projects import ProjectRegistry
from contextweave.server.context import ToolContext, inproc_opener
from contextweave.server.mcp import build_backend_server

pytestmark = pytest.mark.daemon


@pytest.fixture
def live(env: dict[str, str], project: Project) -> Iterator[dict[str, str]]:
    yield env
    pid = read_pid(env)
    if pid is not None and lock_is_held(DaemonPaths.from_env(env).pid):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def test_start_status_stop(live: dict[str, str]) -> None:
    assert daemon_status(live) is None
    assert not is_running(live)
    assert ensure_running(live)
    status = daemon_status(live)
    assert status is not None and status["pid"] == read_pid(live)
    assert ensure_running(live)
    assert stop_daemon(live)
    assert daemon_status(live) is None
    assert not lock_is_held(DaemonPaths.from_env(live).pid)
    assert stop_daemon(live)


def test_second_daemon_exits_while_first_holds_lock(live: dict[str, str]) -> None:
    assert ensure_running(live)
    first = read_pid(live)
    result = subprocess.run(
        [sys.executable, "-m", "contextweave.cli", "daemon", "run"],
        env=live,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode != 0
    assert read_pid(live) == first
    assert is_running(live)
    stop_daemon(live)


def structured(result: types.CallToolResult) -> dict[str, JSON]:
    assert not result.is_error
    content = result.structured_content
    assert content is not None
    return {key: value for key, value in content.items() if isinstance(key, str)}


def test_shim_end_to_end_and_reconnect_after_kill(live: dict[str, str], project: Project) -> None:
    config = Config()
    local = ToolContext(
        config,
        ProjectRegistry(DaemonPaths.from_env(live).pid.parent),
        inproc_opener(config, readonly=True),
        cwd=project.root,
    )
    backend = shim_backend(local, client_for(live, 10.0), lambda: ensure_running(live))
    pids: list[int | None] = []

    async def lookup(client: Client) -> JSON:
        found = structured(await client.call_tool("get_symbol", {"name": "beta"}))
        return found["found"]

    async def main() -> None:
        async with Client(build_backend_server(backend)) as client:
            assert await lookup(client) is True
            pids.append(read_pid(live))
            first = pids[0]
            assert first is not None
            os.kill(first, signal.SIGKILL)
            assert eventually(lambda: not is_running(live))
            assert await lookup(client) is True
            pids.append(read_pid(live))

    anyio.run(main)
    assert pids[0] != pids[1]
    assert is_running(live)
    assert stop_daemon(live)
