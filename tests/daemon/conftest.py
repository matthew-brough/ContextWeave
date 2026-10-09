import contextlib
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest
from daemon_support import SOURCES, Project, RunningDaemon, eventually

from contextweave.core.config import Config
from contextweave.core.paths import user_state_dir
from contextweave.core.refs import project_key
from contextweave.daemon.client import DaemonUnavailable
from contextweave.daemon.lifecycle import client_for
from contextweave.daemon.server import run_daemon
from contextweave.index.indexer import index_project
from contextweave.index.projects import ProjectRegistry


@pytest.fixture
def env() -> Iterator[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="cw") as raw:
        base = Path(raw)
        yield {
            "HOME": str(base / "home"),
            "XDG_STATE_HOME": str(base / "state"),
            "XDG_CACHE_HOME": str(base / "cache"),
            "XDG_CONFIG_HOME": str(base / "config"),
            "XDG_RUNTIME_DIR": str(base / "run"),
        }


@pytest.fixture
def project(env: dict[str, str], tmp_path: Path) -> Project:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".git").mkdir()
    for name, text in SOURCES.items():
        (root / name).write_text(text, encoding="utf-8")
    index_project(root, Config())
    ProjectRegistry(user_state_dir(env)).register(root)
    return Project(root.resolve(), project_key(root.resolve()))


@pytest.fixture
def daemon(env: dict[str, str], project: Project) -> Iterator[RunningDaemon]:
    thread = threading.Thread(
        target=run_daemon,
        args=(env, Config()),
        kwargs={"poll_interval_s": 0.1, "debounce_s": 0.05, "housekeeping_s": 0.1},
        daemon=True,
    )
    thread.start()
    client = client_for(env, 10.0)

    def up() -> bool:
        try:
            client.call("ping")
        except DaemonUnavailable:
            return False
        return True

    assert eventually(up)
    yield RunningDaemon(env, client)
    with contextlib.suppress(DaemonUnavailable):
        client.call("shutdown")
    thread.join(10)
