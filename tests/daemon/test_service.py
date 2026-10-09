import sqlite3
import time
from pathlib import Path

import pytest
from daemon_support import Project, RunningDaemon, eventually

from contextweave.core.config import Config
from contextweave.core.types import JSON
from contextweave.daemon.client import DaemonUnavailable, RemoteError
from contextweave.daemon.lifecycle import client_for
from contextweave.daemon.projects import ProjectManager
from contextweave.daemon.rebuild import SHADOW_NAME, build_shadow, swap_shadow
from contextweave.daemon.scheduler import Scheduler
from contextweave.daemon.watcher import PollWatcher
from contextweave.index.indexer import index_project
from contextweave.index.root import state_dir
from contextweave.index.store import DB_NAME, SchemaMismatchError, Store

pytestmark = pytest.mark.daemon


def indexed_paths(root: Path) -> set[str]:
    with Store.open(root, readonly=True) as store:
        return {record.path for record in store.files()}


def status_projects(daemon: RunningDaemon) -> list[JSON]:
    status = daemon.client.call("status")
    assert isinstance(status, dict)
    projects = status["projects"]
    assert isinstance(projects, list)
    return projects


def settled(daemon: RunningDaemon) -> bool:
    project = status_projects(daemon)[0]
    assert isinstance(project, dict)
    reconciles = project["reconciles"]
    return project["pending"] == 0 and isinstance(reconciles, int) and reconciles >= 1


def test_unknown_method_and_bad_params_are_errors_not_crashes(daemon: RunningDaemon) -> None:
    with pytest.raises(RemoteError) as unknown:
        daemon.client.call("nope")
    assert unknown.value.code == "unknown_method"
    with pytest.raises(RemoteError) as bad:
        daemon.client.call("project.open", {})
    assert bad.value.code == "bad_params"
    with pytest.raises(RemoteError) as missing:
        daemon.client.call("project.open", {"project": "/nowhere"})
    assert missing.value.code == "not_indexed"
    assert daemon.client.call("ping")


def test_tool_call_runs_in_daemon_and_stat_checks_on_read(
    daemon: RunningDaemon, project: Project
) -> None:
    args: dict[str, JSON] = {
        "name": "get_symbol",
        "arguments": {"name": "alpha"},
        "cwd": str(project.root),
    }
    first = daemon.client.call("tool.call", args)
    assert isinstance(first, dict)
    structured = first["structured"]
    assert isinstance(structured, dict) and structured["found"] is True
    assert structured["project"] == {"key": project.key, "root": str(project.root), "via": "cwd"}
    (project.root / "alpha.py").write_text("def alpha():\n    return 100\n")
    second = daemon.client.call("tool.call", args)
    assert isinstance(second, dict) and "return 100" in str(second["structured"])
    with pytest.raises(RemoteError) as unknown:
        daemon.client.call("tool.call", {"name": "bogus"})
    assert unknown.value.code == "unknown_tool"


def test_project_opens_on_first_request_and_closes(daemon: RunningDaemon, project: Project) -> None:
    assert status_projects(daemon) == []
    daemon.client.call(
        "tool.call",
        {"name": "file_outline", "arguments": {"path": "alpha.py"}, "cwd": str(project.root)},
    )
    opened = status_projects(daemon)
    assert len(opened) == 1 and isinstance(opened[0], dict) and opened[0]["key"] == project.key
    closed = daemon.client.call("project.close", {"project": project.key})
    assert closed == {"key": project.key, "closed": True}
    assert status_projects(daemon) == []


def test_open_reconciles_changes_made_while_daemon_was_down(
    daemon: RunningDaemon, project: Project
) -> None:
    (project.root / "alpha.py").write_text("def alpha():\n    return 'changed'\n")
    (project.root / "delta.py").write_text("def delta():\n    return 4\n")
    (project.root / "gamma.py").unlink()
    daemon.client.call("project.open", {"project": str(project.root)})
    assert eventually(lambda: indexed_paths(project.root) == {"alpha.py", "beta.py", "delta.py"})
    with Store.open(project.root, readonly=True) as store:
        assert "changed" in str(store.get_file("alpha.py")) or store.stat_check("alpha.py")
        assert store.stat_check("alpha.py").__class__.__name__ == "Fresh"


def test_explicit_reconcile_reports_plan(daemon: RunningDaemon, project: Project) -> None:
    (project.root / "epsilon.py").write_text("def epsilon():\n    pass\n")
    (project.root / "beta.py").unlink()
    result = daemon.client.call("project.reconcile", {"project": project.key, "wait": True})
    assert result == {"changed": ["epsilon.py"], "removed": ["beta.py"]}
    assert "epsilon.py" in indexed_paths(project.root)


def test_watcher_keeps_open_project_current(daemon: RunningDaemon, project: Project) -> None:
    daemon.client.call("project.open", {"project": project.key})
    (project.root / "zeta.py").write_text("def zeta():\n    pass\n")
    assert eventually(lambda: "zeta.py" in indexed_paths(project.root))
    time.sleep(0.5)
    (project.root / "zeta.py").unlink()
    assert eventually(lambda: "zeta.py" not in indexed_paths(project.root))


def test_reindex_with_wait(daemon: RunningDaemon, project: Project) -> None:
    (project.root / "alpha.py").write_text("def alpha():\n    return 7\n")
    result = daemon.client.call(
        "index.reindex", {"project": project.key, "paths": ["alpha.py"], "wait": True}
    )
    assert result == {"queued": 1, "drained": True}


def test_idle_projects_are_reaped(project: Project) -> None:
    now = [0.0]
    scheduler = Scheduler()
    manager = ProjectManager(
        scheduler,
        lambda root, key, on_events: PollWatcher(root, (), 60.0, on_events),
        idle_s=30.0,
        clock=lambda: now[0],
    )
    manager.open(project.key, project.root)
    now[0] = 29.0
    assert manager.reap_idle() == []
    manager.open(project.key, project.root)
    now[0] = 58.0
    assert manager.reap_idle() == []
    now[0] = 60.0
    assert manager.reap_idle() == [project.key]
    assert manager.keys() == []


def test_shadow_rebuild_swaps_atomically_while_live_serves(project: Project) -> None:
    live = state_dir(project.root) / DB_NAME
    (project.root / "fresh.py").write_text("def fresh():\n    pass\n")
    seen_during_build: list[set[str]] = []

    def finish(store: Store) -> None:
        seen_during_build.append(indexed_paths(project.root))
        assert not live.with_name(SHADOW_NAME).exists()

    report = build_shadow(project.root, Config(), finish=finish)
    assert report.files == 4
    assert seen_during_build == [{"alpha.py", "beta.py", "gamma.py"}]
    assert (state_dir(project.root) / SHADOW_NAME).is_file()
    assert indexed_paths(project.root) == {"alpha.py", "beta.py", "gamma.py"}
    assert swap_shadow(project.root)
    assert not (state_dir(project.root) / SHADOW_NAME).exists()
    assert indexed_paths(project.root) == {"alpha.py", "beta.py", "gamma.py", "fresh.py"}
    assert not swap_shadow(project.root)


def test_rebuild_rpc_swaps_and_stale_schema_triggers_rebuild(
    daemon: RunningDaemon, project: Project
) -> None:
    (project.root / "fresh.py").write_text("def fresh():\n    pass\n")
    result = daemon.client.call("index.rebuild", {"project": project.key, "wait": True})
    assert result == {"started": True, "done": True}
    assert "fresh.py" in indexed_paths(project.root)
    assert eventually(lambda: settled(daemon))
    daemon.client.call("project.close", {"project": project.key})
    connection = sqlite3.connect(state_dir(project.root) / DB_NAME)
    connection.execute("PRAGMA user_version = 999")
    connection.commit()
    connection.close()
    with pytest.raises(SchemaMismatchError):
        Store.open(project.root, readonly=True)
    daemon.client.call("project.open", {"project": project.key})

    def healthy() -> bool:
        try:
            return "fresh.py" in indexed_paths(project.root)
        except SchemaMismatchError:
            return False

    assert eventually(healthy)


def test_status_and_shutdown_leave_no_socket(daemon: RunningDaemon) -> None:
    status = daemon.client.call("status")
    assert isinstance(status, dict) and status["queue"] == 0
    socket_path = daemon.client.socket_path
    daemon.client.call("shutdown")
    assert eventually(lambda: not socket_path.exists())
    time.sleep(0.05)
    with pytest.raises(DaemonUnavailable):
        client_for(daemon.env, 1.0).call("ping")


def test_index_project_unaffected_by_daemon_import(project: Project) -> None:
    assert index_project(project.root, Config()).unchanged == 3
