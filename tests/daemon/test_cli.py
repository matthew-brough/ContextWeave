import json
from pathlib import Path

import pytest
from daemon_support import Project, RunningDaemon

from contextweave.cli import main

pytestmark = pytest.mark.daemon


def use_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_status_and_stop_without_daemon(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_env(monkeypatch, env)
    assert main(["daemon", "status"]) == 1
    assert "not running" in capsys.readouterr().out
    assert main(["daemon", "stop"]) == 0
    assert main(["daemon", "call", "ping"]) == 1


def test_call_against_running_daemon(
    daemon: RunningDaemon,
    project: Project,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_env(monkeypatch, daemon.env)
    assert main(["daemon", "call", "ping"]) == 0
    assert "pid" in json.loads(capsys.readouterr().out)
    assert main(["daemon", "call", "project.open", json.dumps({"project": project.key})]) == 0
    assert main(["daemon", "status"]) == 0
    assert project.key in capsys.readouterr().out
    assert main(["daemon", "call", "nope"]) == 1
    assert main(["daemon", "call", "ping", "[1]"]) == 2


def test_serve_non_inproc_exits_cleanly_on_closed_stdin(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_env(monkeypatch, env)
    with (tmp_path / "stdin").open("w+") as stdin:
        monkeypatch.setattr("sys.stdin", stdin)
        assert main(["serve"]) == 0
