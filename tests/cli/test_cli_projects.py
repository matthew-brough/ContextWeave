import json
import shutil
from pathlib import Path

import pytest
from cli_support import cw, make_project

from contextweave.core.paths import user_state_dir
from contextweave.core.refs import project_key
from contextweave.index.projects import ProjectRegistry
from contextweave.memory.store import MemoryStore, memory_db_path


def test_projects_lists(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert project_key(initialised) in cw(capsys, env, initialised, "projects").out
    rows = json.loads(cw(capsys, env, initialised, "projects", "--json").out)
    assert rows[0]["root"] == str(initialised)
    assert rows[0]["missing"] is False


def test_projects_empty(
    tmp_path: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert "no registered projects" in cw(capsys, env, tmp_path, "projects").out


def test_forget_by_cwd_removes_state(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "forget")
    assert result.code == 0
    assert not (initialised / ".contextweave").exists()
    assert ProjectRegistry(user_state_dir(env)).list() == []


def test_forget_keep_memory(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    key = project_key(initialised)
    MemoryStore.open(memory_db_path(key, env), key).close()
    result = cw(capsys, env, initialised, "forget", key, "--keep-memory")
    assert "memory kept" in result.out
    assert memory_db_path(key, env).is_file()


def test_forget_missing(
    initialised: Path,
    env: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    other = make_project(tmp_path, "other")
    ProjectRegistry(user_state_dir(env)).register(other)
    shutil.rmtree(other)
    result = cw(capsys, env, initialised, "forget", "--missing")
    assert result.code == 0
    assert project_key(other) in result.out
    assert [entry.root for entry in ProjectRegistry(user_state_dir(env)).list()] == [initialised]


def test_forget_unknown_and_conflicting_args(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cw(capsys, env, initialised, "forget", "aaaaaaaaaaaa").code == 1
    assert cw(capsys, env, initialised, "forget", "x", "--missing").code == 2
