import json
from pathlib import Path

import pytest
from cli_support import cw

from contextweave.cli_commands import agents_cmd
from contextweave.core.paths import user_state_dir
from contextweave.index.projects import ProjectRegistry


def test_init_indexes_registers_and_installs(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, root, "init")
    assert result.code == 0, result.err
    assert "indexed: 1 added" in result.out
    assert "project registered" in result.out
    assert (root / ".contextweave" / "index.db").is_file()
    assert (root / ".contextweave" / ".gitignore").read_text() == "*\n"
    assert ProjectRegistry(user_state_dir(env)).get(root) is not None
    settings = json.loads((root / ".claude" / "settings.json").read_text())
    assert "SessionStart" in settings["hooks"]
    assert "contextweave:begin" in (root / "CLAUDE.md").read_text()
    assert "contextweave" in json.loads((root / ".mcp.json").read_text())["mcpServers"]


def test_init_is_resumable(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    cw(capsys, env, initialised, "init")
    (initialised / "extra.py").write_text("def gamma():\n    return 3\n", encoding="utf-8")
    result = cw(capsys, env, initialised, "init")
    assert result.code == 0
    assert "1 added" in result.out
    assert "0 removed" in result.out


def test_dry_run_writes_nothing(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, root, "init", "--dry-run")
    assert result.code == 0
    assert "would create" in result.out
    assert not (root / ".claude").exists()
    assert not (root / ".contextweave").exists()
    assert ProjectRegistry(user_state_dir(env)).list() == []


def test_parent_conflict_blocks_unless_forced(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    (root / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"context-engine": {"command": "cce"}}}), encoding="utf-8"
    )
    blocked = cw(capsys, env, root, "init")
    assert blocked.code != 0
    assert "context-engine" in blocked.err
    assert not (root / ".contextweave").exists()
    forced = cw(capsys, env, root, "init", "--force")
    assert forced.code == 0


def test_uninstall_keeps_index_and_registry(
    initialised: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    result = cw(capsys, env, initialised, "uninstall")
    assert result.code == 0
    assert not (initialised / ".claude" / "settings.json").exists()
    assert "contextweave:begin" not in (
        (initialised / "CLAUDE.md").read_text() if (initialised / "CLAUDE.md").exists() else ""
    )
    assert (initialised / ".contextweave" / "index.db").is_file()
    assert ProjectRegistry(user_state_dir(env)).get(initialised) is not None


def test_unknown_agent_is_usage_error(
    root: Path, env: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cw(capsys, env, root, "init", "--agent", "nope").code == 2


def test_interrupted_init_exits_130_and_leaves_no_half_install(
    root: Path,
    env: dict[str, str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(*_args: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(agents_cmd, "full_index", interrupt)
    result = cw(capsys, env, root, "init")
    assert result.code == 130
    assert result.err.strip() == "cw init: interrupted"
    assert ProjectRegistry(user_state_dir(env)).get(root) is None
    assert not (root / ".claude" / "settings.json").exists()
    assert not (root / ".mcp.json").exists()
