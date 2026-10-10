from pathlib import Path

from agents_support import read_json, write_json

from contextweave.agents.cli_entry import run_init, run_uninstall
from contextweave.core.config import Config
from contextweave.core.paths import user_state_dir
from contextweave.index.indexer import IndexReport
from contextweave.index.projects import ProjectRegistry


def registered(env: dict[str, str]) -> list[str]:
    return [entry.key for entry in ProjectRegistry(user_state_dir(env)).list()]


def test_init_indexes_registers_installs(root: Path, home: Path, env: dict[str, str]) -> None:
    result = run_init(root, home=home, env=env)
    assert result.exit_code == 0 and result.registered
    assert (root / ".contextweave" / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert (root / ".contextweave" / "index.db").is_file()
    assert len(registered(env)) == 1
    assert (root / ".mcp.json").is_file() and (root / "CLAUDE.md").is_file()
    assert len(result.changes) == 3


def test_init_order_index_before_register_and_install(
    root: Path, home: Path, env: dict[str, str]
) -> None:
    seen: list[tuple[list[str], bool]] = []

    def spy(path: Path, config: Config) -> IndexReport:
        seen.append((registered(env), (root / ".mcp.json").exists()))
        return IndexReport((), (), (), 0, 0)

    run_init(root, home=home, env=env, indexer=spy)
    assert seen == [([], False)]
    assert len(registered(env)) == 1


def test_index_failure_leaves_no_registration(root: Path, home: Path, env: dict[str, str]) -> None:
    def boom(path: Path, config: Config) -> IndexReport:
        raise RuntimeError("disk on fire")

    result = run_init(root, home=home, env=env, indexer=boom)
    assert result.exit_code != 0 and not result.registered
    assert registered(env) == []
    assert not (root / ".mcp.json").exists() and not (root / "CLAUDE.md").exists()


def test_conflict_blocks_init_without_force(root: Path, home: Path, env: dict[str, str]) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"context-engine": {"command": "cce"}}})
    result = run_init(root, home=home, env=env)
    assert result.exit_code != 0 and result.conflicts
    assert "cce uninstall" in result.messages[0]
    assert registered(env) == [] and not (root / "CLAUDE.md").exists()
    assert not (root / ".contextweave").exists()


def test_force_installs_despite_conflict(root: Path, home: Path, env: dict[str, str]) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"ast-mcp": {}}})
    result = run_init(root, home=home, env=env, force=True)
    assert result.exit_code == 0 and result.conflicts
    servers = read_json(root / ".mcp.json")
    assert isinstance(servers, dict) and isinstance(servers["mcpServers"], dict)
    assert set(servers["mcpServers"]) == {"ast-mcp", "contextweave"}


def test_user_level_conflict_blocks(root: Path, home: Path, env: dict[str, str]) -> None:
    write_json(home / ".claude.json", {"mcpServers": {"context-engine": {}}})
    assert run_init(root, home=home, env=env).exit_code != 0


def test_dry_run_writes_nothing(root: Path, home: Path, env: dict[str, str]) -> None:
    result = run_init(root, home=home, env=env, dry_run=True)
    assert result.exit_code == 0 and len(result.changes) == 3
    assert registered(env) == []
    assert not (root / ".contextweave").exists() and not (root / ".mcp.json").exists()


def test_unknown_agent_and_bad_config(root: Path, home: Path, env: dict[str, str]) -> None:
    assert run_init(root, agent="nope", home=home, env=env).exit_code == 2
    (root / ".contextweave.toml").write_text("[nonsense]\nkey = 1\n", encoding="utf-8")
    assert run_init(root, home=home, env=env).exit_code == 2


def test_malformed_agent_config_fails_before_indexing(
    root: Path, home: Path, env: dict[str, str]
) -> None:
    (root / ".mcp.json").write_text("{bad", encoding="utf-8")
    result = run_init(root, home=home, env=env)
    assert result.exit_code != 0
    assert registered(env) == [] and not (root / ".contextweave").exists()


def test_init_plugin_mode(root: Path, home: Path, env: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "plug"
    result = run_init(root, plugin=True, plugin_dir=out, home=home, env=env)
    assert result.exit_code == 0
    assert (out / ".claude-plugin" / "plugin.json").is_file()
    assert not (root / "CLAUDE.md").exists()


def test_uninstall_reverses_init(root: Path, home: Path, env: dict[str, str]) -> None:
    run_init(root, home=home, env=env)
    dry = run_uninstall(root, home=home, dry_run=True)
    assert len(dry.changes) == 3 and (root / ".mcp.json").exists()
    result = run_uninstall(root, home=home)
    assert result.exit_code == 0 and len(result.changes) == 3
    assert not (root / ".mcp.json").exists() and not (root / "CLAUDE.md").exists()
    assert not (root / ".claude").exists()


def test_agent_none_indexes_and_registers_without_agent_files(
    root: Path, home: Path, env: dict[str, str]
) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"context-engine": {"command": "cce"}}})
    result = run_init(root, agent="none", home=home, env=env)
    assert result.exit_code == 0 and result.registered
    assert result.changes == () and result.conflicts == ()
    assert (root / ".contextweave" / "index.db").is_file()
    assert read_json(root / ".mcp.json") == {"mcpServers": {"context-engine": {"command": "cce"}}}
    assert not (root / "CLAUDE.md").exists()
