import tomllib
from pathlib import Path

import pytest
from agents_support import write_json

from contextweave.agents.base import AgentConfigError, InstallOpts
from contextweave.agents.blocks import find_block
from contextweave.agents.cli_entry import run_init, run_uninstall
from contextweave.agents.codex import CodexAdapter
from contextweave.agents.codex.config import codex_home, merge_server, unmerge_server

USER_CONFIG = (
    '# my codex setup\nmodel = "gpt-5"\n\n[mcp_servers.other]\ncommand = "npx"\n'
    'args = ["-y", "thing"]\n\n[profiles.fast]\nmodel = "mini"\n'
)
SECTION = '[mcp_servers.contextweave]\ncommand = "cw"\nargs = [ "serve",]\n'
PATH = Path("config.toml")


def opts(home: Path, env: dict[str, str] | None = None, *, dry_run: bool = False) -> InstallOpts:
    return InstallOpts(home, dry_run=dry_run, env=env or {})


def entry(config: Path) -> object:
    parsed = tomllib.loads(config.read_text(encoding="utf-8"))
    return parsed["mcp_servers"]["contextweave"]


def test_merge_into_empty_creates_section() -> None:
    merged = merge_server(None, PATH)
    assert tomllib.loads(merged) == {
        "mcp_servers": {"contextweave": {"command": "cw", "args": ["serve"]}}
    }


def test_golden_merge_appends_after_user_content() -> None:
    merged = merge_server(USER_CONFIG, PATH)
    assert merged.startswith(USER_CONFIG + "\n[mcp_servers.contextweave]\n")
    parsed = tomllib.loads(merged)
    assert set(parsed["mcp_servers"]) == {"other", "contextweave"}


def test_golden_round_trip_is_byte_identical() -> None:
    assert unmerge_server(merge_server(USER_CONFIG, PATH), PATH) == USER_CONFIG


def test_merge_is_idempotent() -> None:
    once = merge_server(USER_CONFIG, PATH)
    assert merge_server(once, PATH) == once


def test_unmerge_of_only_our_section_returns_none() -> None:
    assert unmerge_server(merge_server(None, PATH), PATH) is None


def test_unmerge_without_section_is_noop() -> None:
    assert unmerge_server(USER_CONFIG, PATH) == USER_CONFIG


def test_section_in_the_middle_keeps_neighbours() -> None:
    text = (
        '[a]\nx = 1\n\n[mcp_servers.contextweave]\ncommand = "cw"\nargs = ["serve"]\n\n[b]\ny = 2\n'
    )
    assert merge_server(text, PATH) == text
    removed = unmerge_server(text, PATH)
    assert removed is not None
    assert tomllib.loads(removed) == {"a": {"x": 1}, "b": {"y": 2}}


def test_stale_entry_is_replaced_not_duplicated() -> None:
    text = '[mcp_servers.contextweave]\ncommand = "old"\n\n[b]\ny = 2\n'
    merged = merge_server(text, PATH)
    parsed = tomllib.loads(merged)
    assert parsed["mcp_servers"]["contextweave"] == {"command": "cw", "args": ["serve"]}
    assert parsed["b"] == {"y": 2}


def test_user_tuned_current_entry_is_left_alone() -> None:
    text = (
        '[mcp_servers.contextweave]\ncommand = "cw"\nargs = ["serve"]\nstartup_timeout_sec = 30\n'
    )
    assert merge_server(text, PATH) == text


def test_subtable_is_removed_with_its_server() -> None:
    text = SECTION + '[mcp_servers.contextweave.env]\nK = "v"\n\n[b]\ny = 2\n'
    removed = unmerge_server(text, PATH)
    assert removed == "[b]\ny = 2\n"


def test_invalid_toml_raises() -> None:
    with pytest.raises(AgentConfigError):
        merge_server("[broken\n", PATH)


def test_inline_definition_is_refused() -> None:
    with pytest.raises(AgentConfigError):
        merge_server('[mcp_servers]\ncontextweave = { command = "x" }\n', PATH)


def test_codex_home_prefers_env(tmp_path: Path) -> None:
    assert codex_home(tmp_path, {}) == tmp_path / ".codex"
    assert codex_home(tmp_path, {"CODEX_HOME": ""}) == tmp_path / ".codex"
    assert codex_home(tmp_path, {"CODEX_HOME": str(tmp_path / "alt")}) == tmp_path / "alt"


def test_install_writes_global_entry_and_project_block(root: Path, home: Path) -> None:
    changes = CodexAdapter().install(root, opts(home))
    assert {change.path for change in changes} == {
        home / ".codex" / "config.toml",
        root / "AGENTS.md",
    }
    assert entry(home / ".codex" / "config.toml") == {"command": "cw", "args": ["serve"]}
    assert find_block((root / "AGENTS.md").read_text(encoding="utf-8")) is not None
    assert not (root / ".mcp.json").exists()


def test_install_honours_codex_home(root: Path, home: Path, tmp_path: Path) -> None:
    alt = tmp_path / "elsewhere" / "codex"
    CodexAdapter().install(root, opts(home, {"CODEX_HOME": str(alt)}))
    assert entry(alt / "config.toml") == {"command": "cw", "args": ["serve"]}
    assert not (home / ".codex").exists()


def test_two_projects_share_one_global_entry(tmp_path: Path, home: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    adapter = CodexAdapter()
    adapter.install(first, opts(home))
    config = home / ".codex" / "config.toml"
    after_first = config.read_bytes()
    changes = adapter.install(second, opts(home))
    assert config.read_bytes() == after_first
    assert [change.path for change in changes] == [second / "AGENTS.md"]


def test_reinstall_is_idempotent(root: Path, home: Path) -> None:
    adapter = CodexAdapter()
    adapter.install(root, opts(home))
    assert adapter.install(root, opts(home)) == []


def test_round_trip_leaves_user_content_byte_identical(root: Path, home: Path) -> None:
    config = home / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(USER_CONFIG, encoding="utf-8")
    (root / "AGENTS.md").write_text("# mine\n", encoding="utf-8")
    adapter = CodexAdapter()
    adapter.install(root, opts(home))
    adapter.uninstall(root, opts(home))
    assert config.read_text(encoding="utf-8") == USER_CONFIG
    assert (root / "AGENTS.md").read_text(encoding="utf-8") == "# mine\n"


def test_uninstall_removes_created_files_and_empty_codex_dir(root: Path, home: Path) -> None:
    adapter = CodexAdapter()
    adapter.install(root, opts(home))
    adapter.uninstall(root, opts(home))
    assert not (home / ".codex").exists() and not (root / "AGENTS.md").exists()


def test_dry_run_writes_nothing(root: Path, home: Path) -> None:
    changes = CodexAdapter().install(root, opts(home, dry_run=True))
    assert len(changes) == 2
    assert not (home / ".codex").exists() and not (root / "AGENTS.md").exists()


def test_no_hooks() -> None:
    assert CodexAdapter().hook_events() == {}


def test_run_init_and_uninstall_via_cli_entry(root: Path, home: Path, env: dict[str, str]) -> None:
    result = run_init(root, agent="codex", home=home, env=env)
    assert result.exit_code == 0 and result.registered
    assert (home / ".codex" / "config.toml").is_file() and (root / "AGENTS.md").is_file()
    assert not (root / ".mcp.json").exists()
    removed = run_uninstall(root, home=home, agent="codex", env=env)
    assert removed.exit_code == 0 and len(removed.changes) == 2
    assert not (home / ".codex").exists()


def test_run_init_honours_codex_home_env(
    root: Path, home: Path, env: dict[str, str], tmp_path: Path
) -> None:
    env["CODEX_HOME"] = str(tmp_path / "chome")
    assert run_init(root, agent="codex", home=home, env=env).exit_code == 0
    assert (tmp_path / "chome" / "config.toml").is_file() and not (home / ".codex").exists()


def test_run_init_agents_md(root: Path, home: Path, env: dict[str, str]) -> None:
    assert run_init(root, agent="agents_md", home=home, env=env).exit_code == 0
    assert (root / "AGENTS.md").is_file() and (root / ".mcp.json").is_file()
    assert not (home / ".codex").exists()
    assert run_uninstall(root, home=home, agent="agents_md", env=env).exit_code == 0
    assert not (root / "AGENTS.md").exists()


def test_uninstall_all_agents_does_not_duplicate_changes(
    root: Path, home: Path, env: dict[str, str]
) -> None:
    run_init(root, agent="codex", home=home, env=env)
    dry = run_uninstall(root, home=home, dry_run=True, env=env)
    paths = [change.path for change in dry.changes]
    assert len(paths) == len(set(paths)) == 2


def test_parent_server_in_codex_config_blocks_init(
    root: Path, home: Path, env: dict[str, str]
) -> None:
    config = home / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text('[mcp_servers.context-engine]\ncommand = "cce"\n', encoding="utf-8")
    blocked = run_init(root, agent="codex", home=home, env=env)
    assert blocked.exit_code != 0 and blocked.conflicts
    assert "cce uninstall" in blocked.messages[0]
    assert not (root / "AGENTS.md").exists()
    assert run_init(root, agent="codex", home=home, env=env, force=True).exit_code == 0


def test_parent_block_in_agents_md_blocks_init(root: Path, home: Path, env: dict[str, str]) -> None:
    (root / "AGENTS.md").write_text(
        "## Context Engine (CCE)\nuse context_search\n", encoding="utf-8"
    )
    assert run_init(root, agent="agents_md", home=home, env=env).exit_code != 0


def test_parent_mcp_json_blocks_agents_md_init(root: Path, home: Path, env: dict[str, str]) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"ast-mcp": {}}})
    assert run_init(root, agent="agents_md", home=home, env=env).exit_code != 0


def test_malformed_codex_config_fails_before_indexing(
    root: Path, home: Path, env: dict[str, str]
) -> None:
    config = home / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text("[broken\n", encoding="utf-8")
    result = run_init(root, agent="codex", home=home, env=env)
    assert result.exit_code != 0 and not (root / ".contextweave").exists()
