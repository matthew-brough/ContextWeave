import json
from pathlib import Path

import pytest
from agents_support import read_json, write_json

from contextweave.agents.base import AgentConfigError, InstallOpts
from contextweave.agents.blocks import find_block
from contextweave.agents.catalog import default_adapters
from contextweave.agents.claude_code.adapter import ClaudeCodeAdapter
from contextweave.core.types import JSON

USER_HOOK: JSON = {"type": "command", "command": "echo mine"}


def opts(
    home: Path, *, dry_run: bool = False, plugin: bool = False, plugin_dir: Path | None = None
) -> InstallOpts:
    return InstallOpts(home, plugin=plugin, plugin_dir=plugin_dir, dry_run=dry_run)


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.parts
    }


def test_registry_holds_every_adapter() -> None:
    adapters = default_adapters()
    assert adapters.names() == ("claude_code", "agents_md", "codex")
    assert adapters.get("claude_code").name == "claude_code"


def test_hook_events_cover_five_events() -> None:
    events = ClaudeCodeAdapter().hook_events()
    assert list(events) == ["SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "SessionEnd"]
    assert events["Stop"].command == "cw hook Stop"
    assert events["SessionStart"].waits and not events["Stop"].waits


def test_detect(root: Path) -> None:
    adapter = ClaudeCodeAdapter()
    assert not adapter.detect(root)
    (root / "CLAUDE.md").write_text("x", encoding="utf-8")
    assert adapter.detect(root)


def test_install_creates_three_files(root: Path, home: Path) -> None:
    changes = ClaudeCodeAdapter().install(root, opts(home))
    assert {change.path.name for change in changes} == {".mcp.json", "CLAUDE.md", "settings.json"}
    assert read_json(root / ".mcp.json") == {
        "mcpServers": {"contextweave": {"command": "cw", "args": ["serve"]}}
    }
    assert find_block((root / "CLAUDE.md").read_text(encoding="utf-8")) is not None
    hooks = read_json(root / ".claude" / "settings.json")
    assert isinstance(hooks, dict) and isinstance(hooks["hooks"], dict)
    assert list(hooks["hooks"]) == [
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "Stop",
        "SessionEnd",
    ]


def test_mcp_merge_preserves_foreign_keys_and_order(root: Path, home: Path) -> None:
    write_json(
        root / ".mcp.json",
        {
            "extra": 1,
            "mcpServers": {"zeta": {"command": "z"}, "alpha": {"command": "a"}},
            "tail": [],
        },
    )
    ClaudeCodeAdapter().install(root, opts(home))
    merged = read_json(root / ".mcp.json")
    assert isinstance(merged, dict)
    assert list(merged) == ["extra", "mcpServers", "tail"]
    servers = merged["mcpServers"]
    assert isinstance(servers, dict)
    assert list(servers) == ["zeta", "alpha", "contextweave"]


def test_install_is_idempotent(root: Path, home: Path) -> None:
    adapter = ClaudeCodeAdapter()
    adapter.install(root, opts(home))
    first = snapshot(root)
    assert adapter.install(root, opts(home)) == []
    assert snapshot(root) == first


def test_settings_merge_keeps_user_hooks(root: Path, home: Path) -> None:
    original: JSON = {
        "model": "x",
        "hooks": {
            "Stop": [{"hooks": [USER_HOOK]}],
            "PreToolUse": [{"matcher": "Bash", "hooks": [USER_HOOK]}],
        },
    }
    write_json(root / ".claude" / "settings.json", original)
    adapter = ClaudeCodeAdapter()
    adapter.install(root, opts(home))
    merged = read_json(root / ".claude" / "settings.json")
    assert isinstance(merged, dict) and isinstance(merged["hooks"], dict)
    stop = merged["hooks"]["Stop"]
    assert isinstance(stop, list) and len(stop) == 2 and stop[0] == {"hooks": [USER_HOOK]}
    assert merged["hooks"]["PreToolUse"] == [{"matcher": "Bash", "hooks": [USER_HOOK]}]
    adapter.uninstall(root, opts(home))
    assert read_json(root / ".claude" / "settings.json") == original


def test_uninstall_reverses_to_original_bytes(root: Path, home: Path) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"other": {"command": "o"}}})
    write_json(root / ".claude" / "settings.json", {"hooks": {"Stop": [{"hooks": [USER_HOOK]}]}})
    (root / "CLAUDE.md").write_text("# Mine\n\nhello\n", encoding="utf-8")
    before = snapshot(root)
    adapter = ClaudeCodeAdapter()
    adapter.install(root, opts(home))
    assert snapshot(root) != before
    adapter.uninstall(root, opts(home))
    assert snapshot(root) == before


def test_uninstall_removes_files_it_created(root: Path, home: Path) -> None:
    before = snapshot(root)
    adapter = ClaudeCodeAdapter()
    adapter.install(root, opts(home))
    adapter.uninstall(root, opts(home))
    assert snapshot(root) == before
    assert not (root / ".claude").exists()


def test_claude_md_block_replaced_in_place(root: Path, home: Path) -> None:
    stale = "intro\n\n<!-- contextweave:begin v=0 -->\nold\n<!-- contextweave:end -->\n\noutro\n"
    (root / "CLAUDE.md").write_text(stale, encoding="utf-8")
    ClaudeCodeAdapter().install(root, opts(home))
    text = (root / "CLAUDE.md").read_text(encoding="utf-8")
    assert text.startswith("intro\n\n<!-- contextweave:begin v=1 -->")
    assert text.endswith("<!-- contextweave:end -->\n\noutro\n")
    assert "old" not in text


def test_dry_run_writes_nothing(root: Path, home: Path) -> None:
    before = snapshot(root)
    changes = ClaudeCodeAdapter().install(root, opts(home, dry_run=True))
    assert len(changes) == 3
    assert all(change.before is None and change.after for change in changes)
    assert snapshot(root) == before
    ClaudeCodeAdapter().install(root, opts(home))
    after_install = snapshot(root)
    removed = ClaudeCodeAdapter().uninstall(root, opts(home, dry_run=True))
    assert len(removed) == 3
    assert snapshot(root) == after_install


def test_plugin_mode_emits_plugin_dir_only(root: Path, home: Path, tmp_path: Path) -> None:
    target = tmp_path / "plugin-out"
    ClaudeCodeAdapter().install(root, opts(home, plugin=True, plugin_dir=target))
    assert not (root / ".mcp.json").exists() and not (root / "CLAUDE.md").exists()
    manifest = read_json(target / ".claude-plugin" / "plugin.json")
    assert isinstance(manifest, dict) and manifest["name"] == "contextweave"
    assert read_json(target / ".mcp.json") == {
        "mcpServers": {"contextweave": {"command": "cw", "args": ["serve"]}}
    }
    hooks = read_json(target / "hooks" / "hooks.json")
    assert isinstance(hooks, dict) and isinstance(hooks["hooks"], dict)
    assert set(hooks["hooks"]) == {
        "SessionStart",
        "UserPromptSubmit",
        "PostToolUse",
        "Stop",
        "SessionEnd",
    }
    assert ClaudeCodeAdapter().install(root, opts(home, plugin=True, plugin_dir=target)) == []


def test_plugin_default_dir_and_uninstall(root: Path, home: Path) -> None:
    ClaudeCodeAdapter().install(root, opts(home, plugin=True))
    default = root / ".contextweave" / "plugin"
    assert (default / ".claude-plugin" / "plugin.json").is_file()
    ClaudeCodeAdapter().uninstall(root, opts(home))
    assert not default.exists()


def test_malformed_json_raises_config_error(root: Path, home: Path) -> None:
    (root / ".mcp.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AgentConfigError):
        ClaudeCodeAdapter().install(root, opts(home))
    assert (root / ".mcp.json").read_text(encoding="utf-8") == "{not json"


def test_wrong_shape_raises_config_error(root: Path, home: Path) -> None:
    (root / ".mcp.json").write_text(json.dumps({"mcpServers": []}), encoding="utf-8")
    with pytest.raises(AgentConfigError):
        ClaudeCodeAdapter().install(root, opts(home))


def test_indent_preserved(root: Path, home: Path) -> None:
    (root / ".mcp.json").write_text('{\n    "a": 1\n}\n', encoding="utf-8")
    ClaudeCodeAdapter().install(root, opts(home))
    assert (
        (root / ".mcp.json")
        .read_text(encoding="utf-8")
        .startswith('{\n    "a": 1,\n    "mcpServers"')
    )
