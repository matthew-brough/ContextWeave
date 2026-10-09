import json
import re
from pathlib import Path

from agents_support import read_json

from contextweave.agents.base import InstallOpts
from contextweave.agents.claude_code import ClaudeCodeAdapter
from contextweave.agents.cli_entry import run_init, run_uninstall
from contextweave.core.types import JSON

KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RESERVED = ("claude", "anthropic", "cc-plugin")


def build(base: Path, home: Path) -> Path:
    target = base / "plug"
    ClaudeCodeAdapter().install(base, InstallOpts(home, plugin=True, plugin_dir=target))
    return target


def as_object(value: JSON) -> dict[str, JSON]:
    assert isinstance(value, dict)
    return value


def test_manifest_is_valid(root: Path, home: Path) -> None:
    manifest = as_object(read_json(build(root, home) / ".claude-plugin" / "plugin.json"))
    name = manifest["name"]
    assert isinstance(name, str) and KEBAB.match(name)
    assert not name.startswith(RESERVED)
    assert isinstance(manifest["version"], str) and isinstance(manifest["description"], str)
    assert set(manifest) <= {"name", "version", "description", "author"}


def test_components_sit_at_plugin_root_not_in_manifest_dir(root: Path, home: Path) -> None:
    target = build(root, home)
    assert sorted(p.name for p in (target / ".claude-plugin").iterdir()) == ["plugin.json"]
    for rel in ("hooks/hooks.json", ".mcp.json", "skills/contextweave-routing/SKILL.md"):
        assert (target / rel).is_file()


def test_mcp_and_hooks_documents(root: Path, home: Path) -> None:
    target = build(root, home)
    assert read_json(target / ".mcp.json") == {
        "mcpServers": {"contextweave": {"command": "cw", "args": ["serve"]}}
    }
    hooks = as_object(as_object(read_json(target / "hooks" / "hooks.json"))["hooks"])
    assert list(hooks) == ["SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "SessionEnd"]
    for groups in hooks.values():
        assert isinstance(groups, list)
        for group in groups:
            entries = as_object(group)["hooks"]
            assert isinstance(entries, list)
            for hook in entries:
                command = as_object(hook)
                assert command["type"] == "command"
                assert str(command["command"]).startswith("cw hook ")


def test_skill_has_frontmatter_and_no_markers(root: Path, home: Path) -> None:
    text = (build(root, home) / "skills" / "contextweave-routing" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    match = re.match(r"---\nname: (.+)\ndescription: (.+)\n---\n", text)
    assert match is not None and match.group(1) == "contextweave-routing"
    assert "contextweave:begin" not in text and "get_symbol" in text and "context_search" in text


def test_plugin_install_is_idempotent_and_json_parses(root: Path, home: Path) -> None:
    target = build(root, home)
    again = ClaudeCodeAdapter().install(root, InstallOpts(home, plugin=True, plugin_dir=target))
    assert again == []
    for path in target.rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_plugin_leaves_project_config_alone(root: Path, home: Path) -> None:
    build(root, home)
    assert not (root / "CLAUDE.md").exists() and not (root / ".mcp.json").exists()


def test_default_plugin_dir_and_uninstall(root: Path, home: Path, env: dict[str, str]) -> None:
    assert run_init(root, plugin=True, home=home, env=env).exit_code == 0
    plugin = root / ".contextweave" / "plugin"
    assert (plugin / ".claude-plugin" / "plugin.json").is_file()
    assert run_uninstall(root, home=home, agent="claude_code", env=env).exit_code == 0
    assert not plugin.exists()
