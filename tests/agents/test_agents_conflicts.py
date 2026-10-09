from pathlib import Path

from agents_support import write_json

from contextweave.agents.base import InstallOpts
from contextweave.agents.claude_code.adapter import ClaudeCodeAdapter
from contextweave.agents.conflicts import (
    Conflict,
    ConflictKind,
    Scope,
    detect_conflicts,
    format_conflicts,
)

CCE_BLOCK = (
    "## Context Engine (CCE)\n\nYou MUST use `context_search` instead of reading files.\n"
    "Call `session_recall` first.\n"
)
AST_BLOCK = "## Code retrieval\n\nUse the `ast-mcp` tools: `file_outline`, `get_symbol`.\n"
CCE_HOOK = {
    "hooks": {
        "SessionStart": [
            {"hooks": [{"type": "command", "command": "/home/u/.cce/hooks/cce_hook.sh Start"}]}
        ]
    }
}


def kinds(found: list[Conflict]) -> set[tuple[ConflictKind, Scope]]:
    return {(item.kind, item.scope) for item in found}


def test_clean_project_has_no_conflicts(root: Path, home: Path) -> None:
    assert detect_conflicts(root, home) == []


def test_project_mcp_keys(root: Path, home: Path) -> None:
    write_json(
        root / ".mcp.json",
        {"mcpServers": {"context-engine": {"command": "cce"}, "ast-mcp": {}, "fine": {}}},
    )
    found = detect_conflicts(root, home)
    assert [item.what for item in found] == ['MCP server "context-engine"', 'MCP server "ast-mcp"']
    assert "run `cce uninstall`" in found[0].steps
    assert any('remove the "ast-mcp" key' in step for step in found[1].steps)
    assert all(item.path == root / ".mcp.json" for item in found)


def test_cce_key_and_command_detected(root: Path, home: Path) -> None:
    write_json(
        root / ".mcp.json",
        {"mcpServers": {"cce": {}, "renamed": {"command": "/usr/bin/ast-mcp", "args": []}}},
    )
    assert len(detect_conflicts(root, home)) == 2


def test_project_hooks_and_settings_local(root: Path, home: Path) -> None:
    write_json(root / ".claude" / "settings.json", CCE_HOOK)
    write_json(root / ".claude" / "settings.local.json", CCE_HOOK)
    found = detect_conflicts(root, home)
    assert [item.kind for item in found] == [ConflictKind.HOOK, ConflictKind.HOOK]


def test_unrelated_hooks_not_flagged(root: Path, home: Path) -> None:
    write_json(
        root / ".claude" / "settings.json",
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo success"}]}]}},
    )
    assert detect_conflicts(root, home) == []


def test_claude_md_sections(root: Path, home: Path) -> None:
    (root / "CLAUDE.md").write_text(
        f"# Proj\n\nplain\n\n{CCE_BLOCK}\n### Sub\n\nkeep\n\n{AST_BLOCK}\n## Other\n\nfine\n",
        encoding="utf-8",
    )
    found = detect_conflicts(root, home)
    assert [item.what for item in found] == [
        'routing section "## Context Engine (CCE)"',
        'routing section "## Code retrieval"',
    ]
    assert "lines 5-" in found[0].steps[0]
    assert kinds(found) == {(ConflictKind.MEMORY_BLOCK, Scope.PROJECT)}


def test_own_block_is_not_a_conflict(root: Path, home: Path) -> None:
    ClaudeCodeAdapter().install(root, InstallOpts(home))
    assert detect_conflicts(root, home) == []


def test_user_level_sources(root: Path, home: Path) -> None:
    write_json(
        home / ".claude.json",
        {
            "mcpServers": {"context-engine": {}},
            "projects": {
                str(root): {"mcpServers": {"ast-mcp": {}}},
                str(root.parent / "elsewhere"): {"mcpServers": {"ast-mcp": {}}},
            },
        },
    )
    write_json(home / ".claude" / "settings.json", CCE_HOOK)
    (home / ".claude" / "CLAUDE.md").write_text(CCE_BLOCK, encoding="utf-8")
    nested = home / "work" / "proj"
    nested.mkdir(parents=True)
    (home / "CLAUDE.md").write_text(AST_BLOCK, encoding="utf-8")
    found = detect_conflicts(nested, home)
    assert kinds(found) == {
        (ConflictKind.MCP_SERVER, Scope.USER),
        (ConflictKind.HOOK, Scope.USER),
        (ConflictKind.MEMORY_BLOCK, Scope.USER),
    }
    assert sum(item.kind is ConflictKind.MCP_SERVER for item in found) == 1
    assert {item.path.name for item in found} == {".claude.json", "settings.json", "CLAUDE.md"}


def test_per_project_entry_for_this_root(root: Path, home: Path) -> None:
    write_json(
        home / ".claude.json",
        {
            "projects": {
                str(root): {"mcpServers": {"ast-mcp": {}}},
                "/somewhere/else": {"mcpServers": {"context-engine": {}}},
            }
        },
    )
    found = detect_conflicts(root, home)
    assert len(found) == 1 and "projects" in found[0].steps[0]


def test_home_claude_md_ignored_when_root_outside_home(root: Path, home: Path) -> None:
    (home / "CLAUDE.md").write_text(CCE_BLOCK, encoding="utf-8")
    assert detect_conflicts(root, home) == []


def test_unreadable_json_is_skipped(root: Path, home: Path) -> None:
    (root / ".mcp.json").write_text("{broken", encoding="utf-8")
    assert detect_conflicts(root, home) == []


def test_detection_is_read_only(root: Path, home: Path) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"ast-mcp": {}}})
    before = (root / ".mcp.json").read_bytes()
    detect_conflicts(root, home)
    assert (root / ".mcp.json").read_bytes() == before


def test_format_names_steps(root: Path, home: Path) -> None:
    write_json(root / ".mcp.json", {"mcpServers": {"context-engine": {}}})
    report = format_conflicts(detect_conflicts(root, home))
    assert "cce uninstall" in report and "--force" in report
