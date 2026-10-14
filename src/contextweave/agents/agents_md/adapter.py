from collections.abc import Mapping
from pathlib import Path

from contextweave.agents.base import (
    FileChange,
    HookSpec,
    InstallOpts,
    apply_changes,
    plan_change,
    read_exact,
)
from contextweave.agents.claude_code.files import merge_mcp, unmerge_mcp
from contextweave.agents.memory_block import plan_block_install, plan_block_removal

MCP_FILE = ".mcp.json"
MEMORY_FILE = "AGENTS.md"


def _changes(candidates: list[FileChange | None]) -> list[FileChange]:
    return [change for change in candidates if change is not None]


class AgentsMdAdapter:
    """Generic agents: .mcp.json and an AGENTS.md routing block, no hooks."""

    name = "agents_md"

    def detect(self, root: Path) -> bool:
        return (root / MEMORY_FILE).exists()

    def hook_events(self, timeout_s: int) -> Mapping[str, HookSpec]:
        return {}

    def install(self, root: Path, opts: InstallOpts) -> list[FileChange]:
        mcp_path = root / MCP_FILE
        changes = _changes(
            [
                plan_change(mcp_path, merge_mcp(read_exact(mcp_path), mcp_path)),
                plan_block_install(root / MEMORY_FILE),
            ]
        )
        if not opts.dry_run:
            apply_changes(changes, root)
        return changes

    def uninstall(self, root: Path, opts: InstallOpts) -> list[FileChange]:
        mcp_path = root / MCP_FILE
        mcp = read_exact(mcp_path)
        changes = _changes(
            [
                plan_change(mcp_path, unmerge_mcp(mcp, mcp_path)) if mcp is not None else None,
                plan_block_removal(root / MEMORY_FILE),
            ]
        )
        if not opts.dry_run:
            apply_changes(changes, root)
        return changes
