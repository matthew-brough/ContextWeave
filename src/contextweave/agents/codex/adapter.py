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
from contextweave.agents.codex.config import codex_home, merge_server, unmerge_server
from contextweave.agents.memory_block import plan_block_install, plan_block_removal

CONFIG_FILE = "config.toml"
MEMORY_FILE = "AGENTS.md"


def _changes(candidates: list[FileChange | None]) -> list[FileChange]:
    return [change for change in candidates if change is not None]


class CodexAdapter:
    """Codex: one global [mcp_servers.contextweave] entry plus an AGENTS.md block per project."""

    name = "codex"

    def detect(self, root: Path) -> bool:
        return (root / MEMORY_FILE).exists() or (root / ".codex").exists()

    def hook_events(self) -> Mapping[str, HookSpec]:
        return {}

    def install(self, root: Path, opts: InstallOpts) -> list[FileChange]:
        config_path = codex_home(opts.home, opts.env) / CONFIG_FILE
        config = _changes(
            [plan_change(config_path, merge_server(read_exact(config_path), config_path))]
        )
        memory = _changes([plan_block_install(root / MEMORY_FILE)])
        self._apply(config, memory, config_path, root, opts)
        return [*config, *memory]

    def uninstall(self, root: Path, opts: InstallOpts) -> list[FileChange]:
        config_path = codex_home(opts.home, opts.env) / CONFIG_FILE
        text = read_exact(config_path)
        config = _changes(
            [
                plan_change(config_path, unmerge_server(text, config_path))
                if text is not None
                else None
            ]
        )
        memory = _changes([plan_block_removal(root / MEMORY_FILE)])
        self._apply(config, memory, config_path, root, opts)
        return [*config, *memory]

    @staticmethod
    def _apply(
        config: list[FileChange],
        memory: list[FileChange],
        config_path: Path,
        root: Path,
        opts: InstallOpts,
    ) -> None:
        if opts.dry_run:
            return
        apply_changes(config, config_path.parent.parent)
        apply_changes(memory, root)
