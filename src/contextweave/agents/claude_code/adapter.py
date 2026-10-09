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
from contextweave.agents.claude_code.files import (
    hooks_document,
    mcp_document,
    merge_hooks,
    merge_mcp,
    plugin_manifest,
    routing_skill,
    unmerge_hooks,
    unmerge_mcp,
)
from contextweave.agents.jsonio import Layout, dump_object
from contextweave.agents.memory_block import plan_block_install, plan_block_removal
from contextweave.index.root import STATE_DIR

MCP_FILE = ".mcp.json"
MEMORY_FILE = "CLAUDE.md"
SETTINGS_FILE = Path(".claude") / "settings.json"
PLUGIN_SUBDIR = Path(STATE_DIR) / "plugin"
PLUGIN_MANIFEST = Path(".claude-plugin") / "plugin.json"
PLUGIN_MCP = Path(MCP_FILE)
PLUGIN_HOOKS = Path("hooks") / "hooks.json"
PLUGIN_SKILL = Path("skills") / "contextweave-routing" / "SKILL.md"
HOOK_EVENTS = ("SessionStart", "UserPromptSubmit", "PostToolUse", "Stop", "SessionEnd")
WAITING_EVENTS = frozenset({"SessionStart"})
HOOK_TIMEOUT_S = 1


def _changes(candidates: list[FileChange | None]) -> list[FileChange]:
    return [change for change in candidates if change is not None]


class ClaudeCodeAdapter:
    """Claude Code: .mcp.json, CLAUDE.md routing block, settings hooks, or a plugin directory."""

    name = "claude_code"

    def detect(self, root: Path) -> bool:
        return any((root / marker).exists() for marker in (".claude", MEMORY_FILE, MCP_FILE))

    def hook_events(self) -> Mapping[str, HookSpec]:
        return {
            event: HookSpec(event, f"cw hook {event}", HOOK_TIMEOUT_S, event in WAITING_EVENTS)
            for event in HOOK_EVENTS
        }

    def install(self, root: Path, opts: InstallOpts) -> list[FileChange]:
        changes = (
            self._plan_plugin(self._plugin_dir(root, opts))
            if opts.plugin
            else self._plan_project(root)
        )
        if not opts.dry_run:
            apply_changes(changes, root)
        return changes

    def uninstall(self, root: Path, opts: InstallOpts) -> list[FileChange]:
        changes = [*self._plan_unproject(root), *self._plan_unplugin(self._plugin_dir(root, opts))]
        if not opts.dry_run:
            apply_changes(changes, root)
        return changes

    @staticmethod
    def _plugin_dir(root: Path, opts: InstallOpts) -> Path:
        return opts.plugin_dir if opts.plugin_dir is not None else root / PLUGIN_SUBDIR

    def _plan_project(self, root: Path) -> list[FileChange]:
        mcp_path = root / MCP_FILE
        settings_path = root / SETTINGS_FILE
        return _changes(
            [
                plan_change(mcp_path, merge_mcp(read_exact(mcp_path), mcp_path)),
                plan_block_install(root / MEMORY_FILE),
                plan_change(
                    settings_path,
                    merge_hooks(read_exact(settings_path), settings_path, self.hook_events()),
                ),
            ]
        )

    def _plan_unproject(self, root: Path) -> list[FileChange]:
        mcp_path = root / MCP_FILE
        settings_path = root / SETTINGS_FILE
        mcp = read_exact(mcp_path)
        settings = read_exact(settings_path)
        return _changes(
            [
                plan_change(mcp_path, unmerge_mcp(mcp, mcp_path)) if mcp is not None else None,
                plan_block_removal(root / MEMORY_FILE),
                plan_change(
                    settings_path, unmerge_hooks(settings, settings_path, self.hook_events())
                )
                if settings is not None
                else None,
            ]
        )

    def _plan_plugin(self, target: Path) -> list[FileChange]:
        layout = Layout()
        documents = {
            PLUGIN_MANIFEST: dump_object(plugin_manifest(), layout),
            PLUGIN_MCP: dump_object(mcp_document(), layout),
            PLUGIN_HOOKS: dump_object(hooks_document(self.hook_events()), layout),
            PLUGIN_SKILL: routing_skill(),
        }
        return _changes([plan_change(target / rel, text) for rel, text in documents.items()])

    def _plan_unplugin(self, target: Path) -> list[FileChange]:
        manifest = read_exact(target / PLUGIN_MANIFEST)
        if manifest is None or '"contextweave"' not in manifest:
            return []
        paths = (PLUGIN_MANIFEST, PLUGIN_MCP, PLUGIN_HOOKS, PLUGIN_SKILL)
        return _changes([plan_change(target / rel, None) for rel in paths])
