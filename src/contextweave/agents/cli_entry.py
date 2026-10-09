import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contextweave.agents.base import AgentConfigError, FileChange, InstallOpts
from contextweave.agents.catalog import DEFAULT_AGENT, default_adapters
from contextweave.agents.conflicts import Conflict, detect_conflicts, format_conflicts
from contextweave.agents.hooks import run_hook as dispatch_hook
from contextweave.agents.transport import HookTransport, IpcTransport
from contextweave.core.config import Config, ConfigError, load_config
from contextweave.core.paths import user_state_dir
from contextweave.core.registry import UnknownKeyError
from contextweave.daemon.lifecycle import client_for
from contextweave.index.indexer import IndexReport, index_project
from contextweave.index.projects import ProjectRegistry
from contextweave.index.root import ensure_self_ignore

type Indexer = Callable[[Path, Config], IndexReport]

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of init or uninstall for the CLI to print and exit with."""

    exit_code: int
    messages: tuple[str, ...] = ()
    changes: tuple[FileChange, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    registered: bool = False


def _failure(code: int, message: str) -> CommandResult:
    return CommandResult(code, (message,))


def describe_index(report: IndexReport) -> tuple[str, ...]:
    """Summary lines of an init index pass, including model work left to finish."""
    lines = [
        f"indexed: {len(report.added)} added, {len(report.updated)} updated, "
        f"{len(report.removed)} removed, {report.unchanged} unchanged"
    ]
    embed = report.embed
    if embed is not None:
        lines.append(f"embedded {embed.embedded} chunks, {embed.pending} pending")
        lines.extend(f"degraded {entry['stage']}: {entry['reason']}" for entry in embed.degraded)
        if embed.pending:
            lines.append("rerun `cw index` to finish embedding")
    return tuple(lines)


def run_init(
    root: Path,
    agent: str = DEFAULT_AGENT,
    plugin: bool = False,
    force: bool = False,
    dry_run: bool = False,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    plugin_dir: Path | None = None,
    indexer: Indexer = index_project,
) -> CommandResult:
    """Conflict check, self-ignore, full index, register, then install; index precedes register."""
    project = root.resolve()
    environment = os.environ if env is None else env
    user_home = home if home is not None else Path.home()
    try:
        adapter = default_adapters().get(agent)
    except UnknownKeyError:
        return _failure(EXIT_USAGE, f"unknown agent: {agent}")
    try:
        config = load_config(project, strict=True, env=environment).config
    except ConfigError as exc:
        return _failure(EXIT_USAGE, str(exc))
    conflicts = tuple(detect_conflicts(project, user_home))
    if conflicts and not force:
        return CommandResult(EXIT_FAILED, (format_conflicts(list(conflicts)),), (), conflicts)
    opts = InstallOpts(user_home, plugin, plugin_dir, dry_run=True)
    try:
        planned = adapter.install(project, opts)
    except AgentConfigError as exc:
        return _failure(EXIT_FAILED, f"cannot merge agent config: {exc}")
    if dry_run:
        return CommandResult(EXIT_OK, ("dry run: no files written",), tuple(planned), conflicts)
    ensure_self_ignore(project)
    try:
        report = indexer(project, config)
    except Exception as exc:
        return _failure(EXIT_FAILED, f"index failed, project not registered: {exc}")
    ProjectRegistry(user_state_dir(environment)).register(project)
    changes = adapter.install(project, InstallOpts(user_home, plugin, plugin_dir, dry_run=False))
    return CommandResult(
        EXIT_OK, describe_index(report), tuple(changes), conflicts, registered=True
    )


def run_uninstall(
    root: Path,
    home: Path | None = None,
    dry_run: bool = False,
    agent: str | None = None,
) -> CommandResult:
    """Reverse init for agent config only; the index and registry entry stay."""
    project = root.resolve()
    opts = InstallOpts(home if home is not None else Path.home(), dry_run=dry_run)
    adapters = default_adapters()
    try:
        selected = [adapters.get(agent)] if agent is not None else list(adapters.values())
    except UnknownKeyError:
        return _failure(EXIT_USAGE, f"unknown agent: {agent}")
    changes: list[FileChange] = []
    try:
        for adapter in selected:
            changes.extend(adapter.uninstall(project, opts))
    except AgentConfigError as exc:
        return _failure(EXIT_FAILED, f"cannot update agent config: {exc}")
    return CommandResult(EXIT_OK, (), tuple(changes))


def run_hook(
    event: str,
    stdin_bytes: bytes,
    env: Mapping[str, str] | None = None,
    transport: HookTransport | None = None,
) -> tuple[bytes, int]:
    """Entry for `cw hook <event>`: stdout bytes and an exit code that is always 0."""
    environment = os.environ if env is None else env
    chosen = (
        transport if transport is not None else IpcTransport(client_for(environment), environment)
    )
    return dispatch_hook(event, stdin_bytes, env=environment, transport=chosen)
