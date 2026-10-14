from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from contextweave.core.atomic import write_atomic
from contextweave.core.config import HooksConfig
from contextweave.core.registry import Registry


@dataclass(frozen=True, slots=True)
class InstallOpts:
    """Options for one install or uninstall; home and env locate user-level agent config."""

    home: Path
    plugin: bool = False
    plugin_dir: Path | None = None
    dry_run: bool = False
    env: Mapping[str, str] = field(default_factory=dict[str, str])
    hook_timeout_s: int = HooksConfig().agent_timeout_s


@dataclass(frozen=True, slots=True)
class FileChange:
    """One file edit; before is None when created, after is None when deleted."""

    path: Path
    before: str | None
    after: str | None


@dataclass(frozen=True, slots=True)
class HookSpec:
    """One hook an adapter wires: the agent event, its command and whether the agent waits."""

    event: str
    command: str
    timeout_s: int
    waits: bool


class AgentAdapter(Protocol):
    """Writes and removes one agent's config; the only place agent specifics live."""

    name: str

    def detect(self, root: Path) -> bool: ...

    def install(self, root: Path, opts: InstallOpts) -> list[FileChange]: ...

    def uninstall(self, root: Path, opts: InstallOpts) -> list[FileChange]: ...

    def hook_events(self, timeout_s: int) -> Mapping[str, HookSpec]: ...


type AdapterRegistry = Registry[str, AgentAdapter]


class AgentConfigError(Exception):
    """An agent config file that cannot be merged safely."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path


def read_exact(path: Path) -> str | None:
    """Read a file as utf-8 without newline translation; None when absent."""
    try:
        return path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None
    except UnicodeDecodeError as exc:
        raise AgentConfigError(path, f"not valid utf-8: {exc.reason}") from exc


def _prune_empty_dirs(start: Path, stop: Path) -> None:
    for directory in (start, *start.parents):
        if directory == stop or stop not in directory.parents:
            return
        try:
            directory.rmdir()
        except OSError:
            return


def apply_changes(changes: list[FileChange], stop: Path) -> None:
    """Write or delete every changed file; emptied directories below stop are removed."""
    for change in changes:
        if change.after is None:
            change.path.unlink(missing_ok=True)
            _prune_empty_dirs(change.path.parent, stop)
        else:
            write_atomic(change.path, change.after.encode("utf-8"))


def plan_change(path: Path, after: str | None) -> FileChange | None:
    """Build a FileChange against the file's current content; None when nothing changes."""
    before = read_exact(path)
    if after is not None and before is not None:
        after = _match_line_endings(before, after)
    if before == after:
        return None
    return FileChange(path, before, after)


def _match_line_endings(before: str, after: str) -> str:
    """Write CRLF back to a file that used CRLF, so a Windows user's file round-trips."""
    if "\r\n" not in before or "\r\n" in after:
        return after
    return after.replace("\n", "\r\n")
