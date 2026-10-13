import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from contextweave.core.config import Config
from contextweave.core.errors import ErrorEntry
from contextweave.core.interrupt import deferred_interrupt
from contextweave.core.paths import user_state_dir
from contextweave.core.project import ResolvedProject, resolve_project
from contextweave.daemon.models import ModelRuntime
from contextweave.index.projects import ProjectRegistry

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


@dataclass(frozen=True, slots=True)
class CommandContext:
    """Environment and working directory a command runs against."""

    env: Mapping[str, str]
    cwd: Path

    def registry(self) -> ProjectRegistry:
        return ProjectRegistry(user_state_dir(self.env))


class SubparserRegistry(Protocol):
    """The add_parser surface of an argparse subparsers action."""

    def add_parser(self, name: str, *, help: str) -> argparse.ArgumentParser: ...


def say(message: str) -> None:
    print(message)


def complain(command: str, message: str) -> None:
    print(f"cw {command}: {message}", file=sys.stderr)


def current_project(command: str, ctx: CommandContext) -> ResolvedProject | None:
    """The registered project containing cwd; prints the reason to stderr when there is none."""
    resolution = resolve_project({}, [], ctx.cwd, ctx.registry().known_roots())
    if isinstance(resolution, ResolvedProject):
        return resolution
    complain(command, error_text(resolution))
    return None


def error_text(error: ErrorEntry) -> str:
    message = error.get("message", error["code"])
    detail = error.get("detail")
    hint = detail.get("hint") if detail else None
    return f"{message} (run {hint})" if isinstance(hint, str) else message


def open_models(config: Config, env: Mapping[str, str]) -> ModelRuntime | None:
    """Model clients and derived cache, or None when they cannot be built."""
    try:
        return ModelRuntime.create(config, env)
    except Exception as exc:
        complain("models", f"unavailable, continuing without them: {exc}")
        return None


def close_models(models: ModelRuntime) -> None:
    """Stop the model servers this run launched; a second Ctrl+C waits so none are orphaned."""
    with deferred_interrupt():
        models.close()
