import os
from collections.abc import Mapping
from pathlib import Path

from contextweave.core.paths import projects_dir
from contextweave.core.refs import project_key

CONFIG_FILE = ".contextweave.toml"
STATE_DIR = ".contextweave"
GIT_MARKER = ".git"
SELF_IGNORE = "*\n"
LOCAL_STATE = "state"


def discover_root(start: Path) -> Path:
    """Nearest ancestor with a contextweave marker, else nearest with .git, else start."""
    origin = start.resolve()
    if origin.is_file():
        origin = origin.parent
    candidates = (origin, *origin.parents)
    for candidate in candidates:
        if (candidate / CONFIG_FILE).is_file() or (candidate / STATE_DIR).is_dir():
            return candidate
    for candidate in candidates:
        if (candidate / GIT_MARKER).exists():
            return candidate
    return origin


def state_dir(root: Path) -> Path:
    """The writable per-project directory."""
    return root / STATE_DIR


def local_state(root: Path, env: Mapping[str, str]) -> Path:
    """Where a root keeps its state; root/.contextweave links here."""
    return projects_dir(env) / project_key(root) / LOCAL_STATE


def ensure_self_ignore(root: Path, env: Mapping[str, str] = os.environ) -> Path:
    """Create the state directory with a gitignore of its own; the user's gitignore is untouched.

    A new state directory lives under the user state dir; root/.contextweave links to it.
    """
    directory = state_dir(root)
    if not directory.exists() and not directory.is_symlink():
        target = local_state(root, env)
        target.mkdir(parents=True, exist_ok=True)
        directory.symlink_to(target, target_is_directory=True)
    directory.mkdir(parents=True, exist_ok=True)
    ignore = directory / ".gitignore"
    if not ignore.is_file() or ignore.read_text(encoding="utf-8") != SELF_IGNORE:
        ignore.write_text(SELF_IGNORE, encoding="utf-8")
    return directory
