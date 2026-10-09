from pathlib import Path

CONFIG_FILE = ".contextweave.toml"
STATE_DIR = ".contextweave"
GIT_MARKER = ".git"
SELF_IGNORE = "*\n"


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


def ensure_self_ignore(root: Path) -> Path:
    """Create the state directory with a gitignore of its own; the user's gitignore is untouched."""
    directory = state_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    ignore = directory / ".gitignore"
    if not ignore.is_file() or ignore.read_text(encoding="utf-8") != SELF_IGNORE:
        ignore.write_text(SELF_IGNORE, encoding="utf-8")
    return directory
