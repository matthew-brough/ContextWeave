import re
import shutil
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import tomli_w

from contextweave.core.paths import MEMORY_DB_NAME
from contextweave.core.refs import project_key
from contextweave.index.root import LOCAL_STATE, STATE_DIR, is_dir_link, remove_dir_link

META_FILE = "meta.toml"
_HEX12 = re.compile(r"[0-9a-f]{12}")


@dataclass(frozen=True, slots=True)
class ProjectEntry:
    """One registered project; missing is true when its root no longer exists."""

    key: str
    root: Path
    created: datetime
    last_used: datetime
    missing: bool


@dataclass(frozen=True, slots=True)
class ForgetResult:
    key: str
    root: Path
    removed_root_state: bool
    kept_memory: bool


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectRegistry:
    """Registry of projects under <state>/projects/<key>/meta.toml."""

    def __init__(self, state: Path, clock: Callable[[], datetime] = _utc_now) -> None:
        self._projects = state / "projects"
        self._clock = clock

    def _dir(self, key: str) -> Path:
        return self._projects / key

    def _write(self, key: str, root: Path, created: datetime, last_used: datetime) -> None:
        directory = self._dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"root": str(root), "created": created, "last_used": last_used}
        (directory / META_FILE).write_text(tomli_w.dumps(meta), encoding="utf-8")

    def _read(self, key: str) -> ProjectEntry | None:
        meta_path = self._dir(key) / META_FILE
        try:
            data = tomllib.loads(meta_path.read_text(encoding="utf-8"))
        except OSError, tomllib.TOMLDecodeError:
            return None
        root, created, last_used = data.get("root"), data.get("created"), data.get("last_used")
        if (
            not isinstance(root, str)
            or not isinstance(created, datetime)
            or not isinstance(last_used, datetime)
        ):
            return None
        root_path = Path(root)
        return ProjectEntry(key, root_path, created, last_used, not root_path.is_dir())

    def register(self, root: Path) -> ProjectEntry:
        """Register a root, or refresh last_used when it is already registered."""
        resolved = root.resolve()
        key = project_key(resolved)
        now = self._clock()
        existing = self._read(key)
        self._write(key, resolved, existing.created if existing else now, now)
        entry = self._read(key)
        if entry is None:
            raise OSError(f"registry write failed for {key}")
        return entry

    def touch(self, key: str) -> ProjectEntry | None:
        """Set last_used to now for a registered key."""
        existing = self._read(key)
        if existing is None:
            return None
        self._write(key, existing.root, existing.created, self._clock())
        return self._read(key)

    def list(self) -> list[ProjectEntry]:
        """Return all registered projects ordered by key."""
        if not self._projects.is_dir():
            return []
        entries = (self._read(child.name) for child in sorted(self._projects.iterdir()))
        return [entry for entry in entries if entry is not None]

    def get(self, ident: str | Path) -> ProjectEntry | None:
        """Look up by 12-hex key or by root path."""
        return self._read(self._key_for(ident))

    def known_roots(self) -> dict[str, Path]:
        """Return key to root for every registered project."""
        return {entry.key: entry.root for entry in self.list()}

    def _key_for(self, ident: str | Path) -> str:
        if isinstance(ident, str) and _HEX12.fullmatch(ident):
            return ident
        return project_key(Path(ident))

    def forget(self, ident: str | Path, *, keep_memory: bool = False) -> ForgetResult | None:
        """Delete a project's root state, user state and registry entry."""
        key = self._key_for(ident)
        entry = self._read(key)
        if entry is None:
            return None
        directory = self._dir(key)
        root_state = entry.root / STATE_DIR
        if is_dir_link(root_state):
            target = root_state.resolve()
            removed_root_state = target.name == LOCAL_STATE and target.parent.name == key
            if removed_root_state:
                remove_dir_link(root_state)
                shutil.rmtree(target, ignore_errors=True)
        else:
            removed_root_state = root_state.is_dir()
            if removed_root_state:
                shutil.rmtree(root_state)
        if keep_memory:
            for child in directory.iterdir():
                if not child.name.startswith(MEMORY_DB_NAME):
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        else:
            shutil.rmtree(directory)
        return ForgetResult(key, entry.root, removed_root_state, keep_memory)

    def forget_missing(self, *, keep_memory: bool = False) -> list[ForgetResult]:
        """Forget every project whose root no longer exists."""
        results = (
            self.forget(entry.key, keep_memory=keep_memory)
            for entry in self.list()
            if entry.missing
        )
        return [result for result in results if result is not None]
