import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec

from contextweave.index.root import GIT_MARKER, STATE_DIR
from contextweave.lang.registry import LangSpec, spec_for_path

BINARY_SNIFF_BYTES = 8192
ALWAYS_SKIPPED = frozenset({GIT_MARKER, STATE_DIR})
GITIGNORE = ".gitignore"


@dataclass(frozen=True, slots=True)
class WalkedFile:
    """A text file under the root; spec is None when no registry row matches."""

    path: str
    mtime_ns: int
    size: int
    spec: LangSpec | None


@dataclass(frozen=True, slots=True)
class _Scope:
    base: str
    spec: GitIgnoreSpec


def is_binary(path: Path) -> bool:
    """True when a NUL byte appears in the first 8 KiB; unreadable files count as binary."""
    try:
        with path.open("rb") as handle:
            return b"\0" in handle.read(BINARY_SNIFF_BYTES)
    except OSError:
        return True


def walk_project(root: Path, ignore: Sequence[str] = ()) -> tuple[WalkedFile, ...]:
    """Every indexable file under root, sorted by posix path; never shells to git."""
    config_spec = GitIgnoreSpec.from_lines(ignore)
    found: list[WalkedFile] = []
    _walk_dir(root, root, "", (), config_spec, found)
    return tuple(sorted(found, key=lambda item: item.path))


def _walk_dir(
    root: Path,
    directory: Path,
    relative: str,
    scopes: tuple[_Scope, ...],
    config_spec: GitIgnoreSpec,
    found: list[WalkedFile],
) -> None:
    scopes = _with_local_gitignore(directory, relative, scopes)
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError:
        return
    for entry in entries:
        if entry.name in ALWAYS_SKIPPED:
            continue
        child = f"{relative}{entry.name}"
        try:
            if entry.is_symlink() and (entry.is_dir() or not _inside(root, Path(entry.path))):
                continue
            is_dir = entry.is_dir(follow_symlinks=False)
            if _ignored(child, is_dir, scopes, config_spec):
                continue
            if is_dir:
                _walk_dir(root, Path(entry.path), f"{child}/", scopes, config_spec, found)
            elif entry.is_file():
                _add_file(entry, child, found)
        except OSError:
            continue


def _add_file(entry: os.DirEntry[str], child: str, found: list[WalkedFile]) -> None:
    if is_binary(Path(entry.path)):
        return
    stat = entry.stat()
    found.append(WalkedFile(child, stat.st_mtime_ns, stat.st_size, spec_for_path(child)))


def _inside(root: Path, candidate: Path) -> bool:
    return candidate.resolve().is_relative_to(root.resolve())


def _with_local_gitignore(
    directory: Path, relative: str, scopes: tuple[_Scope, ...]
) -> tuple[_Scope, ...]:
    local = directory / GITIGNORE
    if not local.is_file():
        return scopes
    try:
        lines = local.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return scopes
    return (*scopes, _Scope(relative, GitIgnoreSpec.from_lines(lines)))


def _ignored(
    relative: str, is_dir: bool, scopes: tuple[_Scope, ...], config_spec: GitIgnoreSpec
) -> bool:
    candidate = f"{relative}/" if is_dir else relative
    if config_spec.check_file(candidate).include:
        return True
    for scope in reversed(scopes):
        verdict = scope.spec.check_file(candidate.removeprefix(scope.base)).include
        if verdict is not None:
            return verdict
    return False
