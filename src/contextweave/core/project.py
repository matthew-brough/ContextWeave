import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

from contextweave.core.errors import ErrorEntry
from contextweave.core.refs import RefError, parse_ref, project_key
from contextweave.core.types import JSON
from contextweave.index.root import discover_root

type Via = Literal["arg", "path", "ref", "roots", "cwd"]

_HEX12 = re.compile(r"[0-9a-f]{12}")
_REF_ARGS = ("ref", "target")


class ProjectEcho(TypedDict):
    key: str
    root: str
    via: Via


@dataclass(frozen=True, slots=True)
class ResolvedProject:
    key: str
    root: Path
    via: Via

    def echo(self) -> ProjectEcho:
        """Return the payload form project: {key, root, via}."""
        return ProjectEcho(key=self.key, root=str(self.root), via=self.via)


type Resolution = ResolvedProject | ErrorEntry


def _no_project(known: Mapping[str, Path], message: str) -> ErrorEntry:
    roots: list[JSON] = [str(root) for root in sorted(known.values())]
    return ErrorEntry(code="no_project", message=message, detail={"known": roots})


def _discover(start: Path) -> Path | None:
    try:
        return discover_root(start)
    except OSError, RuntimeError:
        return None


def _finish(root: Path | None, via: Via, known: Mapping[str, Path]) -> Resolution:
    if root is None:
        return _no_project(known, "root not resolvable")
    key = project_key(root)
    registered = known.get(key)
    if registered is None:
        return ErrorEntry(
            code="not_indexed",
            message=f"project at {root} has no index",
            detail={"hint": "cw init", "key": key, "root": str(root)},
        )
    return ResolvedProject(key, registered, via)


def _from_arg(value: str, cwd: Path | None, known: Mapping[str, Path]) -> Resolution:
    if _HEX12.fullmatch(value):
        registered = known.get(value)
        if registered is None:
            return _no_project(known, f"unknown project key {value}")
        return ResolvedProject(value, registered, "arg")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() and cwd is not None:
        candidate = cwd / candidate
    if not candidate.is_dir():
        return _no_project(known, f"project path not found: {value}")
    return _finish(_discover(candidate), "arg", known)


def _from_ref(value: str, known: Mapping[str, Path]) -> Resolution | None:
    try:
        ref = parse_ref(value)
    except RefError:
        return None
    registered = known.get(ref.key)
    if registered is None:
        return _no_project(known, f"unknown project key {ref.key}")
    return ResolvedProject(ref.key, registered, "ref")


def _string(args: Mapping[str, JSON], name: str) -> str | None:
    value = args.get(name)
    return value if isinstance(value, str) and value else None


def resolve_project(
    args: Mapping[str, JSON],
    roots: Sequence[Path],
    cwd: Path | None,
    known: Mapping[str, Path],
) -> Resolution:
    """Resolve the project for one call; first hit wins and no failure raises."""
    if (project := _string(args, "project")) is not None:
        return _from_arg(project, cwd, known)
    path = _string(args, "path")
    if path is not None and Path(path).is_absolute():
        return _finish(_discover(Path(path)), "path", known)
    for name in _REF_ARGS:
        value = _string(args, name)
        if value is not None and (resolution := _from_ref(value, known)) is not None:
            return resolution
    if len(roots) == 1:
        return _finish(_discover(roots[0]), "roots", known)
    if len(roots) > 1:
        return _no_project(known, "multiple MCP roots; pass project")
    if cwd is not None:
        return _finish(_discover(cwd), "cwd", known)
    return _no_project(known, "no project could be determined")
