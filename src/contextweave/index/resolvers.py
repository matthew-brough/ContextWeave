import json
import posixpath
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from contextweave.core.registry import Registry
from contextweave.core.types import JSON

JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
STRIPPABLE = (".js", ".jsx", ".mjs", ".cjs")


@dataclass(frozen=True, slots=True)
class ImportSite:
    """One import statement to resolve; level counts the leading dots of a relative import."""

    id: int
    file_id: int
    path: str
    module: str
    names: tuple[str, ...]
    alias: str | None
    kind: str
    line: int
    level: int = 0


@dataclass
class ResolveContext:
    """Project facts a resolver may consult."""

    root: Path
    paths: Mapping[int, str]
    scopes: Mapping[str, tuple[int, ...]]
    file_scopes: Mapping[int, tuple[str, ...]]
    _ids: dict[str, int] = field(init=False, default_factory=dict[str, int])

    def __post_init__(self) -> None:
        self._ids = {path: file_id for file_id, path in self.paths.items()}

    def id_of(self, path: str) -> int | None:
        """File id of an indexed path."""
        return self._ids.get(path)

    def first_existing(self, candidates: tuple[str, ...]) -> tuple[int, ...]:
        """Ids of the first candidate path that is indexed, else empty."""
        for candidate in candidates:
            file_id = self._ids.get(candidate)
            if file_id is not None:
                return (file_id,)
        return ()

    def unique_suffix(self, candidates: tuple[str, ...]) -> tuple[int, ...]:
        """Id of the single indexed file ending in a candidate path, else empty."""
        for candidate in candidates:
            hits = tuple(
                file_id for path, file_id in self._ids.items() if path.endswith("/" + candidate)
            )
            if len(hits) == 1:
                return hits
        return ()

    def in_directory(self, directory: str, suffix: str) -> tuple[int, ...]:
        """Ids of indexed files directly inside a directory with a suffix."""
        return tuple(
            file_id
            for path, file_id in sorted(self._ids.items())
            if posixpath.dirname(path) == directory and path.endswith(suffix)
        )

    def text(self, relative: str) -> str | None:
        """Text of a project file, None when unreadable."""
        try:
            return (self.root / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    @cached_property
    def go_module(self) -> str | None:
        """Module path declared by the root go.mod."""
        text = self.text("go.mod")
        if text is None:
            return None
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "module":
                return parts[1]
        return None

    @cached_property
    def ts_paths(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Alias patterns of tsconfig.json or jsconfig.json joined with baseUrl."""
        for name in ("tsconfig.json", "jsconfig.json"):
            text = self.text(name)
            if text is None:
                continue
            try:
                loaded: JSON = json.loads(text)
            except ValueError:
                continue
            return _ts_aliases(loaded)
        return ()


def _ts_aliases(loaded: JSON) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(loaded, dict):
        return ()
    options = loaded.get("compilerOptions")
    if not isinstance(options, dict):
        return ()
    base = options.get("baseUrl")
    prefix = base if isinstance(base, str) else "."
    paths = options.get("paths")
    if not isinstance(paths, dict):
        return ()
    aliases: list[tuple[str, tuple[str, ...]]] = []
    for pattern, targets in paths.items():
        if isinstance(targets, list):
            joined = tuple(
                posixpath.normpath(posixpath.join(prefix, target))
                for target in targets
                if isinstance(target, str)
            )
            aliases.append((pattern, joined))
    return tuple(aliases)


type ImportResolver = Callable[[ResolveContext, ImportSite], tuple[int, ...]]
type SiblingFinder = Callable[[ResolveContext, int], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class Resolver:
    """Language rules that turn an import into files and name the files sharing a scope."""

    imports: ImportResolver
    siblings: SiblingFinder


def _no_siblings(_context: ResolveContext, _file_id: int) -> tuple[int, ...]:
    return ()


def _python_imports(context: ResolveContext, site: ImportSite) -> tuple[int, ...]:
    module = site.module.replace(".", "/")
    if site.level > 0:
        base = site.path
        for _ in range(site.level):
            base = posixpath.dirname(base)
        joined = posixpath.normpath(posixpath.join(base, module)) if module else base
        return _python_files(context, site, joined)
    roots = ("", "src/")
    for root in roots:
        found = _python_files(context, site, f"{root}{module}".rstrip("/"))
        if found:
            return found
    return context.unique_suffix((f"{module}.py", f"{module}/__init__.py"))


def _python_files(context: ResolveContext, site: ImportSite, stem: str) -> tuple[int, ...]:
    module = context.first_existing((f"{stem}.py", f"{stem}/__init__.py", f"{stem}.pyi"))
    members = (
        tuple(
            file_id
            for name in dict.fromkeys((*site.names, *([site.alias] if site.alias else [])))
            for file_id in context.first_existing(
                (f"{stem}/{name}.py", f"{stem}/{name}/__init__.py")
            )
        )
        if site.kind == "from_import"
        else ()
    )
    return tuple(dict.fromkeys((*module, *members)))


def _js_candidates(base: str) -> tuple[str, ...]:
    stripped = tuple(base[: -len(ext)] for ext in STRIPPABLE if base.endswith(ext))
    stems = (base, *stripped)
    return tuple(
        candidate
        for stem in stems
        for candidate in (
            stem,
            *(stem + ext for ext in JS_EXTENSIONS),
            *(f"{stem}/index{ext}" for ext in JS_EXTENSIONS),
        )
    )


def _js_imports(context: ResolveContext, site: ImportSite) -> tuple[int, ...]:
    module = site.module
    if module.startswith("."):
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(site.path), module))
        return context.first_existing(_js_candidates(joined))
    for pattern, targets in context.ts_paths:
        star = pattern.endswith("*")
        stem = pattern.removesuffix("*")
        if (star and module.startswith(stem)) or module == pattern:
            rest = module.removeprefix(stem) if star else ""
            for target in targets:
                found = context.first_existing(_js_candidates(target.replace("*", rest)))
                if found:
                    return found
    return ()


def _go_imports(context: ResolveContext, site: ImportSite) -> tuple[int, ...]:
    prefix = context.go_module
    if prefix is None or not (site.module == prefix or site.module.startswith(prefix + "/")):
        return ()
    directory = site.module.removeprefix(prefix).strip("/")
    files = context.in_directory(directory, ".go")
    return tuple(file_id for file_id in files if not context.paths[file_id].endswith("_test.go"))


def _go_siblings(context: ResolveContext, file_id: int) -> tuple[int, ...]:
    directory = posixpath.dirname(context.paths[file_id])
    return tuple(
        other
        for other in context.in_directory(directory, ".go")
        if other != file_id and not context.paths[other].endswith("_test.go")
    )


def _lua_imports(context: ResolveContext, site: ImportSite) -> tuple[int, ...]:
    module = site.module.replace(".", "/")
    candidates = (f"{module}.lua", f"{module}/init.lua")
    return context.first_existing(candidates) or context.unique_suffix(candidates)


def _csharp_imports(context: ResolveContext, site: ImportSite) -> tuple[int, ...]:
    direct = context.scopes.get(site.module, ())
    if direct:
        return direct
    owner = site.module.rpartition(".")[0]
    return context.scopes.get(owner, ()) if owner else ()


def _csharp_siblings(context: ResolveContext, file_id: int) -> tuple[int, ...]:
    shared: dict[int, None] = {}
    for scope in context.file_scopes.get(file_id, ()):
        for other in context.scopes.get(scope, ()):
            if other != file_id:
                shared[other] = None
    return tuple(shared)


RESOLVERS: Registry[str, Resolver] = Registry()
RESOLVERS.register("python", Resolver(_python_imports, _no_siblings))
for _lang in ("javascript", "typescript", "tsx"):
    RESOLVERS.register(_lang, Resolver(_js_imports, _no_siblings))
RESOLVERS.register("go", Resolver(_go_imports, _go_siblings))
RESOLVERS.register("lua", Resolver(_lua_imports, _no_siblings))
RESOLVERS.register("csharp", Resolver(_csharp_imports, _csharp_siblings))
