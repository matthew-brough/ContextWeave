from dataclasses import dataclass
from pathlib import Path

from contextweave.index.walk import walk_project
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.parse import parse_bytes


@dataclass(frozen=True, slots=True)
class SymbolMap:
    """Qualified names of a checkout, by defining path and by name."""

    by_path: dict[str, tuple[str, ...]]
    by_name: dict[str, tuple[str, ...]]

    def paths_of(self, qualified_name: str) -> tuple[str, ...]:
        return self.by_name.get(qualified_name, ())


def build_symbol_map(root: Path) -> SymbolMap:
    """Extract every symbol under root with the lang extractors."""
    by_path: dict[str, tuple[str, ...]] = {}
    by_name: dict[str, list[str]] = {}
    for walked in walk_project(root):
        if walked.spec is None:
            continue
        try:
            source = (root / walked.path).read_bytes()
        except OSError:
            continue
        parsed = parse_bytes(walked.path, source).parsed
        if parsed is None:
            continue
        names = tuple(symbol.qualified_name for symbol in extract_symbols(parsed).symbols)
        by_path[walked.path] = names
        for name in names:
            paths = by_name.setdefault(name, [])
            if walked.path not in paths:
                paths.append(walked.path)
    return SymbolMap(by_path, {name: tuple(paths) for name, paths in by_name.items()})
