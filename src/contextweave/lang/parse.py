from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Tree

from contextweave.lang.model import LangError
from contextweave.lang.registry import LangSpec, load_language, spec_for_path
from contextweave.lang.ts import parse_tree


@dataclass(frozen=True, slots=True)
class ParsedFile:
    """Source bytes with their syntax tree; every range indexes into source."""

    path: Path
    spec: LangSpec
    source: bytes
    tree: Tree

    def slice(self, start_byte: int, end_byte: int) -> str:
        """Exact source text of a byte range."""
        return self.source[start_byte:end_byte].decode("utf-8", errors="replace")

    def text(self, node: Node) -> str:
        """Exact source text of a node."""
        return self.slice(node.start_byte, node.end_byte)


@dataclass(frozen=True, slots=True)
class ParseResult:
    """Parsed file, or None with the reason in errors; errors may accompany a parse."""

    parsed: ParsedFile | None
    errors: tuple[LangError, ...]


def parse_bytes(path: str | Path, source: bytes) -> ParseResult:
    """Parse source bytes by the registry row of path; never raises."""
    candidate = Path(path)
    key = str(candidate)
    spec = spec_for_path(candidate)
    if spec is None:
        return ParseResult(
            None, (LangError("unsupported_lang", key, f"no registry row for {candidate.name}"),)
        )
    language = load_language(spec.lang)
    if language is None:
        return ParseResult(None, (LangError("no_grammar", key, f"could not load {spec.source}"),))
    try:
        tree = parse_tree(language, source)
    except Exception as exc:
        return ParseResult(None, (LangError("parse_failed", key, f"{type(exc).__name__}: {exc}"),))
    parsed = ParsedFile(candidate, spec, source, tree)
    syntax = (
        ()
        if not tree.root_node.has_error
        else (LangError("syntax_error", key, "tree contains ERROR nodes"),)
    )
    return ParseResult(parsed, syntax)


def parse_file(path: str | Path) -> ParseResult:
    """Read and parse a file; never raises."""
    candidate = Path(path)
    try:
        source = candidate.read_bytes()
    except OSError as exc:
        return ParseResult(None, (LangError("unreadable", str(candidate), type(exc).__name__),))
    return parse_bytes(candidate, source)
