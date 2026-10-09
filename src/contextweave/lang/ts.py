from importlib import import_module

from tree_sitter import Language, Node, Parser, Query, QueryCursor, QueryError, Tree
from tree_sitter_language_pack import get_language

PACK_PREFIX = "pack:"

type Captures = dict[str, list[Node]]


def load_grammar(source: str) -> Language | None:
    """Load a grammar from `pkg.attr` or `pack:name`; None when unavailable."""
    try:
        if source.startswith(PACK_PREFIX):
            return get_language(source.removeprefix(PACK_PREFIX))
        return _from_package(source)
    except Exception:
        return None


def _from_package(source: str) -> Language | None:
    module_name, _, attribute = source.rpartition(".")
    factory: object = getattr(import_module(module_name), attribute)
    if not callable(factory):
        return None
    handle: object = factory()
    return handle if isinstance(handle, Language) else Language(handle)


def parse_tree(language: Language, source: bytes) -> Tree:
    """Parse source bytes with the given grammar."""
    return Parser(language).parse(source)


def compile_query(language: Language, text: str) -> Query | None:
    """Compile a query, None when it does not match the grammar."""
    try:
        return Query(language, text)
    except QueryError:
        return None


def run_matches(query: Query, root: Node) -> list[Captures]:
    """Capture maps of every match of the query under root."""
    return [captures for _index, captures in QueryCursor(query).matches(root)]
