import re
from dataclasses import dataclass
from functools import lru_cache

from tree_sitter import Node, Query

from contextweave.lang.model import LangError
from contextweave.lang.parse import ParsedFile
from contextweave.lang.registry import edge_query_text, load_language
from contextweave.lang.ts import Captures, compile_query, run_matches

REF_KINDS = ("call", "inherit", "implement", "base", "string", "relative")

PATH_LIKE = re.compile(r"^(?=\S*[A-Za-z])\S{3,200}$")


@dataclass(frozen=True, slots=True)
class RawRef:
    """One unresolved reference captured from source; bytes cover the whole reference node."""

    kind: str
    name: str
    qualifier: str | None
    is_self: bool
    start_byte: int
    end_byte: int
    line: int


@dataclass(frozen=True, slots=True)
class EdgeExtraction:
    """References and declared namespaces of one file with the errors met on the way."""

    refs: tuple[RawRef, ...]
    scopes: tuple[str, ...]
    errors: tuple[LangError, ...]


@lru_cache(maxsize=16)
def _compiled(lang: str, source: str) -> Query | None:
    language = load_language(lang)
    return None if language is None else compile_query(language, source)


def extract_edges(parsed: ParsedFile) -> EdgeExtraction:
    """Run the language's edge query over a parsed tree; empty when the language has none."""
    text = edge_query_text(parsed.spec)
    if text is None:
        return EdgeExtraction((), (), ())
    path = str(parsed.path)
    query = _compiled(parsed.spec.lang, text)
    if query is None:
        return EdgeExtraction(
            (),
            (),
            (LangError("bad_query", path, f"edges/{parsed.spec.lang}.scm failed to compile"),),
        )
    try:
        matches = run_matches(query, parsed.tree.root_node)
    except Exception as exc:
        return EdgeExtraction(
            (), (), (LangError("query_failed", path, f"{type(exc).__name__}: {exc}"),)
        )
    refs: dict[tuple[str, int, int, int], RawRef] = {}
    scopes: dict[str, None] = {}
    for captures in matches:
        for namespace in captures.get("scope.namespace", []):
            scopes[parsed.text(namespace)] = None
        ref = _build(captures, parsed)
        if ref is None:
            continue
        key = (ref.kind, ref.start_byte, ref.end_byte, _name_start(captures))
        previous = refs.get(key)
        if previous is None or (ref.is_self and not previous.is_self):
            refs[key] = ref
    ordered = sorted(
        refs.values(), key=lambda ref: (ref.start_byte, ref.end_byte, ref.kind, ref.name)
    )
    return EdgeExtraction(tuple(ordered), tuple(scopes), ())


def _name_start(captures: Captures) -> int:
    names = captures.get("ref.name") or []
    return names[0].start_byte if names else -1


def _build(captures: Captures, parsed: ParsedFile) -> RawRef | None:
    names = captures.get("ref.name") or []
    if not names:
        return None
    for kind in REF_KINDS:
        nodes = captures.get(f"ref.{kind}")
        if nodes:
            return _make(kind, nodes[0], names[0], captures, parsed)
    return None


def _make(
    kind: str, node: Node, name: Node, captures: Captures, parsed: ParsedFile
) -> RawRef | None:
    value = parsed.text(name)
    if kind == "string" and (
        PATH_LIKE.fullmatch(value) is None or not ("." in value or "/" in value)
    ):
        return None
    qualifiers = captures.get("ref.qualifier") or []
    return RawRef(
        kind=kind,
        name=value,
        qualifier=parsed.text(qualifiers[0]) if qualifiers else None,
        is_self=bool(captures.get("ref.self")),
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        line=node.start_point[0] + 1,
    )
