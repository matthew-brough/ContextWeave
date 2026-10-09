import json
import os
from collections.abc import Callable
from pathlib import Path

from contextweave.lang.extract_doc import DocNode, extract_doc
from contextweave.lang.extract_schema import SchemaNode, extract_schema
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.model import Export, Import, Symbol
from contextweave.lang.parse import ParsedFile

type Json = dict[str, Json] | list[Json] | str | int | bool | None


def _symbol(s: Symbol) -> Json:
    return {
        "name": s.name,
        "qualified_name": s.qualified_name,
        "kind": s.kind,
        "lang": s.lang,
        "group": s.group,
        "profile": s.profile,
        "path": s.path,
        "start_byte": s.start_byte,
        "end_byte": s.end_byte,
        "start_line": s.start_line,
        "end_line": s.end_line,
        "signature": s.signature,
        "docstring": s.docstring,
        "parent": s.parent,
    }


def _import(i: Import) -> Json:
    return {
        "module": i.module,
        "names": list(i.names),
        "alias": i.alias,
        "kind": i.kind,
        "lang": i.lang,
        "group": i.group,
        "profile": i.profile,
        "start_byte": i.start_byte,
        "end_byte": i.end_byte,
        "line": i.line,
    }


def _export(e: Export) -> Json:
    return {
        "name": e.name,
        "kind": e.kind,
        "lang": e.lang,
        "group": e.group,
        "profile": e.profile,
        "start_byte": e.start_byte,
        "end_byte": e.end_byte,
        "line": e.line,
    }


def _schema(n: SchemaNode) -> Json:
    return {
        "key_path": n.key_path,
        "kind": n.kind,
        "lang": n.lang,
        "group": n.group,
        "profile": n.profile,
        "path": n.path,
        "start_byte": n.start_byte,
        "end_byte": n.end_byte,
        "start_line": n.start_line,
        "end_line": n.end_line,
        "value_preview": n.value_preview,
        "comment": n.comment,
        "children_count": n.children_count,
        "truncated_subtree": n.truncated_subtree,
        "parent": n.parent,
        "uniformity": n.uniformity,
    }


def _doc(n: DocNode) -> Json:
    return {
        "title": n.title,
        "slug": n.slug,
        "level": n.level,
        "kind": n.kind,
        "lang": n.lang,
        "group": n.group,
        "profile": n.profile,
        "path": n.path,
        "start_byte": n.start_byte,
        "end_byte": n.end_byte,
        "start_line": n.start_line,
        "end_line": n.end_line,
        "info": n.info,
        "parent": n.parent,
    }


def snapshot(parsed: ParsedFile) -> Json:
    """Every extracted record of a file as plain JSON."""
    profile = parsed.spec.profile
    if profile == "schema":
        schema = extract_schema(parsed)
        return {
            "profile": profile,
            "nodes": [_schema(n) for n in schema.nodes],
            "errors": [e.code for e in schema.errors],
        }
    if profile == "outline":
        doc = extract_doc(parsed)
        return {
            "profile": profile,
            "nodes": [_doc(n) for n in doc.nodes],
            "links": [_import(i) for i in doc.links],
            "ledes": dict(doc.ledes),
            "errors": [e.code for e in doc.errors],
        }
    symbols = extract_symbols(parsed)
    return {
        "profile": profile,
        "symbols": [_symbol(s) for s in symbols.symbols],
        "imports": [_import(i) for i in symbols.imports],
        "exports": [_export(e) for e in symbols.exports],
        "errors": [e.code for e in symbols.errors],
    }


def golden_name(name: str) -> str:
    return name.replace("/", "__") + ".json"


def test_extraction_matches_golden(
    fixture_name: str, golden_dir: Path, parse_fixture: Callable[[str], ParsedFile]
) -> None:
    actual = snapshot(parse_fixture(fixture_name))
    path = golden_dir / golden_name(fixture_name)
    if os.environ.get("CW_UPDATE_GOLDEN"):
        path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    expected: Json = json.loads(path.read_text(encoding="utf-8"))
    assert actual == expected


def test_no_orphan_goldens(golden_dir: Path, all_fixture_names: list[str]) -> None:
    expected = {golden_name(name) for name in all_fixture_names}
    assert {p.name for p in golden_dir.glob("*.json")} == expected
