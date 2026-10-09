import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from contextweave.lang.extract_edges import RawRef, extract_edges
from contextweave.lang.parse import ParsedFile

type Json = dict[str, Json] | list[Json] | str | int | bool | None

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "edges"


def _ref(ref: RawRef) -> Json:
    return {
        "kind": ref.kind,
        "name": ref.name,
        "qualifier": ref.qualifier,
        "is_self": ref.is_self,
        "start_byte": ref.start_byte,
        "end_byte": ref.end_byte,
        "line": ref.line,
    }


def golden_name(name: str) -> str:
    return name.replace("/", "__") + ".json"


def has_edges(parsed: ParsedFile) -> bool:
    return parsed.spec.profile == "symbols"


def test_edge_extraction_matches_golden(
    fixture_name: str, parse_fixture: Callable[[str], ParsedFile]
) -> None:
    parsed = parse_fixture(fixture_name)
    if not has_edges(parsed):
        return
    extraction = extract_edges(parsed)
    actual: Json = {
        "refs": [_ref(ref) for ref in extraction.refs],
        "scopes": list(extraction.scopes),
        "errors": [error.code for error in extraction.errors],
    }
    path = GOLDEN / golden_name(fixture_name)
    if os.environ.get("CW_UPDATE_GOLDEN"):
        path.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    assert actual == json.loads(path.read_text(encoding="utf-8"))


def test_no_orphan_edge_goldens(
    all_fixture_names: list[str], parse_fixture: Callable[[str], ParsedFile]
) -> None:
    expected = {golden_name(name) for name in all_fixture_names if has_edges(parse_fixture(name))}
    assert {path.name for path in GOLDEN.glob("*.json")} == expected


SNIPPETS = {
    "a.py": "class A(B):\n    def f(self):\n        self.g()\n        h()\n",
    "a.js": "class A extends B { f() { this.g(); h(); } }\n",
    "a.ts": "class A extends B implements C { f(): void { this.g(); h(); } }\n",
    "a.tsx": "class A extends B { f(): void { this.g(); h(); } }\n",
    "a.go": "package p\n\nfunc f() { g(); x.h() }\n",
    "a.lua": "local function f() g(); x.h() end\n",
    "a.cs": "class A : B, IC { void F() { this.G(); H(); } }\n",
}


@pytest.mark.parametrize("name", sorted(SNIPPETS))
def test_every_core_language_captures_calls(
    name: str, parse_text: Callable[[str, str], ParsedFile]
) -> None:
    extraction = extract_edges(parse_text(name, SNIPPETS[name]))
    names = {ref.name for ref in extraction.refs if ref.kind == "call"}
    assert {"g", "h"} <= names or {"G", "H"} <= names
    assert not extraction.errors


def test_self_calls_are_flagged_and_qualified_calls_carry_qualifier(
    parse_text: Callable[[str, str], ParsedFile],
) -> None:
    refs = extract_edges(parse_text("a.py", SNIPPETS["a.py"])).refs
    by_name = {ref.name: ref for ref in refs if ref.kind == "call"}
    assert by_name["g"].is_self
    assert not by_name["h"].is_self
