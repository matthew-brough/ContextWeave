from collections.abc import Callable

from contextweave.lang.extract_doc import DocExtraction, extract_doc
from contextweave.lang.extract_schema import (
    SCAN_LIMIT,
    SchemaExtraction,
    SchemaNode,
    extract_schema,
)
from contextweave.lang.extract_symbols import extract_symbols
from contextweave.lang.parse import ParsedFile

type Loader = Callable[[str], ParsedFile]
type TextLoader = Callable[[str, str], ParsedFile]

FORBIDDEN_KINDS = {"function", "class", "method", "interface"}


def schema(parse_fixture: Loader, name: str, max_depth: int = 4) -> dict[str, SchemaNode]:
    return {n.key_path: n for n in extract_schema(parse_fixture(name), max_depth).nodes}


def doc(parse_fixture: Loader, name: str) -> DocExtraction:
    return extract_doc(parse_fixture(name))


def test_json_value_types_and_depth_cap(parse_fixture: Loader) -> None:
    nodes = schema(parse_fixture, "data/sample.json")
    assert nodes["name"].kind == "string"
    assert nodes["port"].kind == "number"
    assert nodes["debug"].kind == "bool"
    assert nodes["missing"].kind == "null"
    assert nodes["server.host"].parent == "server"
    assert nodes["routes"].children_count == 2
    assert "routes[].path" in nodes
    assert "routes[1].path" not in nodes
    assert "tags[]" not in nodes
    capped = schema(parse_fixture, "data/sample.json", max_depth=1)
    assert capped["server"].truncated_subtree
    assert "server.host" not in capped


def test_json_byte_range_round_trips(parse_fixture: Loader) -> None:
    parsed = parse_fixture("data/sample.json")
    node = {n.key_path: n for n in extract_schema(parsed).nodes}["server.host"]
    assert "localhost" in parsed.slice(node.start_byte, node.end_byte)


def test_json5_schema_walks_members(parse_fixture: Loader) -> None:
    nodes = schema(parse_fixture, "data/sample.json5")
    assert nodes["name"].kind == "string"
    assert nodes["name"].comment == "// Service name."
    assert nodes["routes"].uniformity == "uniform"
    assert "routes[].method" in nodes


def test_uniformity_is_reported(parse_fixture: Loader) -> None:
    nodes = schema(parse_fixture, "data/mixed.json")
    assert nodes["records"].uniformity == "mixed"
    assert nodes["tokens"].uniformity == "mixed"
    assert nodes["secret"].value_preview is not None
    assert nodes["records"].value_preview is None


def test_yaml_comment_parked_on_previous_sibling(parse_fixture: Loader) -> None:
    nodes = schema(parse_fixture, "data/sample.yaml")
    assert nodes["name"].comment == "# Service configuration."
    assert nodes["database"].comment == "# Database settings."
    assert nodes["database"].children_count == 2


def test_toml_repeated_tables_collapse(parse_fixture: Loader) -> None:
    nodes = schema(parse_fixture, "data/sample.toml")
    assert nodes["server"].kind == "table"
    assert nodes["items[]"].kind == "array"
    assert nodes["items[]"].uniformity is not None


def test_xml_siblings_collapse_with_uniformity(parse_fixture: Loader) -> None:
    result = extract_schema(parse_fixture("data/sample.xml"))
    assert result.nodes
    assert {n.kind for n in result.nodes} <= {"element", "attr"}
    assert any(n.uniformity in ("uniform", "mixed") for n in result.nodes)


def test_csv_columns_never_rows(parse_fixture: Loader, parse_text: TextLoader) -> None:
    nodes = schema(parse_fixture, "data/small.csv")
    assert all(n.value_preview is None and n.parent is None for n in nodes.values())
    rows = "\n".join(f"{i},user{i}" for i in range(SCAN_LIMIT + 5))
    big = extract_schema(parse_text("big.csv", f"id,name\n{rows}\n"))
    assert {n.key_path: n.uniformity for n in big.nodes} == {
        "id": "unverified",
        "name": "unverified",
    }
    assert all(n.children_count == SCAN_LIMIT + 5 for n in big.nodes)


def test_schema_kinds_are_value_types_only(parse_fixture: Loader) -> None:
    for name in (
        "data/sample.json",
        "data/sample.json5",
        "data/sample.yaml",
        "data/sample.toml",
        "data/sample.xml",
        "data/small.csv",
    ):
        result: SchemaExtraction = extract_schema(parse_fixture(name))
        assert not {n.kind for n in result.nodes} & FORBIDDEN_KINDS
        assert all(n.profile == "schema" and n.group == "data" for n in result.nodes)


def test_schema_degrades_on_truncated_input(parse_text: TextLoader) -> None:
    result = extract_schema(parse_text("t.json", '{"a": 1, "b": {"c": 2'))
    assert any(n.key_path == "a" for n in result.nodes)


def test_schema_walker_missing_for_non_schema_lang(parse_fixture: Loader) -> None:
    result = extract_schema(parse_fixture("core/sample.py"))
    assert [e.code for e in result.errors] == ["unsupported_lang"]
    assert result.nodes == ()


def test_markdown_outline_slugs(parse_fixture: Loader) -> None:
    ex = doc(parse_fixture, "docs/sample.md")
    assert [(n.kind, n.slug) for n in ex.nodes] == [
        ("heading", "project-title"),
        ("heading", "alpha"),
        ("heading", "deep-section"),
        ("code_block", "code-python"),
        ("table", "table"),
        ("heading", "alpha-1"),
        ("heading", "second-top"),
        ("heading", "skipped-level"),
    ]
    assert ex.ledes["project-title"] == "The lede paragraph for the title."
    assert not {n.kind for n in ex.nodes} & FORBIDDEN_KINDS
    assert all(n.profile == "outline" and n.group == "docs" for n in ex.nodes)


def test_markdown_links_by_bounded_regex_with_byte_ranges(parse_fixture: Loader) -> None:
    parsed = parse_fixture("docs/sample.md")
    links = extract_doc(parsed).links
    assert [(i.kind, i.module) for i in links] == [
        ("link", "https://example.com/a"),
        ("image", "img/b.png"),
    ]
    for link in links:
        assert parsed.slice(link.start_byte, link.end_byte) == link.module


def test_markdown_link_offsets_are_bytes_not_chars(parse_text: TextLoader) -> None:
    parsed = parse_text("m.md", "# T\n\nnaïve [x](https://e.com/é)\n")
    (link,) = extract_doc(parsed).links
    assert parsed.slice(link.start_byte, link.end_byte) == "https://e.com/é"


def test_html_outline_and_links(parse_fixture: Loader) -> None:
    parsed = parse_fixture("web/sample.html")
    ex = extract_doc(parsed)
    assert ex.nodes
    assert {n.kind for n in ex.nodes} <= {"heading", "landmark", "element"}
    assert all(n.profile == "outline" and n.group == "web" for n in ex.nodes)
    assert ex.links
    assert all(i.profile == "outline" for i in ex.links)


def test_outline_walker_missing_for_non_doc_lang(parse_fixture: Loader) -> None:
    result = extract_doc(parse_fixture("core/sample.py"))
    assert [e.code for e in result.errors] == ["unsupported_lang"]


def test_defs_profile_may_emit_functions_but_outline_and_schema_never(
    parse_fixture: Loader,
) -> None:
    sh = extract_symbols(parse_fixture("scripting/sample.sh"))
    assert any(s.kind == "function" for s in sh.symbols)
    scss = extract_symbols(parse_fixture("web/sample.scss"))
    assert any(s.kind == "function" for s in scss.symbols)
