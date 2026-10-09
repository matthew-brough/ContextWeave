import os
from collections.abc import Callable
from pathlib import Path

from contextweave.core.budget import measure_payload
from contextweave.core.config import Config
from contextweave.core.types import JSON
from contextweave.index.indexer import index_project
from contextweave.server.context import ToolContext
from contextweave.server.tools import TOOLS, ToolResult


def run(context: ToolContext, tool: str, /, **args: JSON) -> ToolResult:
    return TOOLS.get(tool).run(context, args)


def items(result: ToolResult, field: str) -> list[dict[str, JSON]]:
    value = result.structured[field]
    assert isinstance(value, list)
    return [item for item in value if isinstance(item, dict)]


def reindex(project: Path) -> None:
    index_project(project, Config())


def error_codes(result: ToolResult) -> list[str]:
    return [str(error["code"]) for error in items(result, "errors")]


def test_registry_has_seven_tools() -> None:
    assert TOOLS.names() == (
        "context_search",
        "get_symbol",
        "file_outline",
        "search_symbols",
        "ast_query",
        "expand",
        "related",
    )


def test_get_symbol_source(context: ToolContext, key: str) -> None:
    result = run(context, "get_symbol", name="Widget.render", path="sample.py")
    assert result.structured["found"] is True
    (symbol,) = items(result, "symbols")
    assert symbol["qualified_name"] == "Widget.render"
    assert symbol["profile"] == "symbols"
    assert symbol["group"] == "core"
    assert str(symbol["ref"]).startswith(f"cw://{key}/sample.py#Widget.render@")
    assert "def render(self, depth: int)" in str(symbol["source"])
    project = result.structured["project"]
    assert isinstance(project, dict) and project["key"] == key
    assert result.structured["errors"] == []
    assert "Widget.render" in result.text


def test_get_symbol_views(context: ToolContext) -> None:
    signature = items(run(context, "get_symbol", name="top_level", view="signature"), "symbols")[0]
    assert "source" not in signature and "def top_level" in str(signature["signature"])
    doc = items(run(context, "get_symbol", name="top_level", view="doc"), "symbols")[0]
    assert "Top-level function doc" in str(doc["docstring"])
    annotated = run(context, "get_symbol", name="top_level", view="annotation")
    assert items(annotated, "symbols")[0]["annotation"] is None
    assert items(annotated, "degraded")[0]["stage"] == "annotate"


def test_get_symbol_unique_name_is_not_ambiguous(context: ToolContext) -> None:
    result = run(context, "get_symbol", name="Widget", path="sample.py")
    assert result.structured["found"] is True
    assert result.structured["ambiguous"] is False


def test_get_symbol_no_match(context: ToolContext) -> None:
    result = run(context, "get_symbol", name="zzz_missing")
    assert result.structured["found"] is False
    assert result.structured["symbols"] == []


def test_get_symbol_ambiguity_reports_candidates(project: Path, context: ToolContext) -> None:
    (project / "twin.py").write_text(
        "class Twin:\n    pass\n\n\ndef Twin2():\n    pass\n\n\ndef twin():\n    pass\n",
        encoding="utf-8",
    )
    reindex(project)
    result = run(context, "get_symbol", name="twin")
    assert result.structured["ambiguous"] is True
    assert result.structured["found"] is False
    assert len(items(result, "candidates")) >= 2


def test_get_symbol_same_name_and_kind_returns_all(project: Path, context: ToolContext) -> None:
    (project / "a.py").write_text("def shared():\n    return 1\n", encoding="utf-8")
    (project / "b.py").write_text("def shared():\n    return 2\n", encoding="utf-8")
    reindex(project)
    result = run(context, "get_symbol", name="shared")
    assert result.structured["found"] is True
    assert result.structured["ambiguous"] is False
    assert len(items(result, "symbols")) == 2


def test_get_symbol_on_data_and_docs(context: ToolContext) -> None:
    data = run(context, "get_symbol", name="server.host")
    (node,) = items(data, "symbols")
    assert node["profile"] == "schema" and node["group"] == "data"
    heading = run(context, "search_symbols", query="", group="docs")
    assert "bad_args" in error_codes(heading)
    docs = run(context, "search_symbols", query="e", group="docs")
    assert items(docs, "hits")
    assert all(hit["profile"] == "outline" for hit in items(docs, "hits"))


def test_file_outline_symbols(context: ToolContext) -> None:
    result = run(context, "file_outline", path="sample.py")
    assert result.structured["profile"] == "symbols"
    names = {str(node["qualified_name"]) for node in items(result, "symbols")}
    assert {"top_level", "Widget", "undocumented"} <= names
    widget = next(n for n in items(result, "symbols") if n["qualified_name"] == "Widget")
    assert widget["children"]
    assert result.structured["imports"]


def test_file_outline_names_mode(context: ToolContext) -> None:
    result = run(context, "file_outline", path="sample.py", mode="names")
    first = items(result, "symbols")[0]
    assert set(first) <= {"ref", "qualified_name", "start_line", "children"}
    assert "imports" not in result.structured


def test_file_outline_schema_and_outline_profiles(context: ToolContext) -> None:
    schema = run(context, "file_outline", path="sample.json")
    assert schema.structured["profile"] == "schema"
    assert "schema" in schema.structured
    outline = run(context, "file_outline", path="sample.md")
    assert outline.structured["profile"] == "outline"
    assert "outline" in outline.structured


def test_file_outline_depth(context: ToolContext) -> None:
    shallow = run(context, "file_outline", path="sample.py", max_depth=1)
    assert all("children" not in node for node in items(shallow, "symbols"))


def test_search_symbols(context: ToolContext) -> None:
    result = run(context, "search_symbols", query="render")
    hits = items(result, "hits")
    assert {str(hit["qualified_name"]) for hit in hits} >= {"Widget.render", "Render"}
    assert str(hits[0]["name"]).lower() == "render"
    assert result.structured["total_matches"] == len(hits)
    filtered = run(context, "search_symbols", query="render", kind="class")
    assert items(filtered, "hits") == []
    globbed = run(context, "search_symbols", query="e", path_glob="*.go")
    assert all(str(hit["path"]).endswith(".go") for hit in items(globbed, "hits"))
    limited = run(context, "search_symbols", query="e", limit=1)
    assert len(items(limited, "hits")) == 1


def test_ast_query(context: ToolContext) -> None:
    result = run(
        context,
        "ast_query",
        path="sample.py",
        query="(function_definition name: (identifier) @name)",
    )
    texts = {str(match["text"]) for match in items(result, "matches")}
    assert {"top_level", "render", "undocumented"} <= texts
    only = run(
        context,
        "ast_query",
        path="sample.py",
        query="(function_definition name: (identifier) @name)",
        captures=["other"],
    )
    assert items(only, "matches") == []


def test_ast_query_bad_query(context: ToolContext) -> None:
    result = run(context, "ast_query", path="sample.py", query="(not_a_node_type) @x")
    assert "bad_query" in error_codes(result)
    assert result.structured["matches"] == []


def test_expand_ok_moved_gone(project: Path, context: ToolContext) -> None:
    first = run(context, "get_symbol", name="undocumented")
    ref = str(items(first, "symbols")[0]["ref"])
    ok = run(context, "expand", ref=ref)
    assert ok.structured["status"] == "ok"
    assert "def undocumented" in str(ok.structured["source"])

    path = project / "sample.py"
    text = path.read_text(encoding="utf-8")
    path.write_text("# shift\n\n" + text, encoding="utf-8")
    moved = run(context, "expand", ref=ref)
    assert moved.structured["status"] == "ok"

    other = project / "moved.py"
    other.write_text("def undocumented(x):\n    return x\n", encoding="utf-8")
    path.write_text(text.replace("def undocumented(x):\n    return x\n", ""), encoding="utf-8")
    reindex(project)
    relocated = run(context, "expand", ref=ref)
    assert relocated.structured["status"] == "moved"
    assert "moved.py" in str(relocated.structured["current"])

    other.unlink()
    gone = run(context, "expand", ref=ref)
    assert gone.structured["status"] == "gone"


def test_expand_gone_points_to_changed_symbol(project: Path, context: ToolContext) -> None:
    ref = str(items(run(context, "get_symbol", name="undocumented"), "symbols")[0]["ref"])
    path = project / "sample.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace("return x", "return x + 1"), encoding="utf-8"
    )
    result = run(context, "expand", ref=ref)
    assert result.structured["status"] == "gone"
    assert "undocumented" in str(result.structured["current"])


def test_expand_byte_anchor(context: ToolContext) -> None:
    outline = run(context, "ast_query", path="sample.py", query="(import_statement) @i")
    assert items(outline, "matches")
    first = items(outline, "matches")[0]
    from contextweave.core.refs import ByteAnchor, Ref, encode_ref, project_key
    from contextweave.server.nodes import short_hash

    project = outline.structured["project"]
    assert isinstance(project, dict)
    root = Path(str(project["root"]))
    data = (root / "sample.py").read_bytes()
    start, end = int(str(first["start_byte"])), int(str(first["end_byte"]))
    ref = encode_ref(
        Ref(project_key(root), "sample.py", ByteAnchor(start, end), short_hash(data[start:end]))
    )
    result = run(context, "expand", ref=ref)
    assert result.structured["status"] == "ok"
    assert result.structured["source"] == data[start:end].decode()


def test_expand_bad_ref_never_raises(context: ToolContext) -> None:
    result = run(context, "expand", ref="nonsense")
    assert error_codes(result)


def test_stale_file_reflects_new_bytes(project: Path, context: ToolContext) -> None:
    path = project / "sample.py"
    path.write_text(
        path.read_text(encoding="utf-8").replace("return True", "return 'edited-body'"),
        encoding="utf-8",
    )
    result = run(context, "get_symbol", name="top_level")
    assert "edited-body" in str(items(result, "symbols")[0]["source"])


def test_stale_same_size_edit_is_detected(project: Path, context: ToolContext) -> None:
    path = project / "sample.py"
    stat = path.stat()
    path.write_text(
        path.read_text(encoding="utf-8").replace("return True", "return None"), encoding="utf-8"
    )
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    result = run(context, "get_symbol", name="top_level")
    assert "return None" in str(items(result, "symbols")[0]["source"])


def test_stale_new_symbol_found_by_search_readonly(
    project: Path, make_context: Callable[..., ToolContext]
) -> None:
    path = project / "sample.py"
    path.write_text(path.read_text(encoding="utf-8") + "\n\ndef brand_new():\n    pass\n")
    result = run(make_context(readonly=True), "search_symbols", query="brand_new")
    assert [hit["qualified_name"] for hit in items(result, "hits")] == ["brand_new"]
    from contextweave.index.store import Store

    with Store.open(project, readonly=True) as store:
        assert store.symbols_by_name("brand_new") == ()


def test_stale_deleted_file_disappears_from_search(project: Path, context: ToolContext) -> None:
    (project / "sample.go").unlink()
    result = run(context, "search_symbols", query="e", path_glob="*.go")
    assert items(result, "hits") == []


def test_unresolvable_project(tmp_path: Path, context: ToolContext) -> None:
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    from dataclasses import replace

    result = run(replace(context, cwd=None), "get_symbol", name="x")
    assert "no_project" in error_codes(result)
    assert result.structured["project"] is None
    detail = items(result, "errors")[0]["detail"]
    assert isinstance(detail, dict) and detail["known"]
    elsewhere = run(replace(context, cwd=lonely), "file_outline", path="x.py")
    assert error_codes(elsewhere)


def test_bad_args_never_raise(context: ToolContext) -> None:
    result = run(context, "get_symbol")
    assert "bad_args" in error_codes(result)
    result = run(context, "search_symbols", query="x", limit="many")
    assert "bad_args" in error_codes(result)


def test_path_outside_project(context: ToolContext) -> None:
    result = run(context, "file_outline", path="../escape.py")
    assert "outside_project" in error_codes(result)


def test_budget_truncation_with_omitted_refs(context: ToolContext) -> None:
    result = run(context, "search_symbols", query="e", max_tokens=250)
    assert result.structured["truncated"] is True
    assert "hint" in result.structured
    omitted = items(result, "omitted")
    assert omitted and all(str(entry["ref"]).startswith("cw://") for entry in omitted)
    assert measure_payload(result.structured, result.text) <= 250


def test_budget_omitted_overflow_pack(context: ToolContext, key: str) -> None:
    result = run(context, "search_symbols", query="e", max_tokens=120)
    omitted = items(result, "omitted")
    assert result.structured["truncated"] is True
    assert omitted
    assert len(omitted) <= context.config.budget.omitted_max


def test_budget_clips_single_oversized_symbol(project: Path, context: ToolContext) -> None:
    body = "\n".join(f"    value_{index} = {index}" for index in range(400))
    (project / "big.py").write_text(f"def huge():\n{body}\n", encoding="utf-8")
    reindex(project)
    result = run(context, "get_symbol", name="huge", max_tokens=300)
    assert result.structured["truncated"] is True
    assert measure_payload(result.structured, result.text) <= 300
    assert "hint" in result.structured


def test_outline_budget_trims_imports_first(context: ToolContext) -> None:
    full = run(context, "file_outline", path="sample.py")
    used = measure_payload(full.structured, full.text)
    trimmed = run(context, "file_outline", path="sample.py", max_tokens=used - 10)
    assert trimmed.structured["truncated"] is True
    assert measure_payload(trimmed.structured, trimmed.text) <= used - 10
    assert "imports" not in trimmed.structured
    assert any(entry["qualified_name"] == "<imports>" for entry in items(trimmed, "omitted"))


def test_expand_clips_to_budget(project: Path, context: ToolContext) -> None:
    body = "\n".join(f"    value_{index} = {index}" for index in range(400))
    (project / "big.py").write_text(f"def huge():\n{body}\n", encoding="utf-8")
    reindex(project)
    ref = str(items(run(context, "get_symbol", name="huge", view="signature"), "symbols")[0]["ref"])
    result = run(context, "expand", ref=ref, max_tokens=300)
    assert result.structured["truncated"] is True
    assert measure_payload(result.structured, result.text) <= 300
