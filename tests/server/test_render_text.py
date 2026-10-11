from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from contextweave.server.context import ToolContext

from .test_tools import run


def test_outline_folds_signature_into_span_line(context: ToolContext) -> None:
    text = run(context, "file_outline", path="sample.py").text
    assert 'L7-9 def top_level(a: int, b: str = "x") -> bool:' in text
    assert "  L15-17 def render(self, depth: int) -> str:" in text
    assert "sig:" not in text
    assert "imports: os L3, pathlib {Path} L4" in text
    assert "\nimport " not in text


def test_search_hit_keeps_qualified_name_when_signature_lacks_it(context: ToolContext) -> None:
    text = run(context, "search_symbols", query="render").text
    assert "method Widget.render L15-17\n  sig: def render(self, depth: int) -> str:" in text


def test_schema_profile_has_no_signature_line(context: ToolContext) -> None:
    text = run(context, "file_outline", path="sample.json").text
    assert "sig:" not in text
    assert "string server.host L8" in text


def test_ast_query_inlines_one_line_captures(context: ToolContext) -> None:
    text = run(
        context, "ast_query", path="sample.py", query="(function_definition name: (_) @n) @f"
    ).text
    assert "@n L7 top_level" in text
    assert "@f L7-9\ndef top_level" in text
    assert "identifier" not in text


def test_project_line_only_for_roots(
    context: ToolContext, project: Path, make_context: Callable[..., ToolContext]
) -> None:
    assert not run(context, "get_symbol", name="top_level").text.startswith("project ")
    detached = replace(make_context(), cwd=None)
    by_arg = run(detached, "get_symbol", name="top_level", project=str(project))
    assert by_arg.structured["found"] is True
    assert not by_arg.text.startswith("project ")


def test_related_without_edges_says_zero(context: ToolContext) -> None:
    text = run(context, "related", target="top_level", rel="callers").text
    assert text == "0 callers"
