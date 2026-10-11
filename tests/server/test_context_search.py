from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from contextweave.core.config import Config
from contextweave.core.types import JSON
from contextweave.eval.mcp_adapters import DISABLE_STAGE_ENV
from contextweave.index.indexer import index_project
from contextweave.index.projects import ProjectRegistry
from contextweave.server.context import PackCache, ToolContext, eval_config, inproc_opener
from contextweave.server.resources import read_resource
from contextweave.server.tools import TOOLS, ToolResult

JSON_OBJECT: TypeAdapter[dict[str, JSON]] = TypeAdapter(dict[str, JSON])


def search(context: ToolContext, **arguments: JSON) -> ToolResult:
    return TOOLS.get("context_search").run(context, {"query": "render top level", **arguments})


def list_of(payload: dict[str, JSON], name: str) -> list[JSON]:
    value = payload[name]
    assert isinstance(value, list)
    return value


def test_registered_within_tool_limit() -> None:
    assert "context_search" in TOOLS
    assert len(TOOLS) <= 10
    spec = TOOLS.get("context_search")
    assert spec.input_schema["required"] == ["query"]


def test_pack_and_text_served(context: ToolContext) -> None:
    result = search(context)
    assert list_of(result.structured, "errors") == []
    items = list_of(result.structured, "items")
    assert items
    first = items[0]
    assert isinstance(first, dict) and str(first["ref"]).startswith("cw://")
    assert result.text.startswith("== ")


def test_budget_ceiling_measured_on_served_text(context: ToolContext) -> None:
    result = search(context, max_tokens=512)
    budget = result.structured["budget"]
    assert isinstance(budget, dict)
    assert budget["ceiling"] == 512
    served = budget["served_tokens"]
    assert isinstance(served, int) and served <= 512


def test_scope_and_mode_arguments(context: ToolContext) -> None:
    result = search(context, scope="sample.go", mode="lexical")
    paths = {
        str(item["path"]) for item in list_of(result.structured, "items") if isinstance(item, dict)
    }
    assert paths <= {"sample.go"}


def test_bad_arguments_report_envelope(context: ToolContext) -> None:
    result = TOOLS.get("context_search").run(context, {"mode": "bogus", "query": "x"})
    errors = list_of(result.structured, "errors")
    assert errors and isinstance(errors[0], dict) and errors[0]["code"] == "bad_args"
    assert list_of(result.structured, "items") == []


def test_unknown_project_reports_envelope(context: ToolContext, tmp_path: Path) -> None:
    result = search(context, project=str(tmp_path / "missing"))
    errors = list_of(result.structured, "errors")
    assert errors and list_of(result.structured, "items") == []


def test_omitted_resource_serves_uncapped_list(context: ToolContext, project: Path) -> None:
    bulk = "\n\n".join(
        f"def handler_{n}(value: int) -> int:\n    return value + {n}" for n in range(400)
    )
    (project / "bulk.py").write_text(bulk)
    index_project(project, Config())
    tight = replace(
        context,
        config=replace(context.config, budget=replace(context.config.budget, omitted_max=2)),
    )
    result = search(tight, query="handler value", max_tokens=512)
    omitted = list_of(result.structured, "omitted")
    tail = omitted[-1]
    assert isinstance(tail, dict) and tail["reason"] == "overflow"
    body = read_resource(tight, str(tail["ref"]))
    assert body is not None
    stored = TypeAdapter(list[JSON]).validate_json(body.text)
    assert len(stored) == int(str(tail["count"])) + len(omitted) - 1


def test_eval_disable_stage_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_context: Callable[..., ToolContext]
) -> None:
    monkeypatch.setenv(DISABLE_STAGE_ENV, "dedupe")
    config = Config()
    built = ToolContext(
        config,
        ProjectRegistry(tmp_path / "state"),
        inproc_opener(config, readonly=True),
        packs=PackCache(1),
    )
    assert built.config.pipeline.dedupe is False
    assert built.config.pipeline.gate is True
    monkeypatch.delenv(DISABLE_STAGE_ENV)
    assert make_context().config.pipeline.dedupe is True


def test_eval_config_ignores_unknown_stage() -> None:
    config = Config()
    assert eval_config(config, {DISABLE_STAGE_ENV: "nope"}) is config
    assert eval_config(config, {}) is config
