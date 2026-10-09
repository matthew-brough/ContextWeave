from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import anyio
import mcp_types as types
from mcp import Client
from mcp.client.session import ClientRequestContext
from pydantic import FileUrl, TypeAdapter

from contextweave.core.types import JSON
from contextweave.server.context import ToolContext
from contextweave.server.mcp import build_server


def drive(context: ToolContext, action: Callable[[Client], Awaitable[None]]) -> None:
    async def main() -> None:
        async with Client(build_server(context)) as client:
            await action(client)

    anyio.run(main)


def structured(result: types.CallToolResult) -> dict[str, JSON]:
    assert not result.is_error
    return TypeAdapter(dict[str, JSON]).validate_python(result.structured_content)


def test_list_tools(context: ToolContext) -> None:
    async def action(client: Client) -> None:
        listed = await client.list_tools()
        assert [tool.name for tool in listed.tools] == [
            "context_search",
            "get_symbol",
            "file_outline",
            "search_symbols",
            "ast_query",
            "expand",
        ]
        assert len(listed.tools) <= 10
        for tool in listed.tools:
            assert tool.annotations is not None and tool.annotations.read_only_hint is True
            assert tool.output_schema is not None
            assert tool.input_schema["type"] == "object"

    drive(context, action)


def test_call_each_tool_validates_structured_content(context: ToolContext, key: str) -> None:
    async def action(client: Client) -> None:
        symbol = await client.call_tool("get_symbol", {"name": "top_level"})
        payload = structured(symbol)
        assert payload["found"] is True
        text = [block for block in symbol.content if isinstance(block, types.TextContent)]
        assert text and "top_level" in text[0].text
        links = [block for block in symbol.content if isinstance(block, types.ResourceLink)]
        assert links and all(link.uri.startswith(f"cw://{key}/") for link in links)

        outline = structured(await client.call_tool("file_outline", {"path": "sample.py"}))
        assert outline["profile"] == "symbols"
        search = structured(await client.call_tool("search_symbols", {"query": "render"}))
        assert search["total_matches"] != 0
        query = structured(
            await client.call_tool(
                "ast_query",
                {"path": "sample.py", "query": "(function_definition name: (_) @n)"},
            )
        )
        assert query["matches"]
        ref = links[0].uri
        expanded = structured(await client.call_tool("expand", {"ref": ref}))
        assert expanded["status"] == "ok"

    drive(context, action)


def test_error_envelope_over_mcp(context: ToolContext) -> None:
    async def action(client: Client) -> None:
        bad = structured(
            await client.call_tool("ast_query", {"path": "sample.py", "query": "(nope) @x"})
        )
        errors = bad["errors"]
        assert isinstance(errors, list) and errors
        unknown = await client.call_tool("nonexistent", {})
        assert unknown.is_error

    drive(context, action)


def test_roots_resolve_project(
    tmp_path: Path, project: Path, make_context: Callable[..., ToolContext]
) -> None:
    base = replace(make_context(), cwd=None)

    async def roots(context: ClientRequestContext) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=FileUrl(project.as_uri()))])

    async def main() -> None:
        async with Client(build_server(base), list_roots_callback=roots, mode="legacy") as client:
            payload = structured(await client.call_tool("get_symbol", {"name": "top_level"}))
            project_echo = payload["project"]
            assert isinstance(project_echo, dict) and project_echo["via"] == "roots"
            assert payload["found"] is True

    anyio.run(main)


def test_no_roots_and_no_cwd_is_no_project(make_context: Callable[..., ToolContext]) -> None:
    base = replace(make_context(), cwd=None)

    async def action(client: Client) -> None:
        payload = structured(await client.call_tool("get_symbol", {"name": "x"}))
        errors = payload["errors"]
        assert isinstance(errors, list)
        first = errors[0]
        assert isinstance(first, dict) and first["code"] == "no_project"

    drive(base, action)


def test_resources(context: ToolContext, key: str) -> None:
    async def action(client: Client) -> None:
        listed = await client.list_resources()
        uris = {str(resource.uri) for resource in listed.resources}
        assert {f"cw://{key}/status", f"cw://{key}/config", f"cw://{key}/languages"} <= uris
        templates = await client.list_resource_templates()
        assert templates.resource_templates

        async def read(uri: str) -> JSON:
            result = await client.read_resource(uri)
            (content,) = result.contents
            assert isinstance(content, types.TextResourceContents)
            return TypeAdapter(JSON).validate_json(content.text)

        status = await read(f"cw://{key}/status")
        assert isinstance(status, dict) and status["files"] == 4
        config = await read(f"cw://{key}/config")
        assert isinstance(config, dict) and "values" in config
        languages = await read(f"cw://{key}/languages")
        assert isinstance(languages, list)
        assert any(isinstance(row, dict) and row["lang"] == "python" for row in languages)

        found = await client.call_tool("get_symbol", {"name": "top_level"})
        symbols = structured(found)["symbols"]
        assert isinstance(symbols, list)
        (item,) = symbols
        assert isinstance(item, dict)
        expanded = await read(str(item["ref"]))
        assert isinstance(expanded, dict) and expanded["status"] == "ok"

    drive(context, action)


def test_pack_resource_for_overflowed_omitted(context: ToolContext, key: str) -> None:
    tight = replace(
        context,
        config=replace(context.config, budget=replace(context.config.budget, omitted_max=3)),
    )

    async def action(client: Client) -> None:
        payload = structured(
            await client.call_tool("search_symbols", {"query": "e", "max_tokens": 400})
        )
        omitted = payload["omitted"]
        assert isinstance(omitted, list)
        tail = omitted[-1]
        assert isinstance(tail, dict) and tail["reason"] == "overflow"
        result = await client.read_resource(str(tail["ref"]))
        (content,) = result.contents
        assert isinstance(content, types.TextResourceContents)
        stored = TypeAdapter(list[JSON]).validate_json(content.text)
        assert len(stored) == int(str(tail["count"]))

    drive(tight, action)
