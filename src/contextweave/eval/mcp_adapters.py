import asyncio
import json
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from contextweave.core.budget import measure_payload
from contextweave.eval.adapters import Adapter, AdapterContext, Served, degraded
from contextweave.eval.jsonvalue import JSON

AST_MCP_COMMAND = "ast-mcp"
CW_COMMAND = "cw"
COVERED_VERDICTS = frozenset({"covered", "complete", "sufficient"})
DISABLE_STAGE_ENV = "CW_EVAL_DISABLE_STAGE"
PROFILE_ENV = "CW_EVAL_PROFILE"
CALL_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class ToolTarget:
    """How to launch an MCP server and which tool answers a query."""

    command: str
    args: tuple[str, ...]
    tool: str
    query_args: Mapping[str, JSON]


def _walk(value: JSON, found: list[dict[str, JSON]]) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("qualified_name"), str):
            found.append(value)
        for child in value.values():
            _walk(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk(child, found)


def extract_items(payload: JSON) -> tuple[str, ...]:
    """Qualified names of every symbol entry in a tool payload, in order, deduplicated."""
    entries: list[dict[str, JSON]] = []
    _walk(payload, entries)
    names = (entry["qualified_name"] for entry in entries)
    return tuple(dict.fromkeys(name for name in names if isinstance(name, str)))


def extract_covered(payload: JSON) -> bool | None:
    """The coverage verdict as covered or not, or None when the payload reports none."""
    if not isinstance(payload, dict):
        return None
    coverage = payload.get("coverage")
    verdict = coverage.get("verdict") if isinstance(coverage, dict) else coverage
    return verdict in COVERED_VERDICTS if isinstance(verdict, str) else None


def extract_model_calls(payload: JSON) -> int:
    """The reported model call count, or zero."""
    if not isinstance(payload, dict):
        return 0
    calls = payload.get("model_calls")
    return calls if isinstance(calls, int) and not isinstance(calls, bool) else 0


def _payload(structured: Mapping[str, object] | None, text: str) -> JSON:
    if structured is not None:
        return json.loads(json.dumps(structured))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def _query(session: ClientSession, target: ToolTarget, query: str) -> Served:
    started = time.perf_counter()
    arguments: dict[str, JSON] = {**target.query_args, "query": query}
    result = await session.call_tool(
        target.tool, arguments, read_timeout_seconds=CALL_TIMEOUT_SECONDS
    )
    elapsed = (time.perf_counter() - started) * 1000
    text = "\n".join(block.text for block in result.content if block.type == "text")
    structured = result.structured_content
    payload = _payload(structured, text)
    tokens = measure_payload(structured if structured is not None else {}, text)
    error = (text or "tool error") if result.is_error else None
    return Served(
        extract_items(payload),
        tokens,
        elapsed,
        extract_model_calls(payload),
        extract_covered(payload),
        error,
    )


async def _drive(
    target: ToolTarget, context: AdapterContext, queries: Sequence[str]
) -> tuple[Served, ...]:
    env = {**context.env, PROFILE_ENV: context.profile}
    if context.ablate is not None:
        env[DISABLE_STAGE_ENV] = context.ablate
    params = StdioServerParameters(
        command=target.command, args=list(target.args), env=env, cwd=context.root
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        if target.tool not in tools:
            return degraded(len(queries), f"tool missing: {target.tool}")
        return tuple([await _query(session, target, query) for query in queries])


class McpAdapter:
    """Drives one MCP stdio server; missing command, tool or crash yields degraded results."""

    def __init__(self, target: ToolTarget, context: AdapterContext) -> None:
        self._target = target
        self._context = context

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]:
        if shutil.which(self._target.command) is None:
            return degraded(len(queries), f"command not found: {self._target.command}")
        try:
            return asyncio.run(_drive(self._target, self._context, queries))
        except Exception as exc:
            return degraded(len(queries), f"{type(exc).__name__}: {exc}")


def ast_mcp_adapter(context: AdapterContext) -> Adapter:
    """ast-mcp search_symbols over stdio."""
    target = ToolTarget(AST_MCP_COMMAND, ("serve",), "search_symbols", {})
    return McpAdapter(target, context)


def cw_adapter(context: AdapterContext) -> Adapter:
    """cw serve --inproc context_search over stdio."""
    target = ToolTarget(CW_COMMAND, ("serve", "--inproc"), "context_search", {})
    return McpAdapter(target, context)
