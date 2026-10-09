import warnings
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.session import ServerSession
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPDeprecationWarning, MCPError
from mcp.shared.message import ServerMessageMetadata
from pydantic import TypeAdapter, ValidationError

from contextweave.core.types import JSON
from contextweave.server import resources
from contextweave.server.context import ToolContext
from contextweave.server.resources import ResourceBody, ResourceInfo
from contextweave.server.tools import TOOLS, ToolResult, ToolSpec

SERVER_NAME = "contextweave"
INSTRUCTIONS = (
    "Structural code retrieval. Every response declares profile and group; read them before the "
    "payload. Refs (cw://...) returned by one tool are accepted by expand."
)
MAX_TOOLS = 10

type ToolRequest = ServerRequestContext[None]

_ARGUMENTS: TypeAdapter[dict[str, JSON]] = TypeAdapter(dict[str, JSON])


@asynccontextmanager
async def _lifespan(_: Server[None]) -> AsyncGenerator[None]:
    yield


class RootsTracker:
    """Caches the client's MCP roots until the client reports a change."""

    def __init__(self) -> None:
        self._roots: tuple[Path, ...] | None = None

    def invalidate(self) -> None:
        self._roots = None

    async def current(
        self, session: ServerSession, request_id: types.RequestId | None
    ) -> tuple[Path, ...]:
        if self._roots is None:
            fetched = await self._fetch(session, request_id)
            if fetched is None:
                return ()
            self._roots = fetched
        return self._roots

    @staticmethod
    async def _fetch(
        session: ServerSession, request_id: types.RequestId | None
    ) -> tuple[Path, ...] | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MCPDeprecationWarning)
                listed = await session.send_request(
                    types.ListRootsRequest(),
                    types.ListRootsResult,
                    metadata=ServerMessageMetadata(related_request_id=request_id),
                )
        except Exception:
            return None
        found: list[Path] = []
        for root in listed.roots:
            parsed = urlparse(str(root.uri))
            if parsed.scheme == "file":
                found.append(Path(unquote(parsed.path)))
        return tuple(found)


def _tool(spec: ToolSpec) -> types.Tool:
    return types.Tool(
        name=spec.name,
        title=spec.title,
        description=spec.description,
        input_schema=spec.input_schema,
        output_schema=spec.output_schema,
        annotations=types.ToolAnnotations(
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    )


def _result(result: ToolResult) -> types.CallToolResult:
    content: list[types.ContentBlock] = [types.TextContent(type="text", text=result.text)]
    content.extend(types.ResourceLink(uri=uri, name=uri.rsplit("/", 1)[-1]) for uri in result.links)
    return types.CallToolResult(content=content, structured_content=result.structured)


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)], is_error=True
    )


class ToolBackend(Protocol):
    """Where tool bodies and resources actually run: this process or the daemon."""

    def run_tool(
        self, name: str, arguments: Mapping[str, JSON], roots: tuple[Path, ...]
    ) -> ToolResult: ...

    def list_resources(self) -> list[ResourceInfo]: ...

    def read_resource(self, uri: str) -> ResourceBody | None: ...


class InprocBackend:
    """Runs tools in this process against a base context."""

    def __init__(self, base: ToolContext) -> None:
        self._base = base

    def run_tool(
        self, name: str, arguments: Mapping[str, JSON], roots: tuple[Path, ...]
    ) -> ToolResult:
        return TOOLS.get(name).run(replace(self._base, roots=roots), arguments)

    def list_resources(self) -> list[ResourceInfo]:
        return resources.list_resources(self._base)

    def read_resource(self, uri: str) -> ResourceBody | None:
        return resources.read_resource(self._base, uri)


def build_server(base: ToolContext) -> Server[None]:
    """MCP server exposing the registered tools and cw:// resources over a base context."""
    return build_backend_server(InprocBackend(base))


def build_backend_server(backend: ToolBackend) -> Server[None]:
    """MCP server whose tool calls and resource reads go through the given backend."""
    if len(TOOLS) > MAX_TOOLS:
        raise RuntimeError(f"{len(TOOLS)} tools registered; the limit is {MAX_TOOLS}")
    tracker = RootsTracker()

    async def list_tools(
        ctx: ToolRequest, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[_tool(spec) for spec in TOOLS.values()])

    async def call_tool(
        ctx: ToolRequest, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        if params.name not in TOOLS:
            return _error(f"unknown tool: {params.name}")
        try:
            arguments = _ARGUMENTS.validate_python(params.arguments or {})
        except ValidationError as exc:
            return _error(f"bad arguments: {exc.error_count()} problem(s)")
        roots = await tracker.current(ctx.session, ctx.request_id)
        result = await anyio.to_thread.run_sync(backend.run_tool, params.name, arguments, roots)
        return _result(result)

    async def list_resources(
        ctx: ToolRequest, params: types.PaginatedRequestParams | None
    ) -> types.ListResourcesResult:
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    uri=item.uri,
                    name=item.name,
                    description=item.description,
                    mime_type=resources.JSON_MIME,
                )
                for item in await anyio.to_thread.run_sync(backend.list_resources)
            ]
        )

    async def list_templates(
        ctx: ToolRequest, params: types.PaginatedRequestParams | None
    ) -> types.ListResourceTemplatesResult:
        return types.ListResourceTemplatesResult(
            resource_templates=[
                types.ResourceTemplate(
                    uri_template=item.uri_template, name=item.name, description=item.description
                )
                for item in resources.list_templates()
            ]
        )

    async def read_resource(
        ctx: ToolRequest, params: types.ReadResourceRequestParams
    ) -> types.ReadResourceResult:
        body = await anyio.to_thread.run_sync(backend.read_resource, params.uri)
        if body is None:
            raise MCPError(types.INVALID_PARAMS, f"resource not found: {params.uri}")
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(uri=body.uri, mime_type=body.mime_type, text=body.text)
            ]
        )

    async def roots_changed(ctx: ToolRequest, params: types.NotificationParams) -> None:
        tracker.invalidate()

    server = Server(
        SERVER_NAME,
        instructions=INSTRUCTIONS,
        lifespan=_lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_list_resource_templates=list_templates,
        on_read_resource=read_resource,
    )
    server.add_notification_handler(
        "notifications/roots/list_changed", types.NotificationParams, roots_changed
    )
    return server


async def serve_backend(backend: ToolBackend) -> None:
    server = build_backend_server(backend)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


async def serve_stdio(base: ToolContext) -> None:
    await serve_backend(InprocBackend(base))


def run_stdio(base: ToolContext) -> None:
    """Serve over stdio until the client disconnects."""
    anyio.run(serve_stdio, base)


def run_backend_stdio(backend: ToolBackend) -> None:
    """Serve a backend over stdio until the client disconnects."""
    anyio.run(serve_backend, backend)
