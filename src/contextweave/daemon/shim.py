from collections.abc import Callable, Mapping
from pathlib import Path

from contextweave.core.types import JSON
from contextweave.daemon.client import DaemonUnavailable, IpcClient, RemoteError
from contextweave.daemon.codec import (
    decode_body,
    decode_resources,
    decode_tool_result,
)
from contextweave.server.context import ToolContext
from contextweave.server.mcp import InprocBackend, ToolBackend, run_backend_stdio
from contextweave.server.resources import ResourceBody, ResourceInfo
from contextweave.server.tools import ToolResult

type Ensure = Callable[[], bool]

DEGRADED_REASON = "daemon_unreachable"


class DaemonBackend:
    """Forwards tool calls to the daemon: respawn and retry once, then answer read-only locally."""

    def __init__(self, client: IpcClient, ensure: Ensure, fallback: ToolBackend, cwd: Path) -> None:
        self._client = client
        self._ensure = ensure
        self._fallback = fallback
        self._cwd = cwd

    def _call(self, method: str, params: dict[str, JSON]) -> JSON:
        try:
            return self._client.call(method, params)
        except DaemonUnavailable:
            if not self._ensure():
                raise
            return self._client.call(method, params)

    def run_tool(
        self, name: str, arguments: Mapping[str, JSON], roots: tuple[Path, ...]
    ) -> ToolResult:
        params: dict[str, JSON] = {
            "name": name,
            "arguments": dict(arguments),
            "cwd": str(self._cwd),
            "roots": [str(root) for root in roots],
        }
        try:
            return decode_tool_result(self._call("tool.call", params))
        except DaemonUnavailable, RemoteError:
            return _degrade(self._fallback.run_tool(name, arguments, roots))

    def list_resources(self) -> list[ResourceInfo]:
        try:
            return decode_resources(self._call("resource.list", {}))
        except DaemonUnavailable, RemoteError:
            return self._fallback.list_resources()

    def read_resource(self, uri: str) -> ResourceBody | None:
        try:
            return decode_body(self._call("resource.read", {"uri": uri}))
        except DaemonUnavailable, RemoteError:
            return self._fallback.read_resource(uri)


def _degrade(result: ToolResult) -> ToolResult:
    entry: JSON = {"stage": "daemon", "reason": DEGRADED_REASON}
    structured = dict(result.structured)
    degraded = structured.get("degraded")
    structured["degraded"] = [*degraded, entry] if isinstance(degraded, list) else [entry]
    return ToolResult(structured, result.text, result.links)


def shim_backend(context: ToolContext, client: IpcClient, ensure: Ensure) -> DaemonBackend:
    """Backend whose fallback is a read-only in-process context (V3 served from memory)."""
    return DaemonBackend(client, ensure, InprocBackend(context), context.cwd or Path.cwd())


def run_shim(backend: ToolBackend) -> None:
    """Speak MCP over stdio, forwarding every call through the backend."""
    run_backend_stdio(backend)
