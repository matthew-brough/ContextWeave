from contextweave.core.types import JSON
from contextweave.daemon.client import RemoteError
from contextweave.server.resources import ResourceBody, ResourceInfo
from contextweave.server.tools import ToolResult


def encode_tool_result(result: ToolResult) -> JSON:
    return {"structured": result.structured, "text": result.text}


def decode_tool_result(value: JSON) -> ToolResult:
    """Rebuild a ToolResult from its wire form; RemoteError when the shape is wrong."""
    if not isinstance(value, dict):
        raise RemoteError("bad_response", "tool result is not a map")
    structured, text = value.get("structured"), value.get("text")
    if not isinstance(structured, dict) or not isinstance(text, str):
        raise RemoteError("bad_response", "tool result fields have the wrong types")
    return ToolResult(structured, text)


def encode_resources(items: list[ResourceInfo]) -> JSON:
    return [{"uri": item.uri, "name": item.name, "description": item.description} for item in items]


def decode_resources(value: JSON) -> list[ResourceInfo]:
    if not isinstance(value, list):
        raise RemoteError("bad_response", "resource list is not a list")
    found: list[ResourceInfo] = []
    for item in value:
        if isinstance(item, dict):
            uri, name, description = item.get("uri"), item.get("name"), item.get("description")
            if isinstance(uri, str) and isinstance(name, str) and isinstance(description, str):
                found.append(ResourceInfo(uri, name, description))
    return found


def encode_body(body: ResourceBody | None) -> JSON:
    if body is None:
        return None
    return {"uri": body.uri, "mime_type": body.mime_type, "text": body.text}


def decode_body(value: JSON) -> ResourceBody | None:
    if not isinstance(value, dict):
        return None
    uri, mime, text = value.get("uri"), value.get("mime_type"), value.get("text")
    if isinstance(uri, str) and isinstance(mime, str) and isinstance(text, str):
        return ResourceBody(uri, mime, text)
    return None
