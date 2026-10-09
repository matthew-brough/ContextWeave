import asyncio
import socket
import struct
from dataclasses import dataclass

import msgpack
from pydantic import TypeAdapter, ValidationError

from contextweave.core.types import JSON

HEADER = struct.Struct(">I")
MAX_FRAME_BYTES = 32 * 1024 * 1024

type Message = dict[str, JSON]

_JSON_VALUE: TypeAdapter[JSON] = TypeAdapter(JSON)


class FrameError(Exception):
    """A frame was oversized, truncated or not a msgpack map of JSON values."""


@dataclass(frozen=True, slots=True)
class Request:
    id: int
    method: str
    params: dict[str, JSON]


@dataclass(frozen=True, slots=True)
class WireError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class Response:
    id: int
    result: JSON = None
    error: WireError | None = None


def _json(value: object) -> JSON:
    try:
        return _JSON_VALUE.validate_python(value, strict=True)
    except ValidationError as exc:
        raise FrameError(f"unsupported value: {exc.error_count()} problem(s)") from exc


def pack_frame(message: Message) -> bytes:
    """Length-prefixed msgpack encoding of one message map."""
    try:
        body = msgpack.packb(message, use_bin_type=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FrameError(str(exc)) from exc
    if len(body) > MAX_FRAME_BYTES:
        raise FrameError("frame too large")
    return HEADER.pack(len(body)) + body


def unpack_payload(payload: bytes) -> Message:
    """Decode a frame body into a message map."""
    try:
        decoded = _json(msgpack.unpackb(payload, raw=False))
    except FrameError:
        raise
    except Exception as exc:
        raise FrameError(f"undecodable frame: {exc}") from exc
    if not isinstance(decoded, dict):
        raise FrameError("frame is not a map")
    return decoded


class FrameBuffer:
    """Accumulates bytes and yields complete messages."""

    def __init__(self) -> None:
        self._data = bytearray()

    def feed(self, data: bytes) -> list[Message]:
        self._data.extend(data)
        messages: list[Message] = []
        while len(self._data) >= HEADER.size:
            (length,) = HEADER.unpack_from(self._data)
            if length > MAX_FRAME_BYTES:
                raise FrameError("frame too large")
            end = HEADER.size + length
            if len(self._data) < end:
                break
            messages.append(unpack_payload(bytes(self._data[HEADER.size : end])))
            del self._data[:end]
        return messages


def recv_frame(connection: socket.socket) -> Message | None:
    """Read one message from a blocking socket; None on a clean close before any byte."""
    header = _recv_exact(connection, HEADER.size, allow_eof=True)
    if header is None:
        return None
    (length,) = HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise FrameError("frame too large")
    body = _recv_exact(connection, length, allow_eof=False)
    return unpack_payload(body or b"")


def _recv_exact(connection: socket.socket, count: int, *, allow_eof: bool) -> bytes | None:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = connection.recv(remaining)
        if not chunk:
            if allow_eof and remaining == count:
                return None
            raise FrameError("connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


async def read_frame(reader: asyncio.StreamReader) -> Message | None:
    """Read one message from an asyncio stream; None on a clean close."""
    try:
        header = await reader.readexactly(HEADER.size)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise FrameError("connection closed mid-frame") from exc
    (length,) = HEADER.unpack(header)
    if length > MAX_FRAME_BYTES:
        raise FrameError("frame too large")
    try:
        body = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise FrameError("connection closed mid-frame") from exc
    return unpack_payload(body)


def request_message(request: Request) -> Message:
    return {"id": request.id, "method": request.method, "params": request.params}


def response_message(response: Response) -> Message:
    if response.error is not None:
        error: JSON = {"code": response.error.code, "message": response.error.message}
        return {"id": response.id, "error": error}
    return {"id": response.id, "result": response.result}


def parse_request(message: Message) -> Request | None:
    """Request from a message map, or None when it is malformed."""
    ident, method, params = message.get("id"), message.get("method"), message.get("params", {})
    if (
        isinstance(ident, bool)
        or not isinstance(ident, int)
        or not isinstance(method, str)
        or not isinstance(params, dict)
    ):
        return None
    return Request(ident, method, params)


def parse_response(message: Message) -> Response | None:
    """Response from a message map, or None when it is malformed."""
    ident = message.get("id")
    if isinstance(ident, bool) or not isinstance(ident, int):
        return None
    error = message.get("error")
    if isinstance(error, dict):
        code, text = error.get("code"), error.get("message")
        if isinstance(code, str) and isinstance(text, str):
            return Response(ident, error=WireError(code, text))
        return None
    return Response(ident, result=message.get("result"))
