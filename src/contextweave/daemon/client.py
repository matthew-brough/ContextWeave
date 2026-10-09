import itertools
import socket
from pathlib import Path

from contextweave.core.types import JSON
from contextweave.daemon.wire import (
    FrameError,
    Request,
    pack_frame,
    parse_response,
    recv_frame,
    request_message,
)

DEFAULT_TIMEOUT_S = 30.0


class DaemonUnavailable(Exception):
    """The daemon could not be reached or dropped the connection."""


class RemoteError(Exception):
    """The daemon answered with an error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class IpcClient:
    """Blocking client; each call uses its own short-lived connection."""

    def __init__(self, socket_path: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._path = socket_path
        self._timeout_s = timeout_s
        self._ids = itertools.count(1)

    @property
    def socket_path(self) -> Path:
        return self._path

    def call(
        self, method: str, params: dict[str, JSON] | None = None, *, timeout_s: float | None = None
    ) -> JSON:
        """Send one request and return its result; raises DaemonUnavailable or RemoteError."""
        limit = self._timeout_s if timeout_s is None else timeout_s
        request = Request(next(self._ids), method, params or {})
        try:
            frame = pack_frame(request_message(request))
        except FrameError as exc:
            raise RemoteError("bad_request", str(exc)) from exc
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(limit)
                connection.connect(str(self._path))
                connection.sendall(frame)
                message = recv_frame(connection)
        except (OSError, AttributeError, FrameError) as exc:
            raise DaemonUnavailable(f"{type(exc).__name__}: {exc}") from exc
        if message is None:
            raise DaemonUnavailable("daemon closed the connection")
        response = parse_response(message)
        if response is None or response.id != request.id:
            raise DaemonUnavailable("malformed response")
        if response.error is not None:
            raise RemoteError(response.error.code, response.error.message)
        return response.result
