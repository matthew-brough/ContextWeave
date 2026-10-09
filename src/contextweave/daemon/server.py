import asyncio
import contextlib
import signal
import sys
import time
from collections.abc import Mapping

from contextweave.core.config import Config, load_config
from contextweave.daemon.client import RemoteError
from contextweave.daemon.lifecycle import DaemonPaths
from contextweave.daemon.models import ModelRuntime
from contextweave.daemon.pidlock import PidLock
from contextweave.daemon.service import DaemonService
from contextweave.daemon.wire import (
    FrameError,
    Request,
    Response,
    WireError,
    pack_frame,
    parse_request,
    read_frame,
    response_message,
)

HOUSEKEEPING_S = 1.0
SOCKET_MODE = 0o600


class DaemonServer:
    """Serves the IPC method table on a unix socket until shutdown or idle timeout."""

    def __init__(
        self,
        service: DaemonService,
        paths: DaemonPaths,
        idle_s: float,
        housekeeping_s: float = HOUSEKEEPING_S,
    ) -> None:
        self._service = service
        self._paths = paths
        self._idle_s = idle_s
        self._housekeeping_s = housekeeping_s
        self._connections = 0

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._connections += 1
        try:
            while (message := await read_frame(reader)) is not None:
                request = parse_request(message)
                response = (
                    await self._answer(request)
                    if request is not None
                    else Response(-1, error=WireError("bad_request", "malformed request"))
                )
                writer.write(pack_frame(response_message(response)))
                await writer.drain()
        except FrameError, ConnectionError, asyncio.IncompleteReadError:
            return
        finally:
            self._connections -= 1
            writer.close()
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    async def _answer(self, request: Request) -> Response:
        try:
            result = await asyncio.to_thread(self._service.dispatch, request.method, request.params)
        except RemoteError as exc:
            return Response(request.id, error=WireError(exc.code, exc.message))
        return Response(request.id, result=result)

    async def serve(self) -> None:
        """Bind the socket and run housekeeping until stopped."""
        self._paths.socket.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._paths.socket.unlink(missing_ok=True)
        server = await asyncio.start_unix_server(self._handle, path=str(self._paths.socket))
        self._paths.socket.chmod(SOCKET_MODE)
        loop = asyncio.get_running_loop()
        for name in ("SIGTERM", "SIGINT"):
            with contextlib.suppress(NotImplementedError, ValueError, AttributeError, RuntimeError):
                loop.add_signal_handler(getattr(signal, name), self._service.shutdown.set)
        try:
            while not self._service.shutdown.is_set():
                await asyncio.sleep(self._housekeeping_s)
                self._service.projects.reap_idle()
                if self._idle() and self._connections == 0:
                    break
        finally:
            server.close()
            await server.wait_closed()
            self._paths.socket.unlink(missing_ok=True)

    def _idle(self) -> bool:
        return time.monotonic() - self._service.last_activity >= self._idle_s


def run_daemon(
    env: Mapping[str, str],
    config: Config | None = None,
    *,
    models: ModelRuntime | None = None,
    poll_interval_s: float = 2.0,
    debounce_s: float = 0.3,
    housekeeping_s: float = HOUSEKEEPING_S,
    idle_s: float | None = None,
) -> int:
    """Run the per-user daemon in the foreground; 1 when another instance holds the lock."""
    paths = DaemonPaths.from_env(env)
    loaded = config or load_config(None, strict=False, env=env).config
    lock = PidLock(paths.pid)
    if not lock.acquire():
        print("cw daemon: already running", file=sys.stderr)
        return 1
    service = DaemonService.create(
        env, loaded, models=models, poll_interval_s=poll_interval_s, debounce_s=debounce_s
    )
    limit = loaded.daemon.idle_minutes * 60.0 if idle_s is None else idle_s
    server = DaemonServer(service, paths, limit, housekeeping_s)
    service.start()
    try:
        asyncio.run(server.serve())
    finally:
        service.stop()
        lock.release()
    return 0
