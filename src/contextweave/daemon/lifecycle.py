import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contextweave.core.paths import daemon_log_path, daemon_pid_path, daemon_socket_path
from contextweave.core.types import JSON
from contextweave.daemon.client import DaemonUnavailable, IpcClient, RemoteError
from contextweave.daemon.pidlock import lock_is_held

PING_TIMEOUT_S = 5.0
START_WAIT_S = 10.0
STOP_WAIT_S = 5.0
POLL_S = 0.05

type Sleep = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class DaemonPaths:
    socket: Path
    pid: Path
    log: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> DaemonPaths:
        return cls(daemon_socket_path(env), daemon_pid_path(env), daemon_log_path(env))


def client_for(env: Mapping[str, str], timeout_s: float | None = None) -> IpcClient:
    """Client for the user's daemon socket."""
    paths = DaemonPaths.from_env(env)
    return IpcClient(paths.socket) if timeout_s is None else IpcClient(paths.socket, timeout_s)


def is_running(env: Mapping[str, str]) -> bool:
    """True when a daemon answers ping on the user's socket."""
    try:
        client_for(env, PING_TIMEOUT_S).call("ping")
    except DaemonUnavailable, RemoteError:
        return False
    return True


def spawn_detached(env: Mapping[str, str]) -> None:
    """Start `cw daemon run` in its own session without waiting for it."""
    paths = DaemonPaths.from_env(env)
    paths.log.parent.mkdir(parents=True, exist_ok=True)
    with paths.log.open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "contextweave.cli", "daemon", "run"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=dict(env),
            start_new_session=True,
        )


def ensure_running(
    env: Mapping[str, str], *, wait_s: float = START_WAIT_S, sleep: Sleep = time.sleep
) -> bool:
    """Spawn the daemon unless one answers, then wait up to wait_s for it to come up."""
    if is_running(env):
        return True
    try:
        spawn_detached(env)
    except OSError:
        return False
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_running(env):
            return True
        sleep(POLL_S)
    return False


def read_pid(env: Mapping[str, str]) -> int | None:
    """Pid recorded in the pidfile, if any."""
    try:
        return int(DaemonPaths.from_env(env).pid.read_text(encoding="utf-8").strip())
    except OSError, ValueError:
        return None


def stop_daemon(env: Mapping[str, str], *, wait_s: float = STOP_WAIT_S) -> bool:
    """Ask the daemon to shut down and wait for it to exit; True when none remains."""
    pid_path = DaemonPaths.from_env(env).pid
    try:
        client_for(env, PING_TIMEOUT_S).call("shutdown")
    except DaemonUnavailable, RemoteError:
        return not lock_is_held(pid_path)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not is_running(env) and not lock_is_held(pid_path):
            return True
        time.sleep(POLL_S)
    return False


def daemon_status(env: Mapping[str, str]) -> dict[str, JSON] | None:
    """The daemon's status map, or None when it is not running."""
    try:
        result = client_for(env, PING_TIMEOUT_S).call("status")
    except DaemonUnavailable, RemoteError:
        return None
    return result if isinstance(result, dict) else None
