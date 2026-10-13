import shlex
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from typing import IO, Protocol

import httpx

from contextweave.core.config import DeviceEndpointConfig, ModelsConfig
from contextweave.core.registry import Registry
from contextweave.models.errors import ModelTransportError, ProviderSlotError
from contextweave.models.health import Clock, SlotHealth
from contextweave.models.providers import Device, Slot

type Probe = Callable[[str], bool]
type Sleep = Callable[[float], None]

STOP_GRACE_S = 5.0
POLL_INTERVAL_S = 0.5
PROBE_TIMEOUT_S = 5.0
TAIL_LINES = 12
TAIL_SHOWN = 3
READER_JOIN_S = 1.0


class Profile(StrEnum):
    BULK_ANNOTATE = "bulk_annotate"
    BULK_EMBED = "bulk_embed"
    SERVE = "serve"


class ProcessHandle(Protocol):
    def alive(self) -> bool: ...

    def stop(self) -> None: ...

    def exit_code(self) -> int | None: ...

    def tail(self) -> str: ...


class ProcessLauncher(Protocol):
    def start(self, command: str) -> ProcessHandle: ...


class SubprocessHandle:
    """A supervised child process; keeps the last stderr lines so a failed start can say why."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process
        self._tail: deque[str] = deque(maxlen=TAIL_LINES)
        self._reader: threading.Thread | None = None
        if process.stderr is not None:
            self._reader = threading.Thread(
                target=self._drain, args=(process.stderr,), name="cw-server-stderr", daemon=True
            )
            self._reader.start()

    def _drain(self, stream: IO[bytes]) -> None:
        for line in stream:
            text = line.decode(errors="replace").rstrip()
            if text:
                self._tail.append(text)

    def alive(self) -> bool:
        return self._process.poll() is None

    def exit_code(self) -> int | None:
        return self._process.poll()

    def tail(self) -> str:
        if self._reader is not None and not self.alive():
            self._reader.join(READER_JOIN_S)
        return "\n".join(self._tail)

    def stop(self) -> None:
        if not self.alive():
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()


class SubprocessLauncher:
    """Starts launch commands as detached subprocesses."""

    def start(self, command: str) -> ProcessHandle:
        try:
            process = subprocess.Popen(
                shlex.split(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except (OSError, ValueError) as exc:
            raise ModelTransportError(f"cannot launch `{command}`: {exc}") from exc
        return SubprocessHandle(process)


HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"


def http_probe(transport: httpx.BaseTransport | None = None) -> Probe:
    """Probe alive on a 2xx from /health (llama-server: model loaded), else /v1/models when /health 404s (kev.serve)."""

    def probe(endpoint: str) -> bool:
        base = endpoint.rstrip("/")
        try:
            with httpx.Client(transport=transport, timeout=PROBE_TIMEOUT_S) as client:
                response = client.get(f"{base}{HEALTH_PATH}")
                if response.status_code == HTTPStatus.NOT_FOUND:
                    response = client.get(f"{base}{MODELS_PATH}")
        except httpx.HTTPError:
            return False
        return response.is_success

    return probe


class ServerBackend:
    """Supervises one slot's per-device servers: launches on warm, frees other devices once up."""

    def __init__(
        self,
        slot: Slot,
        servers: Mapping[Device, DeviceEndpointConfig],
        launcher: ProcessLauncher,
        probe: Probe,
        *,
        clock: Clock = time.monotonic,
        sleep: Sleep = time.sleep,
        startup_timeout_s: float = 300.0,
        expand: Callable[[str], str] = lambda command: command,
    ) -> None:
        self._slot = slot
        self._expand = expand
        self._servers = servers
        self._launcher = launcher
        self._probe = probe
        self._clock = clock
        self._sleep = sleep
        self._startup_timeout_s = startup_timeout_s
        self._handles: dict[Device, ProcessHandle] = {}

    def launches(self, device: Device) -> bool:
        """Whether this backend starts the device's server itself."""
        server = self._servers.get(device)
        return server is not None and bool(server.launch)

    def warm(self, device: Device) -> None:
        """Start the device's server when launch is set, wait for it, then stop the others."""
        server = self._servers.get(device)
        if server is None:
            raise ProviderSlotError(f"{self._slot} has no server for device '{device}'")
        handle = self._handles.get(device)
        if server.launch and (handle is None or not handle.alive()):
            handle = self._launcher.start(self._expand(server.launch))
            self._handles[device] = handle
        deadline = self._clock() + (self._startup_timeout_s if server.launch else 0.0)
        while not self._probe(server.endpoint):
            if handle is not None and not handle.alive():
                self._handles.pop(device, None)
                raise ModelTransportError(
                    _with_tail(
                        f"{self._slot} {device} server exited with code {handle.exit_code()}",
                        handle,
                    ),
                    device_tried=str(device),
                )
            if self._clock() >= deadline:
                raise ModelTransportError(
                    self._not_answering(device, server, handle), device_tried=str(device)
                )
            self._sleep(POLL_INTERVAL_S)
        for other in [other for other in self._handles if other is not device]:
            self._handles.pop(other).stop()

    def _not_answering(
        self, device: Device, server: DeviceEndpointConfig, handle: ProcessHandle | None
    ) -> str:
        message = f"{self._slot} {device} endpoint {server.endpoint} not answering"
        if not server.launch:
            return (
                f"{message}; start the server yourself or set "
                f"models.{self._slot}.{device}.launch so the daemon starts it"
            )
        message = f"{message} {self._startup_timeout_s:.0f}s after launch"
        return _with_tail(message, handle) if handle is not None else message

    def stop_all(self) -> None:
        """Stop every supervised process; external endpoints are left alone."""
        for handle in self._handles.values():
            handle.stop()
        self._handles.clear()


def _with_tail(message: str, handle: ProcessHandle) -> str:
    lines = handle.tail().splitlines()[-TAIL_SHOWN:]
    return f"{message}: {' | '.join(lines)}" if lines else message


@dataclass(frozen=True, slots=True)
class ResidencyPlan:
    targets: Mapping[Slot, Device | None]


class Residency:
    """Applies a residency profile by moving each slot's provider to its target device."""

    def __init__(
        self,
        models: ModelsConfig,
        healths: Mapping[Slot, SlotHealth],
        backends: Mapping[Slot, ServerBackend],
    ) -> None:
        self._healths = healths
        self._backends = backends
        self._plans: Registry[Profile, ResidencyPlan] = Registry()
        self._plans.register(
            Profile.BULK_ANNOTATE,
            ResidencyPlan({Slot.ANNOTATE: Device.GPU, Slot.EMBED: None, Slot.CLASSIFY: None}),
        )
        self._plans.register(
            Profile.BULK_EMBED,
            ResidencyPlan({Slot.ANNOTATE: None, Slot.EMBED: Device.GPU, Slot.CLASSIFY: None}),
        )
        self._plans.register(
            Profile.SERVE,
            ResidencyPlan(
                {
                    Slot.EMBED: _device(models.embed.query_device),
                    Slot.ANNOTATE: Device.CPU,
                    Slot.CLASSIFY: Device.GPU,
                }
            ),
        )
        self.current: Profile | None = None

    def apply(self, profile: Profile) -> None:
        """Move every slot to the profile's device; blocking, for daemon-level use only."""
        plan = self._plans.get(profile)
        for slot, wanted in plan.targets.items():
            health = self._healths[slot]
            if wanted is None:
                health.suspend()
                backend = self._backends.get(slot)
                if backend is not None:
                    backend.stop_all()
                continue
            health.resume()
            health.activate(_resolve(health, wanted))
        self.current = profile

    def release(self) -> None:
        """Stop every server this process launched."""
        for backend in self._backends.values():
            backend.stop_all()
        self.current = None


def _device(name: str) -> Device:
    try:
        return Device(name)
    except ValueError:
        return Device.CPU


def _resolve(health: SlotHealth, wanted: Device) -> Device:
    if wanted in health.chain or not health.chain:
        return wanted
    return health.chain[0]
