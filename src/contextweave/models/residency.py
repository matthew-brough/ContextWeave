import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

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
PROBE_TIMEOUT_S = 1.0


class Profile(StrEnum):
    BULK = "bulk"
    SERVE = "serve"


class ProcessHandle(Protocol):
    def alive(self) -> bool: ...

    def stop(self) -> None: ...


class ProcessLauncher(Protocol):
    def start(self, command: str) -> ProcessHandle: ...


class SubprocessHandle:
    """A supervised child process."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self._process = process

    def alive(self) -> bool:
        return self._process.poll() is None

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
        process = subprocess.Popen(
            shlex.split(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return SubprocessHandle(process)


HEALTH_PATH = "/health"


def http_probe(transport: httpx.BaseTransport | None = None) -> Probe:
    """Probe that treats a 2xx from the endpoint's /health as alive (llama-server: model loaded)."""

    def probe(endpoint: str) -> bool:
        try:
            with httpx.Client(transport=transport, timeout=PROBE_TIMEOUT_S) as client:
                response = client.get(f"{endpoint.rstrip('/')}{HEALTH_PATH}")
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
        startup_timeout_s: float = 60.0,
    ) -> None:
        self._slot = slot
        self._servers = servers
        self._launcher = launcher
        self._probe = probe
        self._clock = clock
        self._sleep = sleep
        self._startup_timeout_s = startup_timeout_s
        self._handles: dict[Device, ProcessHandle] = {}

    def warm(self, device: Device) -> None:
        """Start the device's server when launch is set, wait for it, then stop the others."""
        server = self._servers.get(device)
        if server is None:
            raise ProviderSlotError(f"{self._slot} has no server for device '{device}'")
        handle = self._handles.get(device)
        if server.launch and (handle is None or not handle.alive()):
            self._handles[device] = self._launcher.start(server.launch)
        deadline = self._clock() + (self._startup_timeout_s if server.launch else 0.0)
        while not self._probe(server.endpoint):
            if self._clock() >= deadline:
                raise ModelTransportError(
                    f"{self._slot} {device} endpoint {server.endpoint} not answering",
                    device_tried=str(device),
                )
            self._sleep(POLL_INTERVAL_S)
        for other in [other for other in self._handles if other is not device]:
            self._handles.pop(other).stop()

    def stop_all(self) -> None:
        """Stop every supervised process; external endpoints are left alone."""
        for handle in self._handles.values():
            handle.stop()
        self._handles.clear()


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
            Profile.BULK,
            ResidencyPlan({Slot.EMBED: Device.GPU, Slot.ANNOTATE: Device.GPU, Slot.CLASSIFY: None}),
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


def _device(name: str) -> Device:
    try:
        return Device(name)
    except ValueError:
        return Device.CPU


def _resolve(health: SlotHealth, wanted: Device) -> Device:
    if wanted in health.chain or not health.chain:
        return wanted
    return health.chain[0]
