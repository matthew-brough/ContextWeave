from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import httpx

from contextweave.core.config import DeviceEndpointConfig, ModelsConfig
from contextweave.core.registry import UnknownKeyError
from contextweave.core.types import JSON
from contextweave.models.decision import DecisionClient, ensure_decision_client
from contextweave.models.errors import ModelError
from contextweave.models.probe.context import ClassifyCandidate
from contextweave.models.providers import BuildContext, Device, Endpoints, ProviderRegistries
from contextweave.models.wire import parse_json_text

OPENAPI_PATH = "/openapi.json"
WIRE_TIMEOUT_S = 3.0
NO_ENDPOINT = DeviceEndpointConfig()


class ProbeUnavailableError(Exception):
    """A probe target cannot be built or reached."""


@dataclass(frozen=True, slots=True)
class ClassifyTarget:
    candidate: ClassifyCandidate
    device: Device
    endpoint: str


type ClassifyFactory = Callable[[ClassifyTarget], DecisionClient]


@dataclass(frozen=True, slots=True)
class WireInfo:
    source: str | None
    paths: tuple[str, ...]
    mentions_score: bool


NO_WIRE = WireInfo(None, (), False)


class WireInspector(Protocol):
    def inspect(self, endpoint: str) -> WireInfo: ...


class HttpWireInspector:
    """Reads the classify server's own OpenAPI document, from the endpoint given and nowhere else."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def inspect(self, endpoint: str) -> WireInfo:
        url = endpoint.rstrip("/") + OPENAPI_PATH
        try:
            with httpx.Client(transport=self._transport, timeout=WIRE_TIMEOUT_S) as client:
                response = client.get(url)
            if response.status_code >= 400:
                return NO_WIRE
            paths = _paths(parse_json_text(response.text))
        except httpx.HTTPError, ModelError:
            return NO_WIRE
        return WireInfo(url, paths, '"score"' in response.text.lower())


def _paths(document: JSON) -> tuple[str, ...]:
    if not isinstance(document, dict):
        return ()
    section = document.get("paths")
    return tuple(sorted(section)) if isinstance(section, dict) else ()


def registry_factory(
    registries: ProviderRegistries,
    models: ModelsConfig,
    transport: httpx.BaseTransport | None,
) -> ClassifyFactory:
    """Build a decision client per target through the classify provider registry."""

    def build(target: ClassifyTarget) -> DecisionClient:
        candidate = target.candidate
        endpoint = DeviceEndpointConfig(endpoint=target.endpoint)
        classify = replace(
            models.classify,
            provider=candidate.family,
            model=candidate.model,
            devices=(target.device.value,),
            gpu=endpoint if target.device is Device.GPU else NO_ENDPOINT,
            cpu=endpoint if target.device is Device.CPU else NO_ENDPOINT,
        )
        ctx = BuildContext(replace(models, classify=classify), Endpoints(), transport)
        try:
            factory = registries.classify.get(candidate.family)
        except UnknownKeyError as exc:
            raise ProbeUnavailableError(
                f"classify provider '{candidate.family}' is not registered (ships with T20)"
            ) from exc
        return ensure_decision_client(factory(ctx))

    return build
