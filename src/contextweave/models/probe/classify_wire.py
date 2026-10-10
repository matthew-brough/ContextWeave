from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import httpx

from contextweave.core.config import DeviceEndpointConfig, ModelsConfig
from contextweave.core.registry import UnknownKeyError
from contextweave.models.decision import DecisionClient, ensure_decision_client
from contextweave.models.probe.context import ClassifyCandidate
from contextweave.models.probe.served import MODELS_PATH, served_models
from contextweave.models.providers import BuildContext, Device, Endpoints, ProviderRegistries

SYSTEMONE = "systemone"
NONE = "none"
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
    served: tuple[str, ...]


NO_WIRE = WireInfo(None, ())


class WireInspector(Protocol):
    def inspect(self, endpoint: str) -> WireInfo: ...


class HttpWireInspector:
    """Reads the model ids the classify server reports, from the endpoint given and nowhere else."""

    def __init__(self, transport: httpx.BaseTransport | None = None) -> None:
        self._transport = transport

    def inspect(self, endpoint: str) -> WireInfo:
        served = served_models(endpoint, self._transport)
        if served is None:
            return NO_WIRE
        return WireInfo(endpoint.rstrip("/") + MODELS_PATH, served)


def registry_factory(
    registries: ProviderRegistries,
    models: ModelsConfig,
    transport: httpx.BaseTransport | None,
) -> ClassifyFactory:
    """Build a decision client per target through the configured classify provider; family is a label."""
    provider = SYSTEMONE if models.classify.provider == NONE else models.classify.provider

    def build(target: ClassifyTarget) -> DecisionClient:
        candidate = target.candidate
        endpoint = DeviceEndpointConfig(endpoint=target.endpoint)
        classify = replace(
            models.classify,
            provider=provider,
            model=candidate.model,
            devices=(target.device.value,),
            gpu=endpoint if target.device is Device.GPU else NO_ENDPOINT,
            cpu=endpoint if target.device is Device.CPU else NO_ENDPOINT,
        )
        ctx = BuildContext(replace(models, classify=classify), Endpoints(), transport)
        try:
            factory = registries.classify.get(provider)
        except UnknownKeyError as exc:
            raise ProbeUnavailableError(
                f"classify provider '{provider}' is not registered"
            ) from exc
        return ensure_decision_client(factory(ctx))

    return build
