from collections.abc import Mapping, Sequence

import httpx

from contextweave.core.config import DeviceEndpointConfig
from contextweave.models.errors import ProviderSlotError
from contextweave.models.http import JsonClient
from contextweave.models.providers import Device

LOCAL_DEVICES = (Device.GPU, Device.CPU)


def device_table(
    gpu: DeviceEndpointConfig, cpu: DeviceEndpointConfig
) -> dict[Device, DeviceEndpointConfig]:
    """A slot's gpu and cpu server entries keyed by device."""
    return {Device.GPU: gpu, Device.CPU: cpu}


class DeviceHttp:
    """One server endpoint per device in configured chain order; calls go to the active one."""

    def __init__(
        self,
        name: str,
        devices: Sequence[str],
        table: Mapping[Device, DeviceEndpointConfig],
        transport: httpx.BaseTransport | None,
    ) -> None:
        self._name = name
        self._chain = tuple(
            device
            for wanted in devices
            for device in LOCAL_DEVICES
            if device.value == wanted and device in table and table[device].endpoint
        )
        self._servers = {device: table[device] for device in self._chain}
        self._http = {
            device: JsonClient(self._servers[device].endpoint, transport=transport)
            for device in self._chain
        }
        self._device = self._chain[0] if self._chain else Device.NONE

    @property
    def devices(self) -> tuple[Device, ...]:
        return self._chain

    @property
    def device(self) -> Device:
        return self._device

    @property
    def servers(self) -> Mapping[Device, DeviceEndpointConfig]:
        return self._servers

    def set_device(self, device: Device) -> None:
        if device not in self._http:
            raise ProviderSlotError(f"{self._name} has no device '{device}'")
        self._device = device

    def active(self) -> JsonClient:
        """The active device's client."""
        http = self._http.get(self._device)
        if http is None:
            raise ProviderSlotError(f"{self._name} has no active device")
        return http
