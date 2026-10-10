from collections.abc import Mapping

import httpx

from contextweave.core.config import DeviceEndpointConfig
from contextweave.core.types import JSON
from contextweave.models.errors import ModelError
from contextweave.models.http import JsonClient
from contextweave.models.probe.report import Line, Status, missing_summary
from contextweave.models.providers import Device, Supervised

MODELS_PATH = "/v1/models"
SERVED_TIMEOUT_S = 3.0


def served_models(endpoint: str, transport: httpx.BaseTransport | None) -> tuple[str, ...] | None:
    """Model ids a server reports at /v1/models; None when it does not answer."""
    try:
        reply = JsonClient(endpoint, transport=transport).get(
            MODELS_PATH, timeout_s=SERVED_TIMEOUT_S
        )
    except ModelError:
        return None
    return _ids(reply.get("data"))


def _ids(rows: JSON) -> tuple[str, ...]:
    if not isinstance(rows, list):
        return ()
    found: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            ident = row.get("id")
            if isinstance(ident, str):
                found.append(ident)
    return tuple(found)


def serves(ids: tuple[str, ...], model: str) -> bool:
    """True when a reported id names the configured model (llama-server reports the -hf ref or alias)."""
    wanted = model.lower()
    return any(wanted in ident.lower() or ident.lower() in wanted for ident in ids if ident)


def served_lines(
    probe: str,
    client: object,
    model: str,
    devices: tuple[Device, ...],
    transport: httpx.BaseTransport | None,
) -> list[Line]:
    """One line per supervised server: answering, and whether it serves the configured model."""
    if not isinstance(client, Supervised):
        return []
    servers: Mapping[Device, DeviceEndpointConfig] = client.servers
    lines: list[Line] = []
    for device in devices:
        server = servers.get(device)
        if server is None:
            continue
        ids = served_models(server.endpoint, transport)
        if ids is None:
            summary = f"no server at {server.endpoint}; {missing_summary(model)}"
            lines.append(Line(f"{probe}.served", device.value, Status.UNAVAILABLE, summary))
        elif serves(ids, model):
            lines.append(Line(f"{probe}.served", device.value, Status.OK, f"serving {model}"))
        else:
            reported = ", ".join(ids) or "nothing"
            summary = f"server reports {reported}, configured {model}; check --alias or the model"
            lines.append(Line(f"{probe}.served", device.value, Status.WARN, summary))
    return lines
