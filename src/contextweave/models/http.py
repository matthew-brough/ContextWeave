from collections.abc import Mapping

import httpx

from contextweave.core.types import JSON
from contextweave.models.errors import (
    ModelContextError,
    ModelDeadlineError,
    ModelRequestError,
    ModelResponseError,
    ModelTransportError,
)
from contextweave.models.wire import as_object, to_json


class JsonClient:
    """Typed JSON-over-HTTP POST wrapper mapping failures to model errors."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url, transport=transport, headers=dict(headers or {})
        )

    def post(self, path: str, body: Mapping[str, JSON], *, timeout_s: float) -> dict[str, JSON]:
        """POST a JSON body and return the validated JSON object response."""
        try:
            response = self._client.post(path, json=body, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ModelDeadlineError(f"{path} exceeded {timeout_s:.3f}s") from exc
        except httpx.HTTPError as exc:
            raise ModelTransportError(f"{path}: {exc}") from exc
        return self._decode(path, response)

    def get(self, path: str, *, timeout_s: float) -> dict[str, JSON]:
        """GET a path and return the validated JSON object response."""
        try:
            response = self._client.get(path, timeout=timeout_s)
        except httpx.TimeoutException as exc:
            raise ModelDeadlineError(f"{path} exceeded {timeout_s:.3f}s") from exc
        except httpx.HTTPError as exc:
            raise ModelTransportError(f"{path}: {exc}") from exc
        return self._decode(path, response)

    def reachable(self, path: str, *, timeout_s: float) -> bool:
        """True when a GET answers at all."""
        try:
            self._client.get(path, timeout=timeout_s)
        except httpx.HTTPError:
            return False
        return True

    def _decode(self, path: str, response: httpx.Response) -> dict[str, JSON]:
        if 400 <= response.status_code < 500:
            raise _rejection(path, response)
        if response.status_code >= 500:
            raise ModelTransportError(f"{path} returned {response.status_code}")
        try:
            decoded: object = response.json()
        except ValueError as exc:
            raise ModelResponseError(f"{path}: invalid JSON body") from exc
        return as_object(to_json(decoded), path)


def _rejection(path: str, response: httpx.Response) -> ModelRequestError:
    """A 4xx as a request error; llama-server context overflows carry the real token counts."""
    message = f"{path} returned {response.status_code}"
    try:
        decoded: object = response.json()
    except ValueError:
        return ModelRequestError(message)
    error = to_json(decoded)
    detail = error.get("error") if isinstance(error, dict) else None
    if not isinstance(detail, dict):
        return ModelRequestError(message)
    needed = detail.get("n_prompt_tokens")
    available = detail.get("n_ctx")
    if isinstance(needed, int) and isinstance(available, int) and needed > available > 0:
        return ModelContextError(
            f"{message}: {needed} tokens exceed context {available}",
            needed=needed,
            available=available,
        )
    text = detail.get("message")
    return ModelRequestError(f"{message}: {text}" if isinstance(text, str) else message)
