import json
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from contextweave.core.types import JSON


@dataclass(slots=True)
class Recorder:
    handler: Callable[[httpx.Request], httpx.Response]
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def bodies(self) -> list[dict[str, JSON]]:
        return [json.loads(request.content) for request in self.requests]

    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


def recorder(handler: Callable[[httpx.Request], httpx.Response]) -> Recorder:
    return Recorder(handler)


def transport_of(rec: Recorder) -> httpx.MockTransport:
    return httpx.MockTransport(rec)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
