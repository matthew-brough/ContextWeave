import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from contextweave.core.registry import Registry

type Estimator = Callable[[str], int]

DEFAULT_MAX_TOKENS = 4000
SEARCH_MAX_TOKENS = 6000
DEFAULT_ESTIMATOR = "bytes35"

estimators: Registry[str, Estimator] = Registry()


def bytes35(text: str) -> int:
    """Estimate tokens as ceil(utf8 bytes / 3.5)."""
    return math.ceil(len(text.encode()) / 3.5)


estimators.register(DEFAULT_ESTIMATOR, bytes35)


def serialize_structured(structured: Mapping[str, object]) -> str:
    """Serialise a structured payload compactly with sorted keys."""
    return json.dumps(structured, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def measure_payload(
    structured: Mapping[str, object], text: str, estimator: Estimator = bytes35
) -> int:
    """Count tokens of structuredContent plus the text fallback."""
    return estimator(serialize_structured(structured)) + estimator(text)


@dataclass(frozen=True, slots=True)
class Ceiling:
    """Token ceiling for one tool response."""

    limit: int

    def fits(self, used: int) -> bool:
        return used <= self.limit

    def over_by(self, used: int) -> int:
        return max(0, used - self.limit)
