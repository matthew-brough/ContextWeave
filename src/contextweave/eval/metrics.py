import math
from collections.abc import Sequence
from dataclasses import dataclass

from contextweave.eval.adapters import Served
from contextweave.eval.suite import Case


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Scores of one case."""

    query: str
    recall: float
    precision: float
    tokens: int
    latency_ms: float
    model_calls: int
    must_not_hits: int
    verdict_correct: bool | None
    error: str | None
    skips: int = 0


@dataclass(frozen=True, slots=True)
class Metrics:
    """Suite-level aggregates; recall and precision are means over cases."""

    cases: int
    recall: float
    precision: float
    tokens: float
    verdict_accuracy: float | None
    latency_p50_ms: float
    latency_p95_ms: float
    model_calls: int
    must_not_hits: int
    errors: int
    skips: int = 0


def score_case(case: Case, served: Served) -> CaseResult:
    """Score one served result against the case."""
    items = set(served.items)
    expected = set(case.expect)
    hits = len(items & expected)
    verdict_correct = (
        None
        if served.verdict_covered is None
        else served.verdict_covered == (hits == len(expected))
    )
    return CaseResult(
        query=case.query,
        recall=hits / len(expected),
        precision=hits / len(items) if items else 0.0,
        tokens=served.tokens,
        latency_ms=served.latency_ms,
        model_calls=served.model_calls,
        must_not_hits=len(items & set(case.must_not)),
        verdict_correct=verdict_correct,
        error=served.error,
        skips=served.skips,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile; zero for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(results: Sequence[CaseResult]) -> Metrics:
    """Roll case results up into suite metrics."""
    verdicts = [result.verdict_correct for result in results if result.verdict_correct is not None]
    latencies = [result.latency_ms for result in results]
    return Metrics(
        cases=len(results),
        recall=_mean([result.recall for result in results]),
        precision=_mean([result.precision for result in results]),
        tokens=_mean([result.tokens for result in results]),
        verdict_accuracy=sum(verdicts) / len(verdicts) if verdicts else None,
        latency_p50_ms=percentile(latencies, 0.5),
        latency_p95_ms=percentile(latencies, 0.95),
        model_calls=sum(result.model_calls for result in results),
        must_not_hits=sum(result.must_not_hits for result in results),
        errors=sum(1 for result in results if result.error is not None),
        skips=sum(result.skips for result in results),
    )
