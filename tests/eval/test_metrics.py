from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contextweave.eval.adapters import Served
from contextweave.eval.baselines import (
    Baseline,
    BaselineError,
    baseline_from,
    check_regression,
    read_baseline,
    write_baseline,
)
from contextweave.eval.metrics import Metrics, aggregate, percentile, score_case
from contextweave.eval.suite import Case


def test_score_case() -> None:
    case = Case("q", ("a", "b"), ("x",))
    result = score_case(
        case, Served(("a", "x", "z"), tokens=30, latency_ms=5, verdict_covered=True)
    )
    assert result.recall == 0.5
    assert result.precision == pytest.approx(1 / 3)
    assert result.must_not_hits == 1
    assert result.verdict_correct is False


def test_score_empty_and_degraded() -> None:
    result = score_case(Case("q", ("a",)), Served(error="boom"))
    assert (result.recall, result.precision, result.verdict_correct) == (0.0, 0.0, None)
    assert result.error == "boom"


def test_aggregate() -> None:
    cases = [Case("q1", ("a",)), Case("q2", ("b",))]
    results = [
        score_case(cases[0], Served(("a",), 10, 1.0, 2, True)),
        score_case(cases[1], Served((), 30, 9.0, 1, True, "err")),
    ]
    metrics = aggregate(results)
    assert metrics.recall == 0.5
    assert metrics.tokens == 20
    assert metrics.verdict_accuracy == 0.5
    assert metrics.model_calls == 3
    assert metrics.errors == 1
    assert (metrics.latency_p50_ms, metrics.latency_p95_ms) == (1.0, 9.0)


def test_aggregate_without_verdicts() -> None:
    metrics = aggregate([score_case(Case("q", ("a",)), Served(("a",)))])
    assert metrics.verdict_accuracy is None


@given(st.lists(st.floats(0, 1e6), min_size=1), st.floats(0.01, 1.0))
def test_percentile_is_a_member(values: list[float], fraction: float) -> None:
    assert percentile(values, fraction) in values


def test_percentile_empty() -> None:
    assert percentile([], 0.5) == 0.0


def _metrics(recall: float, tokens: float) -> Metrics:
    return Metrics(1, recall, 0.5, tokens, None, 0.0, 0.0, 0, 0, 0)


def test_regression_gate() -> None:
    base = Baseline("s", "nomodel", 0.80, 0.5, 1000)
    assert check_regression(base, _metrics(0.78, 1100)) == ()
    assert len(check_regression(base, _metrics(0.77, 1000))) == 1
    assert len(check_regression(base, _metrics(0.80, 1101))) == 1
    assert len(check_regression(base, _metrics(0.5, 2000))) == 2
    assert check_regression(base, _metrics(0.95, 10)) == ()


def test_baseline_roundtrip(tmp_path: Path) -> None:
    baseline = baseline_from("s", "full", _metrics(0.9, 123.0))
    path = write_baseline(baseline, tmp_path)
    assert path.name == "s.full.json"
    assert read_baseline("s", "full", tmp_path) == baseline


def test_baseline_errors(tmp_path: Path) -> None:
    with pytest.raises(BaselineError):
        read_baseline("s", "nomodel", tmp_path)
    (tmp_path / "s.nomodel.json").write_text('{"recall": "x"}')
    with pytest.raises(BaselineError):
        read_baseline("s", "nomodel", tmp_path)
