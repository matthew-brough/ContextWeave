from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from contextweave.core.registry import Registry
from contextweave.eval.adapters import AdapterContext, AdapterFactory, ScoreSample
from contextweave.eval.metrics import CaseResult, Metrics, aggregate, score_case
from contextweave.eval.suite import Suite
from contextweave.models.calibration import (
    ECE_TARGET,
    CalibrationSample,
    QuestionCalibration,
    report_question,
)


@dataclass(frozen=True, slots=True)
class RunReport:
    """One adapter's run over a suite."""

    adapter: str
    ablate: str | None
    metrics: Metrics
    results: tuple[CaseResult, ...]
    variant: str | None = None
    calibration: tuple[QuestionCalibration, ...] = ()

    @property
    def label(self) -> str:
        base = self.adapter if self.variant is None else f"{self.adapter}[{self.variant}]"
        return base if self.ablate is None else f"{base}-{self.ablate}"


def calibration_of(samples: Sequence[ScoreSample]) -> tuple[QuestionCalibration, ...]:
    """ECE and fitted temperature per question over the pooled samples."""
    grouped: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.question, []).append(CalibrationSample(sample.prob, sample.label))
    return tuple(report_question(question, grouped[question]) for question in sorted(grouped))


def run_suite(
    suite: Suite,
    root: Path,
    adapter: str,
    adapters: Registry[str, AdapterFactory],
    profile: str = "nomodel",
    ablate: str | None = None,
    env: Mapping[str, str] | None = None,
    variant: str | None = None,
) -> RunReport:
    """Serve every case through the named adapter and score it."""
    expected = {case.query: case.expect for case in suite.cases}
    context = AdapterContext(root, profile, ablate, env or {}, expected)
    served = adapters.get(adapter)(context).serve_all([case.query for case in suite.cases])
    results = tuple(score_case(case, item) for case, item in zip(suite.cases, served, strict=True))
    samples = [sample for item in served for sample in item.samples]
    return RunReport(adapter, ablate, aggregate(results), results, variant, calibration_of(samples))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _opt_pct(value: float | None) -> str:
    return "-" if value is None else _pct(value)


def render_table(reports: Sequence[RunReport]) -> str:
    """One row per adapter run."""
    header = (
        "adapter",
        "recall",
        "precision",
        "tokens/case",
        "verdict",
        "p50 ms",
        "p95 ms",
        "model calls",
        "must_not",
        "errors",
        "skips",
    )
    rows: list[tuple[str, ...]] = [header]
    for report in reports:
        m = report.metrics
        label = report.label
        rows.append(
            (
                label,
                _pct(m.recall),
                _pct(m.precision),
                f"{m.tokens:.0f}",
                _opt_pct(m.verdict_accuracy),
                f"{m.latency_p50_ms:.0f}",
                f"{m.latency_p95_ms:.0f}",
                str(m.model_calls),
                str(m.must_not_hits),
                str(m.errors),
                str(m.skips),
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows
    )


def render_delta(base: Metrics, ablated: Metrics, stage: str) -> str:
    """Delta table of an ablated run against the full run."""
    lines = [f"ablation delta: {stage} (ablated - baseline)"]
    lines.append(f"recall      {(ablated.recall - base.recall) * 100:+.1f} pts")
    lines.append(f"precision   {(ablated.precision - base.precision) * 100:+.1f} pts")
    lines.append(f"tokens/case {ablated.tokens - base.tokens:+.0f}")
    lines.append(f"p50 ms      {ablated.latency_p50_ms - base.latency_p50_ms:+.0f}")
    lines.append(f"model calls {ablated.model_calls - base.model_calls:+d}")
    return "\n".join(lines)


def render_calibration(reports: Sequence[RunReport]) -> str:
    """Per-question ECE before and after temperature scaling for every report that has samples."""
    lines: list[str] = []
    for report in reports:
        if not report.calibration:
            continue
        lines.append(f"calibration: {report.label} (target ECE <= {ECE_TARGET})")
        for row in report.calibration:
            verdict = "ok" if row.meets_target else "MISS"
            lines.append(
                f"  {row.question:<18} n={row.samples:<5} ece={row.ece:.3f} "
                f"T={row.temperature:.2f} ece_after={row.ece_after:.3f} {verdict}"
            )
    return "\n".join(lines)
