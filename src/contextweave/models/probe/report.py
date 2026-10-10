from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from contextweave.core.types import JSON


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


PROBED = (Status.OK, Status.WARN)


@dataclass(frozen=True, slots=True)
class Line:
    probe: str
    device: str
    status: Status
    summary: str
    metrics: Mapping[str, float] = field(default_factory=dict[str, float])
    facts: Mapping[str, str] = field(default_factory=dict[str, str])
    reached: bool = True


@dataclass(frozen=True, slots=True)
class Host:
    cpu: str
    cores: int
    gpu: str
    gpu_total_mib: float | None


@dataclass(frozen=True, slots=True)
class ProbeReport:
    host: Host
    lines: tuple[Line, ...]
    recommendations: tuple[str, ...]

    @property
    def probed(self) -> bool:
        return any(line.reached and line.status in PROBED for line in self.lines)

    @property
    def exit_code(self) -> int:
        return 0 if self.probed else 1


def missing_summary(model: str) -> str:
    return f"start llama-server for {model} per docs/runbooks/model-probe.md yourself"


def line_json(line: Line) -> dict[str, JSON]:
    metrics: dict[str, JSON] = {key: value for key, value in line.metrics.items()}
    facts: dict[str, JSON] = {key: value for key, value in line.facts.items()}
    return {
        "probe": line.probe,
        "device": line.device,
        "status": line.status.value,
        "summary": line.summary,
        "metrics": metrics,
        "facts": facts,
    }


def report_json(report: ProbeReport) -> dict[str, JSON]:
    """Machine record of a probe run."""
    host: dict[str, JSON] = {
        "cpu": report.host.cpu,
        "cores": report.host.cores,
        "gpu": report.host.gpu,
        "gpu_total_mib": report.host.gpu_total_mib,
    }
    lines: list[JSON] = [line_json(line) for line in report.lines]
    recommendations: list[JSON] = list(report.recommendations)
    return {
        "probed": report.probed,
        "host": host,
        "lines": lines,
        "recommendations": recommendations,
    }
