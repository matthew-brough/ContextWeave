from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from contextweave.models.probe.context import MIN_CPU_WINDOW, ProbeEnv
from contextweave.models.providers import Device


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    PENDING = "pending"


@dataclass(frozen=True, slots=True)
class Criterion:
    name: str
    verdict: Verdict
    detail: str


@dataclass(frozen=True, slots=True)
class Sweep:
    window: int
    p50_ms: float
    rows_ok: bool


def max_window_within(sweeps: Sequence[Sweep], budget_ms: float) -> int:
    """Largest swept window whose p50 fits the budget, 0 when none does."""
    fitting = [sweep.window for sweep in sweeps if sweep.rows_ok and sweep.p50_ms <= budget_ms]
    return max(fitting, default=0)


def evaluate(
    env: ProbeEnv, device: Device, sweeps: Sequence[Sweep], peak_mib: float | None
) -> tuple[Criterion, ...]:
    """Probe-measurable fit criteria; quality and calibration stay pending for the T20 eval."""
    return (
        _latency(env, device, sweeps),
        _memory(env, device, peak_mib),
        Criterion("pipeline_p50", Verdict.PENDING, "full warm pipeline p50 <= 1.0 s: T20 eval"),
        Criterion(
            "quality", Verdict.PENDING, "top agreement >= 95%, recall drop <= 2 pts: T20 eval"
        ),
        Criterion(
            "calibration", Verdict.PENDING, "ECE <= 0.05 per binary/score question: T20 eval"
        ),
    )


def _latency(env: ProbeEnv, device: Device, sweeps: Sequence[Sweep]) -> Criterion:
    budget = env.gate_budget_ms
    required = env.gate.window if device is Device.GPU else MIN_CPU_WINDOW
    best = max_window_within(sweeps, budget)
    detail = f"largest window within {budget:.0f} ms = {best}, need >= {required}"
    return Criterion("gate_latency", Verdict.PASS if best >= required else Verdict.FAIL, detail)


def _memory(env: ProbeEnv, device: Device, peak_mib: float | None) -> Criterion:
    if device is not Device.GPU:
        return Criterion("vram", Verdict.PASS, "cpu device holds no vram")
    if peak_mib is None:
        return Criterion("vram", Verdict.PENDING, "nvidia-smi unavailable, measure by hand")
    limit = float(env.options.vram_mib)
    detail = f"peak {peak_mib:.0f} MiB used on gpu, limit {limit:.0f} MiB"
    return Criterion("vram", Verdict.PASS if peak_mib <= limit else Verdict.FAIL, detail)


def measured_fit(criteria: Sequence[Criterion]) -> bool:
    """True when no measurable criterion fails and gate latency passed."""
    latency = next((item for item in criteria if item.name == "gate_latency"), None)
    return (
        latency is not None
        and latency.verdict is Verdict.PASS
        and all(item.verdict is not Verdict.FAIL for item in criteria)
    )


def criteria_facts(criteria: Sequence[Criterion]) -> Mapping[str, str]:
    return {f"criterion.{item.name}": f"{item.verdict.value}: {item.detail}" for item in criteria}
