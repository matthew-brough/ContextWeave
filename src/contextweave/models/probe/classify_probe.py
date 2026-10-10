import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from contextweave.core.config import ClassifyConfig
from contextweave.models.decision import Decision, DecisionClient, Question
from contextweave.models.probe.classify_fit import (
    Criterion,
    Sweep,
    criteria_facts,
    evaluate,
    max_window_within,
    measured_fit,
)
from contextweave.models.probe.classify_wire import (
    ClassifyFactory,
    ClassifyTarget,
    WireInfo,
    WireInspector,
)
from contextweave.models.probe.context import ClassifyCandidate, ProbeEnv
from contextweave.models.probe.report import PROBED, Line, Status
from contextweave.models.probe.served import serves
from contextweave.models.probe.timing import sample_ms, summarize, time_ms
from contextweave.models.probe.util import Failure, attempt, gpu_used_mib
from contextweave.models.providers import Device

BYTES_PER_TOKEN = 3.5
BATCH_GAIN_LIMIT = 0.8
PROB_TOLERANCE = 1e-3
CONFIG_RUNG = 0
WIRE_KEYS = ("wire_source", "served_models")


@dataclass(slots=True)
class ClassifyResult:
    lines: list[Line] = field(default_factory=list[Line])
    recommendations: list[str] = field(default_factory=list[str])


@dataclass(frozen=True, slots=True)
class DeviceOutcome:
    lines: tuple[Line, ...]
    fits: bool
    sweeps: tuple[Sweep, ...]
    client: DecisionClient | None
    criteria: tuple[Criterion, ...]


def default_candidates(config: ClassifyConfig) -> tuple[ClassifyCandidate, ...]:
    """The configured classify model as a single rung-0 candidate."""
    if config.provider == "none":
        return ()
    return (
        ClassifyCandidate(
            config.provider,
            CONFIG_RUNG,
            "config",
            config.model,
            config.gpu.endpoint,
            config.cpu.endpoint,
        ),
    )


def probe_classify(
    env: ProbeEnv,
    candidates: Sequence[ClassifyCandidate],
    factory: ClassifyFactory,
    wire: WireInspector,
) -> ClassifyResult:
    """Walk the fit ladder per family in rung order, stopping at the first rung that fits."""
    result = ClassifyResult()
    if not candidates:
        result.lines.append(Line("classify", "-", Status.SKIPPED, "provider is none"))
        return result
    families: dict[str, list[ClassifyCandidate]] = defaultdict(list)
    for candidate in candidates:
        families[candidate.family].append(candidate)
    for family, rungs in families.items():
        _walk_family(env, family, sorted(rungs, key=lambda item: item.rung), factory, wire, result)
    return result


def _walk_family(
    env: ProbeEnv,
    family: str,
    rungs: Sequence[ClassifyCandidate],
    factory: ClassifyFactory,
    wire: WireInspector,
    result: ClassifyResult,
) -> None:
    winner: ClassifyCandidate | None = None
    for candidate in rungs:
        if winner is not None:
            summary = f"not probed: rung {winner.rung} ({winner.build_id}) already meets fit"
            result.lines.append(Line(f"classify.{candidate.label}", "-", Status.SKIPPED, summary))
            continue
        fit_device = _probe_candidate(env, candidate, factory, wire, result)
        if fit_device is not None:
            winner = candidate
            result.recommendations.append(
                f"classify {family}: rung {candidate.rung} ({candidate.build_id}) meets the "
                f"probe-measurable fit criteria on {fit_device.value}; T20 eval must confirm "
                "quality and calibration"
            )
    if winner is None:
        reached = any(
            line.reached and line.status in PROBED and line.probe.startswith(f"classify.{family}.")
            for line in result.lines
        )
        result.recommendations.append(
            _exhausted(family)
            if reached
            else f"classify {family}: no server reachable on any rung; start one and rerun"
        )


def _exhausted(family: str) -> str:
    return (
        f"classify {family}: no probed rung met fit criteria; record measured gaps in §B "
        "and raise to the user (do not default to fallback scoring)"
    )


def _probe_candidate(
    env: ProbeEnv,
    candidate: ClassifyCandidate,
    factory: ClassifyFactory,
    wire: WireInspector,
    result: ClassifyResult,
) -> Device | None:
    fit_device: Device | None = None
    capabilities_done = False
    for device in env.options.devices:
        endpoint = candidate.endpoint(device)
        label = f"classify.{candidate.label}"
        if not endpoint:
            result.lines.append(Line(label, device.value, Status.SKIPPED, "no endpoint configured"))
            continue
        outcome = _probe_device(env, ClassifyTarget(candidate, device, endpoint), factory)
        result.lines.extend(outcome.lines)
        if outcome.client is None:
            continue
        if not capabilities_done:
            info = wire.inspect(endpoint)
            result.lines.append(_capability_line(env, candidate, device, outcome, info))
            capabilities_done = True
        fit_line = _fit_line(candidate, device, outcome)
        result.lines.append(fit_line)
        if outcome.fits and fit_device is None:
            fit_device = device
        if device is Device.CPU:
            best = max_window_within(outcome.sweeps, env.gate_budget_ms)
            result.recommendations.append(
                f"classify {candidate.label} cpu: largest gate.window within "
                f"{env.gate_budget_ms:.0f} ms = {best} (configured {env.gate.window})"
            )
    return fit_device


def build_states(env: ProbeEnv, window: int) -> list[str]:
    """Distinct synthetic gate states, each sized to gate.card_tokens."""
    sample = env.probe_set.state
    target = int(env.gate.card_tokens * BYTES_PER_TOKEN)
    card = sample.card
    while len(card) < target:
        card += "\n" + sample.card
    return [
        f"query: {sample.query}\ncandidate {index}:\n{card[:target]}" for index in range(window)
    ]


def _gate_questions(env: ProbeEnv) -> list[Question]:
    return [item.question() for item in env.probe_set.gate_questions]


def _windows(env: ProbeEnv) -> list[int]:
    limit = env.gate.window
    wanted = {window for window in env.options.windows if 1 <= window <= limit}
    wanted.add(limit)
    return sorted(wanted)


def _probe_device(env: ProbeEnv, target: ClassifyTarget, factory: ClassifyFactory) -> DeviceOutcome:
    label = f"classify.{target.candidate.label}"
    name = target.device.value
    client = attempt(lambda: factory(target))
    if isinstance(client, Failure):
        return _down(Line(label, name, Status.UNAVAILABLE, client.reason))
    questions = _gate_questions(env)
    cold_states = build_states(env, 1)
    cold = attempt(lambda: time_ms(lambda: client.decide(cold_states, questions), env.clock))
    if isinstance(cold, Failure):
        return _down(Line(label, name, Status.UNAVAILABLE, cold.reason))
    sweeps: list[Sweep] = []
    peaks: list[float] = []
    for window in _windows(env):
        states = build_states(env, window)
        rows: list[int] = []

        def call(states: list[str] = states, rows: list[int] = rows) -> None:
            rows.append(len(client.decide(states, questions)))

        samples = attempt(
            lambda call=call: sample_ms(call, env.clock, runs=env.options.classify_runs, warmup=1)
        )
        if isinstance(samples, Failure):
            return _down(
                Line(label, name, Status.UNAVAILABLE, f"window {window}: {samples.reason}")
            )
        sweeps.append(
            Sweep(window, summarize(samples).p50_ms, all(count == window for count in rows))
        )
        used = gpu_used_mib(env.host)
        if used is not None:
            peaks.append(used)
    peak = max(peaks, default=None)
    criteria = evaluate(env, target.device, sweeps, peak)
    metrics = {"cold_ms": cold, "states_tokens": float(env.gate.card_tokens)}
    for sweep in sweeps:
        metrics[f"gate_w{sweep.window}_p50_ms"] = sweep.p50_ms
        metrics[f"gate_w{sweep.window}_per_state_ms"] = sweep.p50_ms / sweep.window
    if peak is not None:
        metrics["gpu_peak_mib"] = peak
    curve = ", ".join(f"w{sweep.window}={sweep.p50_ms:.0f}ms" for sweep in sweeps)
    line = Line(label, name, Status.OK, f"gate p50 {curve}; cold {cold:.0f} ms", metrics)
    return DeviceOutcome((line,), measured_fit(criteria), tuple(sweeps), client, criteria)


def _down(line: Line) -> DeviceOutcome:
    return DeviceOutcome((line,), False, (), None, ())


def _fit_line(candidate: ClassifyCandidate, device: Device, outcome: DeviceOutcome) -> Line:
    verdict = "meets probe-measurable fit" if outcome.fits else "does not fit"
    return Line(
        f"classify.{candidate.label}.fit",
        device.value,
        Status.OK if outcome.fits else Status.WARN,
        verdict,
        {},
        criteria_facts(outcome.criteria),
    )


def _capability_line(
    env: ProbeEnv,
    candidate: ClassifyCandidate,
    device: Device,
    outcome: DeviceOutcome,
    info: WireInfo,
) -> Line:
    client = outcome.client
    facts: dict[str, str] = {
        "wire_source": info.source or "none",
        "served_models": ",".join(info.served) or "none",
    }
    if client is not None:
        facts["serves_model"] = _serves(candidate, info)
        facts["concurrent_states"] = _batching(outcome.sweeps)
        facts["dynamic_options"] = _dynamic(env, client)
    return Line(
        f"classify.{candidate.label}.protocol",
        device.value,
        Status.OK,
        ", ".join(f"{key}={value}" for key, value in facts.items() if key not in WIRE_KEYS),
        {},
        facts,
    )


def _decide_one(client: DecisionClient, state: str, question: Question) -> list[Decision] | Failure:
    rows = attempt(lambda: client.decide([state], [question]))
    if isinstance(rows, Failure):
        return rows
    return rows[0] if rows else Failure("empty reply")


def _serves(candidate: ClassifyCandidate, info: WireInfo) -> str:
    if info.source is None:
        return "unknown (no /v1/models)"
    return "yes" if serves(info.served, candidate.model) else "no (check --alias or the model)"


def _batching(sweeps: Sequence[Sweep]) -> str:
    single = next((sweep for sweep in sweeps if sweep.window == 1), None)
    batched = [sweep for sweep in sweeps if sweep.window > 1]
    if single is None or not batched:
        return "unknown (needs windows 1 and >1)"
    widest = batched[-1]
    if not widest.rows_ok:
        return f"no (window {widest.window} did not return one row per state)"
    ratio = widest.p50_ms / (widest.window * single.p50_ms) if single.p50_ms > 0 else math.inf
    verdict = "yes" if ratio < BATCH_GAIN_LIMIT else "no"
    return f"{verdict} (w{widest.window} costs {ratio:.2f} x {widest.window} singles)"


def _dynamic(env: ProbeEnv, client: DecisionClient) -> str:
    question = env.probe_set.dynamic_question.question()
    reply = _decide_one(client, build_states(env, 1)[0], question)
    if isinstance(reply, Failure):
        return f"error: {reply.reason}"
    decision = reply[0] if reply else None
    if decision is None:
        return "no (empty reply)"
    keys_ok = set(decision.probs) == set(question.options)
    total_ok = abs(sum(decision.probs.values()) - 1.0) <= PROB_TOLERANCE
    return "yes" if keys_ok and total_ok else "no (reply options differ from request)"
