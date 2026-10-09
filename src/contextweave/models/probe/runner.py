import time
from collections.abc import Mapping
from dataclasses import replace

import httpx

from contextweave.core.config import Config
from contextweave.models.probe.annotate_probe import probe_annotate
from contextweave.models.probe.classify_probe import default_candidates, probe_classify
from contextweave.models.probe.classify_wire import (
    ClassifyFactory,
    HttpWireInspector,
    WireInspector,
    registry_factory,
)
from contextweave.models.probe.context import (
    DEFAULT_OLLAMA_URL,
    DEFAULT_PROBE_DEADLINE_MS,
    EMBED_CPU_QUERY_BUDGET_MS,
    ProbeEnv,
    ProbeOptions,
)
from contextweave.models.probe.embed_probe import probe_embed
from contextweave.models.probe.fixtures import load_probe_set
from contextweave.models.probe.host import HostInspector, SystemHost
from contextweave.models.probe.ollama_tags import fetch_tags
from contextweave.models.probe.report import PROBED, Host, Line, ProbeReport, Status
from contextweave.models.probe.timing import Clock
from contextweave.models.providers import Endpoints, ProviderRegistries, Slot
from contextweave.models.slots import ModelSlots, default_registries

OLLAMA = "ollama"


def run_probe(
    config: Config,
    options: ProbeOptions,
    *,
    registries: ProviderRegistries | None = None,
    host: HostInspector | None = None,
    clock: Clock = time.perf_counter,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    transport: httpx.BaseTransport | None = None,
    classify_factory: ClassifyFactory | None = None,
    wire: WireInspector | None = None,
    probe_deadline_ms: int = DEFAULT_PROBE_DEADLINE_MS,
) -> ProbeReport:
    """Probe the configured model slots on this host; never downloads, never raises."""
    available = registries or default_registries()
    inspector = host or SystemHost()
    models = replace(config.models, deadline_ms=max(probe_deadline_ms, config.models.deadline_ms))
    slots = ModelSlots.from_config(
        models,
        registries=available,
        endpoints=Endpoints(ollama=ollama_url),
        transport=transport,
        submit=lambda job: job(),
    )
    env = ProbeEnv(
        models=models,
        configured_deadline_ms=config.models.deadline_ms,
        gate=config.gate,
        options=options,
        slots=slots,
        probe_set=load_probe_set(),
        host=inspector,
        clock=clock,
        ollama_url=ollama_url,
        transport=transport,
    )
    lines = _issue_lines(slots)
    recommendations: list[str] = []
    catalog = (
        fetch_tags(ollama_url, transport)
        if _needs_tags(models.embed.provider, models.annotate.provider, options)
        else None
    )
    if Slot.EMBED in options.slots:
        embed_lines = probe_embed(env, catalog)
        lines.extend(embed_lines)
        recommendations.extend(_embed_recommendations(embed_lines, config))
    if Slot.ANNOTATE in options.slots:
        annotate_lines = probe_annotate(env, catalog)
        lines.extend(annotate_lines)
        recommendations.extend(_annotate_recommendations(annotate_lines, config))
    if Slot.CLASSIFY in options.slots:
        factory = classify_factory or registry_factory(available, models, transport)
        candidates = options.candidates or default_candidates(config.models.classify)
        classify = probe_classify(env, candidates, factory, wire or HttpWireInspector(transport))
        lines.extend(classify.lines)
        recommendations.extend(classify.recommendations)
    return ProbeReport(_host(inspector), tuple(lines), tuple(recommendations))


def _needs_tags(embed: str, annotate: str, options: ProbeOptions) -> bool:
    return (Slot.EMBED in options.slots and embed == OLLAMA) or (
        Slot.ANNOTATE in options.slots and annotate == OLLAMA
    )


def _host(inspector: HostInspector) -> Host:
    gpu = inspector.gpu()
    return Host(
        inspector.cpu_name(),
        inspector.cores(),
        gpu.name if gpu else "none detected",
        gpu.total_mib if gpu else None,
    )


def _issue_lines(slots: ModelSlots) -> list[Line]:
    return [Line("config", "-", Status.SKIPPED, issue) for issue in slots.issues]


def _find(lines: list[Line], probe: str) -> Mapping[str, Line]:
    return {line.device: line for line in lines if line.probe == probe and line.status in PROBED}


def _embed_recommendations(lines: list[Line], config: Config) -> list[str]:
    queries = _find(lines, "embed.query")
    cpu = queries.get("cpu")
    if cpu is None:
        return []
    p50 = cpu.metrics["p50_ms"]
    device = "cpu" if p50 <= EMBED_CPU_QUERY_BUDGET_MS else "gpu"
    return [
        f"T12 models.embed.query_device = {device!r}: cpu query p50 {p50:.0f} ms "
        f"(cpu is kept while p50 <= {EMBED_CPU_QUERY_BUDGET_MS:.0f} ms; configured "
        f"{config.models.embed.query_device!r})"
    ]


def _annotate_recommendations(lines: list[Line], config: Config) -> list[str]:
    rates = _find(lines, "annotate.rate")
    deadline = config.models.deadline_ms
    notes: list[str] = []
    for device, line in rates.items():
        p95 = line.metrics["p95_ms"]
        if p95 > deadline:
            notes.append(
                f"T19 annotate on {device}: p95 {p95:.0f} ms exceeds deadline_ms {deadline}; "
                f"needs an annotate-specific deadline of at least {_round_up(p95 * 1.25)} ms"
            )
        else:
            notes.append(f"T19 annotate on {device}: p95 {p95:.0f} ms fits deadline_ms {deadline}")
    return notes


def _round_up(value: float, step: int = 500) -> int:
    return int(-(-value // step) * step)
