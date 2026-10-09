import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import httpx
import pytest
from models_support import Recorder, recorder, transport_of

from contextweave.core.config import (
    AnnotateConfig,
    ClassifyConfig,
    Config,
    EmbedConfig,
    ModelsConfig,
)
from contextweave.core.types import JSON
from contextweave.models.decision import Decision, Question
from contextweave.models.errors import ModelTransportError
from contextweave.models.probe.classify_wire import (
    ClassifyFactory,
    ClassifyTarget,
    ProbeUnavailableError,
    WireInfo,
)
from contextweave.models.probe.context import ClassifyCandidate, ProbeOptions
from contextweave.models.probe.host import GpuReading, parse_gpu_row
from contextweave.models.probe.render import render_json, render_record, render_table
from contextweave.models.probe.report import Line, Status, report_json
from contextweave.models.probe.runner import run_probe
from contextweave.models.probe.timing import rate_per_s, sample_ms, summarize
from contextweave.models.providers import Device, EmbedKind, Message, ProviderRegistries, Slot
from contextweave.models.slots import default_registries

EMBED_DIM = 4


class StepClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def spend_ms(self, ms: float) -> None:
        self.now += ms / 1000


@dataclass(slots=True)
class FakeHost:
    used_mib: float = 5000.0
    gpu_present: bool = True

    def cpu_name(self) -> str:
        return "Fake 5700X"

    def cores(self) -> int:
        return 16

    def gpu(self) -> GpuReading | None:
        return GpuReading("Fake GPU", 8192.0, self.used_mib) if self.gpu_present else None


@dataclass(slots=True)
class DeviceEmbed:
    clock: StepClock
    cost_ms: Mapping[Device, float]
    broken: frozenset[Device] = frozenset()
    device: Device = Device.GPU
    moves: list[Device] = field(default_factory=list[Device])

    @property
    def devices(self) -> tuple[Device, ...]:
        return (Device.GPU, Device.CPU)

    def set_device(self, device: Device) -> None:
        self.moves.append(device)
        self.device = device

    def embed(self, texts: Sequence[str], *, kind: EmbedKind) -> list[list[float]]:
        if self.device in self.broken:
            raise ModelTransportError(f"{self.device} down")
        self.clock.spend_ms(self.cost_ms[self.device])
        return [self._vector(text, kind) for text in texts]

    def _vector(self, text: str, kind: EmbedKind) -> list[float]:
        slot = sum(text.encode()) % EMBED_DIM
        vector = [0.0] * EMBED_DIM
        vector[slot] = 1.0
        return vector


@dataclass(slots=True)
class DeviceAnnotate:
    clock: StepClock
    cost_ms: Mapping[Device, float]
    device: Device = Device.GPU

    @property
    def devices(self) -> tuple[Device, ...]:
        return (Device.GPU, Device.CPU)

    def set_device(self, device: Device) -> None:
        self.device = device

    def annotate(
        self, prompt_messages: Sequence[Message], json_schema: Mapping[str, JSON]
    ) -> Mapping[str, JSON]:
        self.clock.spend_ms(self.cost_ms[self.device])
        return {"purpose": "does a thing", "tags": ["thing"]}


@dataclass(slots=True)
class FakeDecision:
    clock: StepClock
    fixed_ms: float
    per_state_ms: float
    calls: int = 0
    short_rows: bool = False
    echo_options: bool = True

    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        self.calls += 1
        self.clock.spend_ms(self.fixed_ms + self.per_state_ms * len(states))
        rows = [[self._decision(question) for question in questions] for _ in states]
        return rows[:-1] if self.short_rows and len(rows) > 1 else rows

    def _decision(self, question: Question) -> Decision:
        options = question.options if self.echo_options else ("other",)
        probs = {option: 1 / len(options) for option in options}
        return Decision(question.name, probs, options[0], probs[options[0]])


class FakeWire:
    def __init__(self, info: WireInfo) -> None:
        self.info = info

    def inspect(self, endpoint: str) -> WireInfo:
        return self.info


NO_DOC = FakeWire(WireInfo(None, (), False))


def config_with(
    embed_provider: str = "fake",
    annotate_provider: str = "fake",
    deadline_ms: int = 1500,
) -> Config:
    models = ModelsConfig(
        deadline_ms=deadline_ms,
        embed=EmbedConfig(provider=embed_provider, model="jina-fake", dim=EMBED_DIM),
        annotate=AnnotateConfig(provider=annotate_provider, model="qwen-fake"),
        classify=ClassifyConfig(provider="none"),
    )
    return replace(Config(), models=models)


def registries_with(embed: DeviceEmbed, annotate: DeviceAnnotate) -> ProviderRegistries:
    registries = default_registries()
    registries.embed.register("fake", lambda ctx: embed)
    registries.annotate.register("fake", lambda ctx: annotate)
    return registries


def probe_fakes(
    clock: StepClock,
    *,
    embed_cost: Mapping[Device, float] | None = None,
    annotate_cost: Mapping[Device, float] | None = None,
    broken: frozenset[Device] = frozenset(),
    slots: tuple[Slot, ...] = (Slot.EMBED, Slot.ANNOTATE),
    deadline_ms: int = 1500,
):
    embed = DeviceEmbed(clock, embed_cost or {Device.GPU: 10.0, Device.CPU: 80.0}, broken)
    annotate = DeviceAnnotate(clock, annotate_cost or {Device.GPU: 400.0, Device.CPU: 3000.0})
    options = ProbeOptions(slots=slots, runs=4, warmup=1, annotate_chunks=3)
    return run_probe(
        config_with(deadline_ms=deadline_ms),
        options,
        registries=registries_with(embed, annotate),
        host=FakeHost(),
        clock=clock,
    )


def by_probe(lines: Sequence[Line], probe: str) -> dict[str, Line]:
    return {line.device: line for line in lines if line.probe == probe}


def test_summarize_median_and_nearest_rank_p95() -> None:
    latency = summarize([10.0, 40.0, 20.0, 30.0])
    assert latency.p50_ms == 25.0
    assert latency.p95_ms == 40.0
    assert (latency.min_ms, latency.max_ms, latency.runs) == (10.0, 40.0, 4)
    many = summarize([float(value) for value in range(1, 101)])
    assert many.p95_ms == 95.0


def test_summarize_rejects_empty() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_sample_ms_discards_warmup_and_times_calls() -> None:
    clock = StepClock()
    costs = iter([500.0, 10.0, 20.0, 30.0])
    samples = sample_ms(lambda: clock.spend_ms(next(costs)), clock, runs=3, warmup=1)
    assert samples == pytest.approx([10.0, 20.0, 30.0])


def test_rate_per_s() -> None:
    assert rate_per_s(6, 2000.0) == 3.0
    assert rate_per_s(6, 0.0) == 0.0


def test_parse_gpu_row() -> None:
    reading = parse_gpu_row("NVIDIA RTX 3070, 8192, 1500")
    assert reading == GpuReading("NVIDIA RTX 3070", 8192.0, 1500.0)
    assert parse_gpu_row("garbage") is None


def test_embed_latency_per_device_and_query_device_recommendation() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,))
    query = by_probe(report.lines, "embed.query")
    assert query["gpu"].metrics["p50_ms"] == pytest.approx(10.0)
    assert query["cpu"].metrics["p50_ms"] == pytest.approx(80.0)
    assert query["gpu"].metrics["gpu_used_mib"] == 5000.0
    assert report.exit_code == 0
    assert any("query_device = 'cpu'" in item for item in report.recommendations)


def test_embed_cpu_over_budget_recommends_gpu() -> None:
    report = probe_fakes(
        StepClock(), slots=(Slot.EMBED,), embed_cost={Device.GPU: 10.0, Device.CPU: 900.0}
    )
    assert any("query_device = 'gpu'" in item for item in report.recommendations)


def test_embed_prefix_line_matches_card_by_default() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,))
    line = by_probe(report.lines, "embed.prefix_vs_card")["-"]
    assert line.status is Status.OK
    assert line.facts["query_prefix_matches_card"] == "true"


def test_embed_prefix_effect_reports_variants() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,))
    effect = next(iter(by_probe(report.lines, "embed.prefix_effect").values()))
    assert {"configured_top1", "swapped_top1", "memory_top1"} <= set(effect.metrics)


def test_device_failure_marks_only_that_line_unavailable() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,), broken=frozenset({Device.GPU}))
    query = by_probe(report.lines, "embed.query")
    assert query["gpu"].status is Status.UNAVAILABLE
    assert query["cpu"].status is Status.OK
    assert report.exit_code == 0


def test_everything_down_exits_nonzero() -> None:
    report = probe_fakes(
        StepClock(), slots=(Slot.EMBED,), broken=frozenset({Device.GPU, Device.CPU})
    )
    assert report.probed is False
    assert report.exit_code == 1


def test_annotate_rate_and_deadline_warning() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.ANNOTATE,))
    rate = by_probe(report.lines, "annotate.rate")
    assert rate["gpu"].status is Status.OK
    assert rate["gpu"].metrics["chunks_per_s"] == pytest.approx(2.5)
    assert rate["cpu"].status is Status.WARN
    assert rate["cpu"].metrics["over_deadline"] == 3.0
    assert any("T19 annotate on cpu" in item and "4000" in item for item in report.recommendations)


def ollama_transport(
    tags: list[str], embed_ok: bool = True
) -> tuple[Recorder, httpx.MockTransport]:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": name} for name in tags]})
        if request.url.path == "/api/embed" and embed_ok:
            return httpx.Response(200, json={"embeddings": [[1.0, 0.0, 0.0, 0.0]]})
        if request.url.path == "/api/chat":
            body = {"message": {"content": json.dumps({"purpose": "p", "tags": ["t"]})}}
            return httpx.Response(200, json=body)
        return httpx.Response(404)

    rec = recorder(handler)
    return rec, transport_of(rec)


FORBIDDEN = ("pull", "create", "delete", "push", "copy", "blobs")


def ollama_config() -> Config:
    models = ModelsConfig(
        embed=EmbedConfig(provider="ollama", model="jina-code-embeddings-1.5b", dim=EMBED_DIM),
        annotate=AnnotateConfig(provider="ollama", model="qwen3.5:4b"),
        classify=ClassifyConfig(provider="none"),
    )
    return replace(Config(), models=models)


def test_missing_models_report_pull_hint_and_make_no_model_calls() -> None:
    rec, transport = ollama_transport([])
    options = ProbeOptions(slots=(Slot.EMBED, Slot.ANNOTATE))
    report = run_probe(ollama_config(), options, host=FakeHost(), transport=transport)
    summaries = {line.probe: line.summary for line in report.lines if line.status is Status.MISSING}
    assert summaries["embed.tag"] == (
        "missing: jina-code-embeddings-1.5b (run `ollama pull jina-code-embeddings-1.5b` yourself)"
    )
    assert (
        summaries["annotate.tag"] == "missing: qwen3.5:4b (run `ollama pull qwen3.5:4b` yourself)"
    )
    assert rec.paths() == ["/api/tags"]
    assert report.exit_code == 1


def test_present_models_probe_without_pull_or_create_calls() -> None:
    rec, transport = ollama_transport(["jina-code-embeddings-1.5b:latest", "qwen3.5:4b"])
    options = ProbeOptions(slots=(Slot.EMBED, Slot.ANNOTATE), runs=2, warmup=0, annotate_chunks=1)
    report = run_probe(ollama_config(), options, host=FakeHost(), transport=transport)
    assert report.exit_code == 0
    assert by_probe(report.lines, "embed.tag")["-"].status is Status.OK
    paths = set(rec.paths())
    assert {"/api/tags", "/api/embed", "/api/chat"} <= paths
    assert not any(word in path for path in paths for word in FORBIDDEN)
    assert all(request.method in ("GET", "POST") for request in rec.requests)
    assert all(
        request.method == "GET" or request.url.path in ("/api/embed", "/api/chat")
        for request in rec.requests
    )


def test_ollama_unreachable_degrades_to_unavailable() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = transport_of(recorder(refuse))
    options = ProbeOptions(slots=(Slot.EMBED, Slot.ANNOTATE))
    report = run_probe(ollama_config(), options, host=FakeHost(), transport=transport)
    assert by_probe(report.lines, "embed.tag")["-"].status is Status.UNAVAILABLE
    assert by_probe(report.lines, "annotate.tag")["-"].status is Status.UNAVAILABLE
    assert report.exit_code == 1


def candidate(rung: int, build: str = "b", family: str = "kev") -> ClassifyCandidate:
    return ClassifyCandidate(family, rung, build, "model", "http://gpu", "http://cpu")


def classify_report(
    clock: StepClock,
    clients: Mapping[tuple[int, Device], FakeDecision],
    candidates: Sequence[ClassifyCandidate],
    *,
    devices: tuple[Device, ...] = (Device.GPU,),
    wire: FakeWire = NO_DOC,
    host: FakeHost | None = None,
):
    def factory(target: ClassifyTarget) -> FakeDecision:
        found = clients.get((target.candidate.rung, target.device))
        if found is None:
            raise ProbeUnavailableError("no server")
        return found

    typed: ClassifyFactory = factory
    options = ProbeOptions(
        slots=(Slot.CLASSIFY,), devices=devices, classify_runs=2, candidates=tuple(candidates)
    )
    return run_probe(
        config_with(),
        options,
        host=host or FakeHost(),
        clock=clock,
        classify_factory=typed,
        wire=wire,
    )


def test_ladder_stops_at_first_fitting_rung() -> None:
    clock = StepClock()
    clients = {
        (1, Device.GPU): FakeDecision(clock, 0.0, 50.0),
        (3, Device.GPU): FakeDecision(clock, 0.0, 10.0),
        (5, Device.GPU): FakeDecision(clock, 0.0, 10.0),
    }
    report = classify_report(clock, clients, [candidate(5), candidate(1), candidate(3)])
    labels = {line.probe: line for line in report.lines}
    assert labels["classify.kev.r1.b.fit"].status is Status.WARN
    assert labels["classify.kev.r3.b.fit"].status is Status.OK
    assert labels["classify.kev.r5.b"].status is Status.SKIPPED
    assert clients[(5, Device.GPU)].calls == 0
    assert any("rung 3" in item and "meets" in item for item in report.recommendations)
    latency = labels["classify.kev.r1.b.fit"].facts["criterion.gate_latency"]
    assert latency.startswith("fail")


def test_ladder_exhausted_raises_to_user() -> None:
    clock = StepClock()
    clients = {
        (1, Device.GPU): FakeDecision(clock, 0.0, 50.0),
        (2, Device.GPU): FakeDecision(clock, 0.0, 60.0),
    }
    report = classify_report(clock, clients, [candidate(1), candidate(2)])
    assert not any(
        line.status is Status.SKIPPED for line in report.lines if line.probe.startswith("classify")
    )
    assert any("no probed rung met fit criteria" in item for item in report.recommendations)


def test_vram_over_limit_fails_fit_even_when_fast() -> None:
    clock = StepClock()
    clients = {(1, Device.GPU): FakeDecision(clock, 0.0, 1.0)}
    report = classify_report(clock, clients, [candidate(1)], host=FakeHost(used_mib=9000.0))
    fit = by_probe(report.lines, "classify.kev.r1.b.fit")["gpu"]
    assert fit.status is Status.WARN
    assert fit.facts["criterion.vram"].startswith("fail")


def test_cpu_device_needs_window_ten_within_budget() -> None:
    clock = StepClock()
    clients = {(6, Device.CPU): FakeDecision(clock, 0.0, 90.0)}
    report = classify_report(clock, clients, [candidate(6)], devices=(Device.CPU,))
    fit = by_probe(report.lines, "classify.kev.r6.b.fit")["cpu"]
    assert fit.status is Status.OK
    gate = by_probe(report.lines, "classify.kev.r6.b")["cpu"]
    assert gate.metrics["gate_w10_p50_ms"] == pytest.approx(900.0)
    assert gate.metrics["gate_w30_p50_ms"] == pytest.approx(2700.0)
    assert any("largest gate.window within 1000 ms = 10" in item for item in report.recommendations)


def test_unreachable_candidate_is_unavailable_and_walk_continues() -> None:
    clock = StepClock()
    clients = {(2, Device.GPU): FakeDecision(clock, 0.0, 5.0)}
    report = classify_report(clock, clients, [candidate(1), candidate(2)])
    assert by_probe(report.lines, "classify.kev.r1.b")["gpu"].status is Status.UNAVAILABLE
    assert by_probe(report.lines, "classify.kev.r2.b.fit")["gpu"].status is Status.OK


def test_protocol_capabilities_batching_dynamic_and_score() -> None:
    clock = StepClock()
    clients = {(1, Device.GPU): FakeDecision(clock, 200.0, 5.0)}
    wire = FakeWire(WireInfo("http://gpu/openapi.json", ("/v1/decide",), True))
    report = classify_report(clock, clients, [candidate(1)], wire=wire)
    facts = by_probe(report.lines, "classify.kev.r1.b.protocol")["gpu"].facts
    assert facts["multi_state_batching"].startswith("yes")
    assert facts["dynamic_options"] == "yes"
    assert facts["score_native"].startswith("yes")
    assert facts["wire_paths"] == "/v1/decide"


def test_protocol_capabilities_negative_cases() -> None:
    clock = StepClock()
    client = FakeDecision(clock, 0.0, 100.0, short_rows=True, echo_options=False)
    report = classify_report(clock, {(1, Device.GPU): client}, [candidate(1)])
    facts = by_probe(report.lines, "classify.kev.r1.b.protocol")["gpu"].facts
    assert facts["multi_state_batching"].startswith("no")
    assert facts["dynamic_options"].startswith("no")
    assert facts["score_native"].startswith("yes (reply keys differ")


def test_classify_none_provider_is_skipped() -> None:
    report = run_probe(
        config_with("none", "none"),
        ProbeOptions(slots=(Slot.CLASSIFY,)),
        host=FakeHost(),
        clock=StepClock(),
    )
    assert [line.status for line in report.lines] == [Status.SKIPPED]
    assert report.exit_code == 1


def test_json_record_shape() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,))
    record = json.loads(render_json(report))
    assert set(record) == {"probed", "host", "lines", "recommendations"}
    assert record["probed"] is True
    assert record["host"]["cpu"] == "Fake 5700X"
    first = record["lines"][0]
    assert set(first) == {"probe", "device", "status", "summary", "metrics", "facts"}
    assert report_json(report)["probed"] is True


def test_record_markdown_for_section_b() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,))
    text = render_record(report, "2026-10-09")
    assert text.startswith("### T10 model probe 2026-10-09")
    assert "| probe | device | status | result |" in text
    assert "| embed.query | gpu | ok |" in text
    assert "decides:" in text
    assert "measurements:" in text


def test_table_lists_every_line() -> None:
    report = probe_fakes(StepClock(), slots=(Slot.EMBED,))
    table = render_table(report)
    assert table.splitlines()[0].startswith("host: cpu Fake 5700X x16")
    assert sum(1 for row in table.splitlines() if row.startswith("embed.query")) == 2
