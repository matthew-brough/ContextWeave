import dataclasses
from collections.abc import Callable, Sequence
from typing import cast

import httpx
import pytest
from models_support import Clock, Recorder, recorder, transport_of

from contextweave.core.config import ClassifyConfig, DeviceEndpointConfig, ModelsConfig
from contextweave.models.decision import (
    Decision,
    DecisionClient,
    Question,
    QuestionKind,
)
from contextweave.models.errors import ModelDeadlineError
from contextweave.models.health import SlotHealth
from contextweave.models.providers import Device, EmbedKind, ProviderRegistries, Slot
from contextweave.models.residency import ClassifyBackend, Probe, Profile
from contextweave.models.slots import Done, ModelSlots, Skipped, default_registries

VECTOR = {"embeddings": [[1.0, 0.0, 0.0]]}


def make(
    models: ModelsConfig,
    clock: Clock,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    registries: ProviderRegistries | None = None,
    probe: Probe = lambda endpoint: True,
) -> tuple[ModelSlots, Recorder]:
    rec = recorder(handler)
    slots = ModelSlots.from_config(
        models,
        registries=registries,
        transport=transport_of(rec),
        clock=clock,
        submit=lambda job: job(),
        launcher=FakeLauncher(),
        probe=probe,
    )
    return slots, rec


class FakeHandle:
    def __init__(self, command: str) -> None:
        self.command = command
        self.running = True

    def alive(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False


class FakeLauncher:
    def __init__(self) -> None:
        self.started: list[FakeHandle] = []

    def start(self, command: str) -> FakeHandle:
        handle = FakeHandle(command)
        self.started.append(handle)
        return handle


def good(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=VECTOR)


def timeout_on_gpu(request: httpx.Request) -> httpx.Response:
    on_gpu = b'"num_gpu":0' not in request.content
    if request.url.path == "/api/embed" and on_gpu and b'"input":""' not in request.content:
        raise httpx.ReadTimeout("slow", request=request)
    return httpx.Response(200, json=VECTOR)


def embed(slots: ModelSlots):
    return slots.embed(["x"], kind=EmbedKind.QUERY, stage="vector")


def test_success_returns_done(models: ModelsConfig, clock: Clock) -> None:
    slots, _ = make(models, clock, good)
    assert isinstance(embed(slots), Done)
    assert slots.healths[Slot.EMBED].skip_rate() == 0.0


def test_none_provider_skips_with_degraded_and_no_health_mark(
    models: ModelsConfig, clock: Clock
) -> None:
    config = dataclasses.replace(models, embed=dataclasses.replace(models.embed, provider="none"))
    slots, rec = make(config, clock, good)
    result = embed(slots)
    assert isinstance(result, Skipped)
    assert result.degraded["stage"] == "vector"
    assert "none" in result.degraded["reason"]
    assert rec.requests == []
    assert slots.healths[Slot.EMBED].skip_rate() == 0.0


def test_deadline_miss_marks_down_then_cpu_after_warmup(models: ModelsConfig, clock: Clock) -> None:
    slots, rec = make(models, clock, timeout_on_gpu)
    health = slots.healths[Slot.EMBED]
    first = embed(slots)
    assert isinstance(first, Skipped)
    assert first.degraded.get("device_tried") == "gpu"
    assert len(rec.requests) == 3 and health.active is Device.CPU
    second = embed(slots)
    assert isinstance(second, Done)
    assert rec.bodies()[-1]["options"] == {"num_gpu": 0}
    assert health.status().down == (Device.GPU,)


def test_one_device_per_call(models: ModelsConfig, clock: Clock) -> None:
    def always_fail(request: httpx.Request) -> httpx.Response:
        if b'"input":[' in request.content:
            raise httpx.ConnectError("down", request=request)
        return httpx.Response(200, json=VECTOR)

    slots, rec = make(models, clock, always_fail)
    embed(slots)
    calls = [b for b in rec.bodies() if isinstance(b["input"], list)]
    assert len(calls) == 1


def test_retry_after_expiry_restores_gpu(models: ModelsConfig, clock: Clock) -> None:
    gpu_broken = [True]

    def handler(request: httpx.Request) -> httpx.Response:
        return timeout_on_gpu(request) if gpu_broken[0] else good(request)

    slots, _ = make(models, clock, handler)
    health = slots.healths[Slot.EMBED]
    embed(slots)
    assert health.active is Device.CPU
    clock.advance(models.retry_after_s - 1)
    embed(slots)
    assert health.active is Device.CPU
    clock.advance(2)
    gpu_broken[0] = False
    assert isinstance(embed(slots), Done)
    assert health.active is Device.GPU


def test_no_ready_device_skips_without_attempt(models: ModelsConfig, clock: Clock) -> None:
    config = dataclasses.replace(
        models, embed=dataclasses.replace(models.embed, provider="openai_compat")
    )

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    slots, rec = make(config, clock, fail)
    first = embed(slots)
    assert isinstance(first, Skipped)
    sent = len(rec.requests)
    second = embed(slots)
    assert isinstance(second, Skipped)
    assert second.degraded["reason"] == "no device ready"
    assert len(rec.requests) == sent


def test_slow_call_by_clock_counts_as_deadline_miss(models: ModelsConfig, clock: Clock) -> None:
    slots, _ = make(models, clock, good)

    def slow() -> str:
        clock.advance(10)
        return "late"

    result = slots.call(Slot.EMBED, slow)
    assert isinstance(result, Skipped)
    assert slots.healths[Slot.EMBED].status().down == (Device.GPU,)
    assert ModelDeadlineError.code_value in ModelDeadlineError("x").code


def test_skip_rate_and_warn_threshold(models: ModelsConfig, clock: Clock) -> None:
    health = SlotHealth(
        Slot.EMBED,
        (Device.REMOTE,),
        warm=lambda device: None,
        submit=lambda job: job(),
        clock=clock,
        retry_after_s=60,
        skip_warn=0.05,
    )
    for _ in range(190):
        health.record_success()
    for _ in range(10):
        health.record_unavailable()
    assert health.skip_rate() == pytest.approx(0.05)
    assert not health.should_warn()
    health.record_unavailable()
    assert health.skip_rate() == pytest.approx(11 / 200)
    assert health.should_warn()
    assert health.status().skips_total == 11


def test_window_keeps_last_200(clock: Clock) -> None:
    health = SlotHealth(
        Slot.EMBED,
        (Device.REMOTE,),
        warm=lambda device: None,
        submit=lambda job: job(),
        clock=clock,
        retry_after_s=60,
        skip_warn=0.05,
    )
    for _ in range(200):
        health.record_unavailable()
    for _ in range(200):
        health.record_success()
    assert health.skip_rate() == 0.0


def test_residency_serve_unloads_and_reloads_on_cpu(models: ModelsConfig, clock: Clock) -> None:
    slots, rec = make(models, clock, good)
    slots.apply_profile(Profile.BULK)
    assert rec.requests == []
    slots.apply_profile(Profile.SERVE)
    bodies = rec.bodies()
    assert [b.get("keep_alive") for b in bodies if rec.paths()] == [0, "10m", 0, "10m"]
    assert rec.paths() == ["/api/embed", "/api/embed", "/api/chat", "/api/chat"]
    assert bodies[1]["options"] == {"num_gpu": 0}
    assert slots.healths[Slot.EMBED].active is Device.CPU
    assert slots.healths[Slot.ANNOTATE].active is Device.CPU
    slots.apply_profile(Profile.BULK)
    assert slots.healths[Slot.EMBED].active is Device.GPU


class StubDecision:
    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        return [[Decision(q.name, {"yes": 1.0}, "yes", 1.0) for q in questions] for _ in states]


class ChattyDecision(StubDecision):
    def annotate(self) -> None: ...


def classify_models(models: ModelsConfig) -> ModelsConfig:
    classify = ClassifyConfig(
        provider="stub",
        gpu=DeviceEndpointConfig("http://gpu", "serve-gpu --port 1"),
        cpu=DeviceEndpointConfig("http://cpu", ""),
    )
    return dataclasses.replace(models, classify=classify)


def stub_registries() -> ProviderRegistries:
    registries = default_registries()
    registries.classify.register("stub", lambda ctx: StubDecision())
    return registries


def test_classify_profiles_supervise_launch(models: ModelsConfig, clock: Clock) -> None:
    config = classify_models(models)
    launcher = FakeLauncher()
    slots = ModelSlots.from_config(
        config,
        registries=stub_registries(),
        transport=transport_of(recorder(good)),
        clock=clock,
        submit=lambda job: job(),
        launcher=launcher,
        probe=lambda endpoint: True,
    )
    health = slots.healths[Slot.CLASSIFY]
    slots.apply_profile(Profile.SERVE)
    assert [h.command for h in launcher.started] == ["serve-gpu --port 1"]
    assert health.active is Device.GPU
    question = Question("q", "t", QuestionKind.BINARY, ("yes", "no"))
    result = slots.call(Slot.CLASSIFY, lambda: slots.clients.decision.decide(["s"], [question]))
    assert isinstance(result, Done)
    slots.apply_profile(Profile.BULK)
    assert launcher.started[0].running is False
    stopped = slots.call(Slot.CLASSIFY, lambda: slots.clients.decision.decide(["s"], [question]))
    assert isinstance(stopped, Skipped)
    assert "residency" in stopped.degraded["reason"]


def test_external_classify_endpoint_only_health_checked(models: ModelsConfig, clock: Clock) -> None:
    launcher = FakeLauncher()
    backend = ClassifyBackend(
        ClassifyConfig(),
        launcher,
        lambda endpoint: False,
        clock=clock,
        sleep=lambda seconds: clock.advance(seconds),
        startup_timeout_s=2,
    )
    with pytest.raises(Exception, match="not answering"):
        backend.warm(Device.GPU)
    assert launcher.started == []


def test_classify_slot_rejects_non_decision_provider(models: ModelsConfig, clock: Clock) -> None:
    config = dataclasses.replace(
        models, classify=dataclasses.replace(models.classify, provider="ollama")
    )
    slots, _ = make(config, clock, good)
    assert any("classify" in issue for issue in slots.issues)
    assert slots.healths[Slot.CLASSIFY].chain == (Device.NONE,)
    registries = default_registries()
    registries.classify.register("chatty", lambda ctx: cast(DecisionClient, ChattyDecision()))
    bad = dataclasses.replace(
        models, classify=dataclasses.replace(models.classify, provider="chatty")
    )
    rejected, _ = make(bad, clock, good, registries=registries)
    assert any("rejected" in issue for issue in rejected.issues)


def test_unknown_embed_provider_disables_slot(models: ModelsConfig, clock: Clock) -> None:
    config = dataclasses.replace(
        models, embed=dataclasses.replace(models.embed, provider="nonesuch")
    )
    slots, _ = make(config, clock, good)
    assert isinstance(embed(slots), Skipped)
    assert slots.issues
