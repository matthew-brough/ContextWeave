import dataclasses
import time
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
from contextweave.models.device_http import device_table
from contextweave.models.errors import ModelDeadlineError, ModelRequestError, ModelTransportError
from contextweave.models.health import SlotHealth
from contextweave.models.providers import Device, EmbedKind, ProviderRegistries, Slot
from contextweave.models.residency import (
    Probe,
    Profile,
    ServerBackend,
    SubprocessLauncher,
    http_probe,
)
from contextweave.models.slots import Done, ModelSlots, Skipped, default_registries

VECTOR = {"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]}
GPU_EMBED_PORT = 8770
CPU_EMBED_PORT = 8771


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
        self.code: int | None = None
        self.stderr = ""

    def alive(self) -> bool:
        return self.running

    def stop(self) -> None:
        self.running = False

    def exit_code(self) -> int | None:
        return self.code

    def tail(self) -> str:
        return self.stderr


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
    if request.url.port == GPU_EMBED_PORT:
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
    assert len(rec.requests) == 1 and health.active is Device.CPU
    second = embed(slots)
    assert isinstance(second, Done)
    assert rec.requests[-1].url.port == CPU_EMBED_PORT
    assert health.status().down == (Device.GPU,)


def test_one_device_per_call(models: ModelsConfig, clock: Clock) -> None:
    def always_fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    slots, rec = make(models, clock, always_fail)
    embed(slots)
    assert len(rec.requests) == 1


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


def taking(clock: Clock, seconds: float) -> Callable[[], str]:
    def call() -> str:
        clock.advance(seconds)
        return "late"

    return call


def test_call_past_hard_deadline_counts_as_deadline_miss(
    models: ModelsConfig, clock: Clock
) -> None:
    slots, _ = make(models, clock, good)
    result = slots.call(Slot.EMBED, taking(clock, models.deadline_ms / 1000 + 1))
    assert isinstance(result, Skipped)
    assert slots.healths[Slot.EMBED].status().down == (Device.GPU,)
    assert ModelDeadlineError.code_value in ModelDeadlineError("x").code


def test_call_past_warn_is_kept_and_counted_slow(models: ModelsConfig, clock: Clock) -> None:
    slots, _ = make(models, clock, good)
    result = slots.call(Slot.EMBED, taking(clock, models.warn_ms / 1000 + 1))
    assert result == Done("late")
    status = slots.healths[Slot.EMBED].status()
    assert status.down == ()
    assert status.slow_rate == 1.0
    assert status.slow_warn
    assert status.skip_rate == 0.0


def test_rejected_request_skips_without_marking_device_down(
    models: ModelsConfig, clock: Clock
) -> None:
    slots, _ = make(models, clock, good)

    def rejected() -> str:
        raise ModelRequestError("/v1/embeddings returned 400")

    assert isinstance(slots.call(Slot.EMBED, rejected), Skipped)
    assert slots.healths[Slot.EMBED].status().down == ()
    assert slots.healths[Slot.EMBED].active is Device.GPU


def test_doc_embeds_use_the_batch_budget(models: ModelsConfig, clock: Clock) -> None:
    slots, _ = make(models, clock, good)
    query_cap = models.deadline_ms / 1000
    assert slots.embed_budget(EmbedKind.DOC).hard_s == models.embed.batch_deadline_ms / 1000
    assert slots.embed_budget(EmbedKind.QUERY).hard_s == query_cap
    doc = slots.call(
        Slot.EMBED, taking(clock, query_cap + 1), budget=slots.embed_budget(EmbedKind.DOC)
    )
    assert doc == Done("late")


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


def test_residency_serve_moves_to_cpu_servers_without_model_calls(
    models: ModelsConfig, clock: Clock
) -> None:
    probed: list[str] = []

    def probe(endpoint: str) -> bool:
        probed.append(endpoint)
        return True

    slots, rec = make(models, clock, good, probe=probe)
    slots.apply_profile(Profile.BULK_EMBED)
    slots.apply_profile(Profile.SERVE)
    assert rec.requests == []
    assert "http://127.0.0.1:8771" in probed
    assert "http://127.0.0.1:8773" in probed
    assert "http://127.0.0.1:8765" in probed
    assert slots.healths[Slot.EMBED].active is Device.CPU
    assert slots.healths[Slot.CLASSIFY].active is Device.GPU
    assert slots.healths[Slot.ANNOTATE].active is Device.CPU
    slots.apply_profile(Profile.BULK_EMBED)
    assert slots.healths[Slot.EMBED].active is Device.GPU


class StubDecision:
    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        return [[Decision(q.name, {"yes": 1.0}, "yes", 1.0) for q in questions] for _ in states]


class ChattyDecision(StubDecision):
    def annotate(self) -> None: ...


def classify_models(models: ModelsConfig, provider: str = "systemone") -> ModelsConfig:
    classify = ClassifyConfig(
        provider=provider,
        gpu=DeviceEndpointConfig("http://gpu", "serve-gpu --port 1"),
        cpu=DeviceEndpointConfig("http://cpu", ""),
    )
    return dataclasses.replace(models, classify=classify)


def stub_registries() -> ProviderRegistries:
    registries = default_registries()
    registries.classify.register("stub", lambda ctx: StubDecision())
    return registries


def noul_reply(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.8}}})


def supervised(models: ModelsConfig, clock: Clock, launcher: FakeLauncher) -> ModelSlots:
    return ModelSlots.from_config(
        models,
        transport=transport_of(recorder(noul_reply)),
        clock=clock,
        submit=lambda job: job(),
        launcher=launcher,
        probe=lambda endpoint: True,
    )


def test_classify_profiles_supervise_launch(models: ModelsConfig, clock: Clock) -> None:
    launcher = FakeLauncher()
    slots = supervised(classify_models(models), clock, launcher)
    health = slots.healths[Slot.CLASSIFY]
    slots.apply_profile(Profile.SERVE)
    assert [h.command for h in launcher.started] == ["serve-gpu --port 1"]
    assert health.active is Device.GPU
    question = Question("q", "t", QuestionKind.NOUL, ("yes", "no"))
    result = slots.ask(["s"], [question], stage="gate")
    assert isinstance(result, Done)
    slots.apply_profile(Profile.BULK_ANNOTATE)
    assert launcher.started[0].running is False
    stopped = slots.ask(["s"], [question], stage="gate")
    assert isinstance(stopped, Skipped)
    assert "residency" in stopped.degraded["reason"]


def test_warm_stops_other_device_process(models: ModelsConfig, clock: Clock) -> None:
    embed_config = dataclasses.replace(
        models.embed,
        gpu=DeviceEndpointConfig("http://127.0.0.1:8770", "embed-gpu"),
        cpu=DeviceEndpointConfig("http://127.0.0.1:8771", "embed-cpu"),
    )
    launcher = FakeLauncher()
    slots = supervised(dataclasses.replace(models, embed=embed_config), clock, launcher)
    health = slots.healths[Slot.EMBED]
    assert health.activate(Device.CPU)
    health.activate(Device.GPU)
    gpu_started = [h for h in launcher.started if h.command == "embed-gpu"]
    health.activate(Device.CPU)
    assert gpu_started and gpu_started[0].running is False
    assert [h.command for h in launcher.started if h.running] == ["embed-cpu"]


def test_bulk_profiles_hold_one_gpu_model_at_a_time(models: ModelsConfig, clock: Clock) -> None:
    gpu_models = dataclasses.replace(
        models,
        embed=dataclasses.replace(
            models.embed, gpu=DeviceEndpointConfig("http://127.0.0.1:8770", "embed-gpu")
        ),
        annotate=dataclasses.replace(
            models.annotate, gpu=DeviceEndpointConfig("http://127.0.0.1:8772", "annotate-gpu")
        ),
    )
    launcher = FakeLauncher()
    slots = supervised(gpu_models, clock, launcher)

    def running() -> list[str]:
        return [h.command for h in launcher.started if h.running]

    slots.apply_profile(Profile.BULK_ANNOTATE)
    assert running() == ["annotate-gpu"]
    slots.apply_profile(Profile.BULK_EMBED)
    assert running() == ["embed-gpu"]
    slots.release()
    assert running() == []


def test_unsupervised_classify_stub_is_never_probed(models: ModelsConfig, clock: Clock) -> None:
    probed: list[str] = []

    def probe(endpoint: str) -> bool:
        probed.append(endpoint)
        return True

    launcher = FakeLauncher()
    slots = ModelSlots.from_config(
        classify_models(models, "stub"),
        registries=stub_registries(),
        transport=transport_of(recorder(good)),
        clock=clock,
        submit=lambda job: job(),
        launcher=launcher,
        probe=probe,
    )
    slots.apply_profile(Profile.SERVE)
    assert launcher.started == []
    assert "http://gpu" not in probed
    assert slots.healths[Slot.CLASSIFY].active is Device.GPU


def test_external_classify_endpoint_only_health_checked(models: ModelsConfig, clock: Clock) -> None:
    launcher = FakeLauncher()
    config = ClassifyConfig()
    backend = ServerBackend(
        Slot.CLASSIFY,
        device_table(config.gpu, config.cpu),
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
        models, classify=dataclasses.replace(models.classify, provider="llama_server")
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


def test_annotate_slot_uses_its_own_deadline(models: ModelsConfig, clock: Clock) -> None:
    slots, _ = make(models, clock, good)
    slow = taking(clock, models.deadline_ms / 1000 + 1)
    assert isinstance(slots.call(Slot.ANNOTATE, slow), Done)
    assert isinstance(slots.call(Slot.EMBED, slow), Skipped)


def test_external_endpoint_is_probed_once(models: ModelsConfig, clock: Clock) -> None:
    probed: list[str] = []

    def probe(endpoint: str) -> bool:
        probed.append(endpoint)
        return False

    backend = ServerBackend(
        Slot.EMBED,
        device_table(models.embed.gpu, models.embed.cpu),
        FakeLauncher(),
        probe,
        clock=clock,
        sleep=lambda seconds: None,
    )
    with pytest.raises(ModelTransportError):
        backend.warm(Device.CPU)
    assert probed == ["http://127.0.0.1:8771"]


@pytest.mark.parametrize(
    ("routes", "alive"),
    [
        ({"/health": 200}, True),
        ({"/health": 503, "/v1/models": 200}, False),
        ({"/health": 404, "/v1/models": 200}, True),
        ({"/health": 404, "/v1/models": 404}, False),
    ],
)
def test_http_probe_falls_back_to_models_route(routes: dict[str, int], alive: bool) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(routes.get(request.url.path, 404))

    assert http_probe(httpx.MockTransport(handle))("http://kev/") is alive


class CrashingLauncher(FakeLauncher):
    def start(self, command: str) -> FakeHandle:
        handle = super().start(command)
        handle.running = False
        handle.code = 1
        handle.stderr = "loading model\nerror: failed to open weights.gguf"
        return handle


def launched(config: DeviceEndpointConfig) -> DeviceEndpointConfig:
    return dataclasses.replace(config, launch="llama-server -b 8192 -ub 8192 --port 8770")


def test_launched_server_that_exits_fails_fast_with_its_stderr(
    models: ModelsConfig, clock: Clock
) -> None:
    sleeps: list[float] = []
    backend = ServerBackend(
        Slot.EMBED,
        device_table(launched(models.embed.gpu), models.embed.cpu),
        CrashingLauncher(),
        lambda endpoint: False,
        clock=clock,
        sleep=sleeps.append,
    )
    with pytest.raises(ModelTransportError, match=r"exited with code 1.*failed to open weights"):
        backend.warm(Device.GPU)
    assert sleeps == []


def test_stop_all_during_warm_stops_the_launch_and_cancels_the_wait(
    models: ModelsConfig, clock: Clock
) -> None:
    launcher = FakeLauncher()
    backend = ServerBackend(
        Slot.EMBED,
        device_table(launched(models.embed.gpu), models.embed.cpu),
        launcher,
        lambda endpoint: False,
        clock=clock,
        sleep=lambda seconds: backend.stop_all(),
    )
    with pytest.raises(ModelTransportError, match="start cancelled"):
        backend.warm(Device.GPU)
    assert [handle.running for handle in launcher.started] == [False]


def test_external_endpoint_failure_names_the_launch_key(models: ModelsConfig, clock: Clock) -> None:
    backend = ServerBackend(
        Slot.EMBED,
        device_table(models.embed.gpu, models.embed.cpu),
        FakeLauncher(),
        lambda endpoint: False,
        clock=clock,
    )
    with pytest.raises(ModelTransportError, match=r"set models\.embed\.gpu\.launch"):
        backend.warm(Device.GPU)


def test_launch_expands_before_start(models: ModelsConfig, clock: Clock) -> None:
    launcher = FakeLauncher()
    backend = ServerBackend(
        Slot.EMBED,
        device_table(launched(models.embed.gpu), models.embed.cpu),
        launcher,
        lambda endpoint: True,
        clock=clock,
        expand=lambda command: f"{command} -m /w.gguf",
    )
    backend.warm(Device.GPU)
    assert launcher.started[0].command == "llama-server -b 8192 -ub 8192 --port 8770 -m /w.gguf"


def test_start_failure_shows_in_status_reason(models: ModelsConfig, clock: Clock) -> None:
    config = dataclasses.replace(
        models,
        embed=dataclasses.replace(models.embed, gpu=launched(models.embed.gpu), devices=("gpu",)),
    )
    slots = ModelSlots.from_config(
        config,
        transport=transport_of(recorder(good)),
        clock=clock,
        submit=lambda job: job(),
        launcher=CrashingLauncher(),
        probe=lambda endpoint: False,
    )
    result = embed(slots)
    assert isinstance(result, Skipped)
    status = slots.healths[Slot.EMBED].status()
    assert status.active is None
    assert status.reason is not None and "failed to open weights" in status.reason
    assert "failed to open weights" in embed_reason(slots)


def embed_reason(slots: ModelSlots) -> str:
    result = embed(slots)
    assert isinstance(result, Skipped)
    return result.degraded["reason"]


def test_subprocess_launcher_keeps_stderr_tail(models: ModelsConfig, clock: Clock) -> None:
    gpu = dataclasses.replace(models.embed.gpu, launch="sh -c 'echo boom >&2; exit 3'")
    backend = ServerBackend(
        Slot.EMBED,
        device_table(gpu, models.embed.cpu),
        SubprocessLauncher(),
        lambda endpoint: False,
        sleep=lambda seconds: time.sleep(0.05),
        startup_timeout_s=10,
    )
    with pytest.raises(ModelTransportError, match="code 3: boom"):
        backend.warm(Device.GPU)


def test_subprocess_launcher_reports_missing_binary() -> None:
    with pytest.raises(ModelTransportError, match="cannot launch"):
        SubprocessLauncher().start("/nonexistent/llama-server --port 1")
