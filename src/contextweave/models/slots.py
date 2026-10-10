import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from contextweave.core.config import ModelsConfig
from contextweave.core.errors import DegradedEntry
from contextweave.core.registry import UnknownKeyError
from contextweave.core.types import JSON
from contextweave.models.calibration import Calibration, CalibrationKey
from contextweave.models.decision import (
    Answers,
    DecisionClient,
    NullDecisionClient,
    Question,
    answers_from,
    ensure_decision_client,
)
from contextweave.models.errors import (
    ModelDeadlineError,
    ModelDisabledError,
    ModelError,
    ModelParseError,
    ProviderSlotError,
)
from contextweave.models.health import Clock, HealthStatus, SlotHealth, Submit
from contextweave.models.llama_server import build_llama_annotate, build_llama_embed
from contextweave.models.null import NONE_CHAIN, build_null_annotate, build_null_embed
from contextweave.models.openai_compat import build_openai_annotate, build_openai_embed
from contextweave.models.providers import (
    AnnotateClient,
    BuildContext,
    Device,
    EmbedClient,
    EmbedKind,
    Endpoints,
    Message,
    Provider,
    ProviderRegistries,
    Slot,
    Supervised,
)
from contextweave.models.residency import (
    Probe,
    ProcessLauncher,
    Profile,
    Residency,
    ServerBackend,
    SubprocessLauncher,
    http_probe,
)
from contextweave.models.systemone import build_systemone

NONE = "none"
LLAMA_SERVER = "llama_server"
SYSTEMONE = "systemone"


@dataclass(frozen=True, slots=True)
class Done[T]:
    value: T


@dataclass(frozen=True, slots=True)
class Skipped:
    degraded: DegradedEntry


type Outcome[T] = Done[T] | Skipped


def default_registries() -> ProviderRegistries:
    """Registries with the built-in providers."""
    registries = ProviderRegistries()
    registries.embed.register(LLAMA_SERVER, build_llama_embed)
    registries.embed.register("openai_compat", build_openai_embed)
    registries.embed.register(NONE, build_null_embed)
    registries.annotate.register(LLAMA_SERVER, build_llama_annotate)
    registries.annotate.register("openai_compat", build_openai_annotate)
    registries.annotate.register(NONE, build_null_annotate)
    registries.classify.register(SYSTEMONE, build_systemone)
    registries.classify.register(NONE, lambda ctx: NullDecisionClient())
    return registries


def thread_submit() -> Submit:
    """Submit jobs to a single background worker thread."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cw-warm")

    def submit(job: Callable[[], None]) -> None:
        pool.submit(job)

    return submit


@dataclass(frozen=True, slots=True)
class SlotClients:
    embed: EmbedClient
    annotate: AnnotateClient
    decision: DecisionClient


@dataclass(slots=True)
class ModelSlots:
    """Facade over the three slots: applies health and deadline, never raises to the pipeline."""

    clients: SlotClients
    healths: Mapping[Slot, SlotHealth]
    residency: Residency
    deadlines_s: Mapping[Slot, float]
    clock: Clock
    issues: tuple[str, ...] = field(default=())
    temperatures: Mapping[str, float] = field(default_factory=dict[str, float])

    @classmethod
    def from_config(
        cls,
        models: ModelsConfig,
        *,
        registries: ProviderRegistries | None = None,
        endpoints: Endpoints | None = None,
        transport: httpx.BaseTransport | None = None,
        clock: Clock = time.monotonic,
        submit: Submit | None = None,
        launcher: ProcessLauncher | None = None,
        probe: Probe | None = None,
        calibration: Calibration | None = None,
    ) -> ModelSlots:
        """Build clients, health state and residency from config."""
        available = registries or default_registries()
        ctx = BuildContext(models, endpoints or Endpoints(), transport)
        issues: list[str] = []
        embed = _build(available.embed.get, models.embed.provider, Slot.EMBED, ctx, issues)
        annotate = _build(
            available.annotate.get, models.annotate.provider, Slot.ANNOTATE, ctx, issues
        )
        decision = _build_decision(available, models.classify.provider, ctx, issues)
        embed = embed or build_null_embed(ctx)
        annotate = annotate or build_null_annotate(ctx)
        classify_off = decision is None or isinstance(decision, NullDecisionClient)
        decision = decision or NullDecisionClient()
        providers: Mapping[Slot, object] = {
            Slot.EMBED: embed,
            Slot.ANNOTATE: annotate,
            Slot.CLASSIFY: decision,
        }
        backends = _backends(
            providers, launcher or SubprocessLauncher(), probe or http_probe(transport), clock
        )
        chains: Mapping[Slot, tuple[Device, ...]] = {
            Slot.EMBED: tuple(embed.devices),
            Slot.ANNOTATE: tuple(annotate.devices),
            Slot.CLASSIFY: NONE_CHAIN
            if classify_off
            else _decision_chain(decision, models, issues),
        }
        warms = {slot: _warm(backends.get(slot), providers[slot]) for slot in Slot}
        worker = submit or thread_submit()
        healths = {
            slot: SlotHealth(
                slot,
                chains[slot],
                warm=warms[slot],
                submit=worker,
                clock=clock,
                retry_after_s=models.retry_after_s,
                skip_warn=models.skip_warn,
                loaded=slot is not Slot.CLASSIFY or classify_off,
            )
            for slot in Slot
        }
        return cls(
            SlotClients(embed, annotate, decision),
            healths,
            Residency(models, healths, backends),
            {
                Slot.EMBED: models.deadline_ms / 1000,
                Slot.ANNOTATE: models.annotate.deadline_ms / 1000,
                Slot.CLASSIFY: models.deadline_ms / 1000,
            },
            clock,
            tuple(issues),
            (calibration or Calibration()).temperatures(_calibration_key(models)),
        )

    def call[T](self, slot: Slot, fn: Callable[[], T], *, stage: str | None = None) -> Outcome[T]:
        """Run one model call on the slot's active device; failures become a Skipped result."""
        name = stage or slot.value
        health = self.healths[slot]
        device = health.acquire()
        if device is None:
            health.record_unavailable()
            return Skipped(DegradedEntry(stage=name, reason=health.unavailable_reason()))
        deadline_s = self.deadlines_s[slot]
        started = self.clock()
        try:
            value = fn()
            if self.clock() - started > deadline_s:
                raise ModelDeadlineError(f"call exceeded {deadline_s:.3f}s")
        except ModelDisabledError as exc:
            return Skipped(DegradedEntry(stage=name, reason=str(exc)))
        except ModelParseError as exc:
            health.record_success()
            return Skipped(DegradedEntry(stage=name, reason=str(exc), device_tried=str(device)))
        except ModelError as exc:
            health.record_failure(device)
            return Skipped(DegradedEntry(stage=name, reason=str(exc), device_tried=str(device)))
        except Exception as exc:
            health.record_failure(device)
            reason = str(exc) or type(exc).__name__
            return Skipped(DegradedEntry(stage=name, reason=reason, device_tried=str(device)))
        health.record_success()
        return Done(value)

    def embed(
        self, texts: Sequence[str], *, kind: EmbedKind, stage: str | None = None
    ) -> Outcome[list[list[float]]]:
        """Embed texts through the embed slot."""
        return self.call(
            Slot.EMBED, lambda: self.clients.embed.embed(texts, kind=kind), stage=stage
        )

    def annotate(
        self,
        prompt_messages: Sequence[Message],
        json_schema: Mapping[str, JSON],
        *,
        stage: str | None = None,
    ) -> Outcome[Mapping[str, JSON]]:
        """Annotate through the annotate slot."""
        return self.call(
            Slot.ANNOTATE,
            lambda: self.clients.annotate.annotate(prompt_messages, json_schema),
            stage=stage,
        )

    def ask(
        self, states: Sequence[str], questions: Sequence[Question], *, stage: str
    ) -> Outcome[list[Answers]]:
        """One decide() call over all states; shape-checked and wrapped with the fitted temperatures."""

        def run() -> list[Answers]:
            rows = self.clients.decision.decide(states, questions)
            return answers_from(rows, len(states), questions, self.temperatures)

        return self.call(Slot.CLASSIFY, run, stage=stage)

    def apply_profile(self, profile: Profile) -> None:
        """Switch residency profile."""
        self.residency.apply(profile)

    def status(self) -> tuple[HealthStatus, ...]:
        """Per-slot health snapshots."""
        return tuple(health.status() for health in self.healths.values())


def _calibration_key(models: ModelsConfig) -> CalibrationKey:
    classify = models.classify
    return CalibrationKey(classify.model, classify.build_id, classify.question_set_version)


def _backends(
    providers: Mapping[Slot, object], launcher: ProcessLauncher, probe: Probe, clock: Clock
) -> dict[Slot, ServerBackend]:
    return {
        slot: ServerBackend(slot, provider.servers, launcher, probe, clock=clock)
        for slot, provider in providers.items()
        if isinstance(provider, Supervised)
    }


def _warm(backend: ServerBackend | None, provider: object) -> Callable[[Device], None]:
    def warm(device: Device) -> None:
        if backend is not None:
            backend.warm(device)
        if isinstance(provider, Provider):
            provider.set_device(device)

    return warm


def _build[C](
    lookup: Callable[[str], Callable[[BuildContext], C]],
    name: str,
    slot: Slot,
    ctx: BuildContext,
    issues: list[str],
) -> C | None:
    try:
        return lookup(name)(ctx)
    except UnknownKeyError:
        issues.append(f"{slot} provider '{name}' is not available; slot disabled")
    return None


def _build_decision(
    registries: ProviderRegistries, name: str, ctx: BuildContext, issues: list[str]
) -> DecisionClient | None:
    try:
        return ensure_decision_client(registries.classify.get(name)(ctx))
    except UnknownKeyError:
        issues.append(f"classify provider '{name}' is not a decision provider; slot disabled")
    except ProviderSlotError as exc:
        issues.append(f"classify provider '{name}' rejected: {exc}")
    return None


def _decision_chain(
    decision: DecisionClient, models: ModelsConfig, issues: list[str]
) -> tuple[Device, ...]:
    if isinstance(decision, Provider):
        return tuple(decision.devices)
    chain: list[Device] = []
    for name in models.classify.devices:
        try:
            chain.append(Device(name))
        except ValueError:
            issues.append(f"classify device '{name}' unknown; ignored")
    return tuple(chain)
