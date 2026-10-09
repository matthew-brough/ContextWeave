from dataclasses import dataclass, field

import httpx

from contextweave.core.config import GateConfig, ModelsConfig
from contextweave.models.probe.fixtures import ProbeSet
from contextweave.models.probe.host import HostInspector
from contextweave.models.probe.timing import Clock
from contextweave.models.providers import Device, Slot
from contextweave.models.slots import ModelSlots

DEFAULT_PROBE_DEADLINE_MS = 120_000
DEFAULT_VRAM_MIB = 8192
DEFAULT_PIPELINE_BUDGET_MS = 1000
DEFAULT_WINDOWS = (1, 5, 10, 20, 30)
MIN_CPU_WINDOW = 10
EMBED_CPU_QUERY_BUDGET_MS = 250.0
ALL_SLOTS = (Slot.EMBED, Slot.ANNOTATE, Slot.CLASSIFY)
ALL_DEVICES = (Device.GPU, Device.CPU)
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass(frozen=True, slots=True)
class ClassifyCandidate:
    family: str
    rung: int
    build_id: str
    model: str
    gpu_endpoint: str
    cpu_endpoint: str

    @property
    def label(self) -> str:
        return f"{self.family}.r{self.rung}.{self.build_id}"

    def endpoint(self, device: Device) -> str:
        return self.gpu_endpoint if device is Device.GPU else self.cpu_endpoint


@dataclass(frozen=True, slots=True)
class ProbeOptions:
    slots: tuple[Slot, ...] = ALL_SLOTS
    devices: tuple[Device, ...] = ALL_DEVICES
    runs: int = 20
    warmup: int = 2
    annotate_chunks: int = 6
    classify_runs: int = 5
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    vram_mib: int = DEFAULT_VRAM_MIB
    pipeline_budget_ms: int = DEFAULT_PIPELINE_BUDGET_MS
    candidates: tuple[ClassifyCandidate, ...] = ()


@dataclass(slots=True)
class ProbeEnv:
    """Everything a probe needs: slots to call, configured limits, and injectable system access."""

    models: ModelsConfig
    configured_deadline_ms: int
    gate: GateConfig
    options: ProbeOptions
    slots: ModelSlots
    probe_set: ProbeSet
    host: HostInspector
    clock: Clock
    ollama_url: str = DEFAULT_OLLAMA_URL
    transport: httpx.BaseTransport | None = None
    notes: list[str] = field(default_factory=list[str])

    @property
    def gate_budget_ms(self) -> float:
        return float(min(self.configured_deadline_ms, self.options.pipeline_budget_ms))
