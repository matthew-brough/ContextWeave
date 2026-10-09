from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from contextweave.core.config import ClassifyConfig, Config, ModelsConfig
from contextweave.models.calibration import Calibration
from contextweave.models.decision import (
    NO,
    YES,
    Decision,
    Question,
    QuestionKind,
    decision_of,
    score_distribution,
)
from contextweave.models.errors import ModelTransportError
from contextweave.models.providers import Device
from contextweave.models.slots import ModelSlots, default_registries


def yes_when_good(state: str) -> float:
    return 0.95 if "good" in state else 0.05


@dataclass(slots=True)
class FakeDecision:
    """Deterministic decision client: a yes-probability per state, fixed choices per question."""

    yes: Callable[[str], float] = yes_when_good
    choices: Mapping[str, str] = field(default_factory=dict[str, str])
    down: bool = False
    short_rows: bool = False
    calls: list[Sequence[str]] = field(default_factory=list[Sequence[str]])
    asked: list[Sequence[Question]] = field(default_factory=list[Sequence[Question]])
    device: Device = Device.GPU

    @property
    def devices(self) -> tuple[Device, ...]:
        return (Device.GPU, Device.CPU)

    def set_device(self, device: Device) -> None:
        self.device = device

    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        self.calls.append(list(states))
        self.asked.append(list(questions))
        if self.down:
            raise ModelTransportError("fake decision is down")
        rows = [[self._answer(state, question) for question in questions] for state in states]
        return rows[:-1] if self.short_rows and rows else rows

    def _answer(self, state: str, question: Question) -> Decision:
        yes = self.yes(state)
        match question.kind:
            case QuestionKind.BINARY:
                return decision_of(question, {YES: yes, NO: 1 - yes})
            case QuestionKind.SCORE:
                return decision_of(question, score_distribution(yes))
            case QuestionKind.CHOICE:
                chosen = self.choices.get(question.name, question.options[0])
                return decision_of(question, {chosen: 1.0})


def classify_models(provider: str = "fake") -> ModelsConfig:
    return ModelsConfig(classify=ClassifyConfig(provider=provider))


def fake_slots(
    client: FakeDecision,
    *,
    clock: Callable[[], float] = lambda: 0.0,
    calibration: Calibration | None = None,
    config: Config | None = None,
) -> ModelSlots:
    registries = default_registries()
    registries.classify.register("fake", lambda ctx: client)
    models = (config or replace(Config(), models=classify_models())).models
    return ModelSlots.from_config(
        models,
        registries=registries,
        clock=clock,
        submit=lambda job: job(),
        probe=lambda endpoint: True,
        calibration=calibration,
    )
