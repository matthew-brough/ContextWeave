from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contextweave.core.config import DeviceEndpointConfig
from contextweave.core.types import JSON
from contextweave.models.decision import (
    Decision,
    Question,
    QuestionKind,
    decision_of,
    score_distribution,
)
from contextweave.models.errors import ModelResponseError, ProviderSlotError
from contextweave.models.http import JsonClient
from contextweave.models.providers import BuildContext, Device
from contextweave.models.wire import as_list, as_object, as_str, field

DECISION_DEVICES = (Device.GPU, Device.CPU)


@dataclass(frozen=True, slots=True)
class DecisionWire:
    """How one model family names the decide request and reads its answers."""

    path: str
    kind_names: Mapping[QuestionKind, str]
    native_score: bool


def _number(value: JSON, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelResponseError(f"{what}: expected number")
    return float(value)


class HttpDecisionClient:
    """Decision protocol over HTTP, one endpoint per device; the active device is set by health."""

    def __init__(self, wire: DecisionWire, ctx: BuildContext) -> None:
        classify = ctx.models.classify
        endpoints: Mapping[Device, DeviceEndpointConfig] = {
            Device.GPU: classify.gpu,
            Device.CPU: classify.cpu,
        }
        self._wire = wire
        self._model = classify.model
        self._deadline_s = ctx.deadline_s
        self._chain = tuple(
            device
            for name in classify.devices
            for device in DECISION_DEVICES
            if device.value == name and endpoints[device].endpoint
        )
        self._http = {
            device: JsonClient(endpoints[device].endpoint, transport=ctx.transport)
            for device in self._chain
        }
        self._device = self._chain[0] if self._chain else Device.NONE

    @property
    def devices(self) -> tuple[Device, ...]:
        return self._chain

    @property
    def device(self) -> Device:
        return self._device

    def set_device(self, device: Device) -> None:
        if device not in self._http:
            raise ProviderSlotError(f"decision client has no device '{device}'")
        self._device = device

    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        if not states:
            return []
        http = self._http.get(self._device)
        if http is None:
            raise ProviderSlotError("decision client has no active device")
        body = self._request(states, questions)
        reply = http.post(self._wire.path, body, timeout_s=self._deadline_s)
        return self._parse(reply, len(states), questions)

    def _request(self, states: Sequence[str], questions: Sequence[Question]) -> dict[str, JSON]:
        asked: list[JSON] = [self._question(question) for question in questions]
        return {"model": self._model, "states": list(states), "questions": asked}

    def _question(self, question: Question) -> JSON:
        native = self._wire.native_score and question.kind is QuestionKind.SCORE
        options: list[JSON] = [] if native else list(question.options)
        return {
            "name": question.name,
            "text": question.text,
            "type": self._wire.kind_names[question.kind],
            "options": options,
        }

    def _parse(
        self, reply: Mapping[str, JSON], states: int, questions: Sequence[Question]
    ) -> list[list[Decision]]:
        rows = as_list(field(reply, "results", "decide"), "results")
        if len(rows) != states:
            raise ModelResponseError(f"expected {states} result rows, got {len(rows)}")
        parsed: list[list[Decision]] = []
        for row in rows:
            answers = as_list(row, "result row")
            if len(answers) != len(questions):
                raise ModelResponseError("result row does not match the question count")
            parsed.append(
                [
                    self._decision(question, as_object(answer, "answer"))
                    for question, answer in zip(questions, answers, strict=True)
                ]
            )
        return parsed

    def _decision(self, question: Question, answer: Mapping[str, JSON]) -> Decision:
        if as_str(field(answer, "name", "answer"), "name") != question.name:
            raise ModelResponseError(f"answer out of order for '{question.name}'")
        if self._wire.native_score and question.kind is QuestionKind.SCORE:
            value = _number(field(answer, "value", question.name), question.name)
            return decision_of(question, score_distribution(value))
        probs = as_object(field(answer, "probs", question.name), question.name)
        return decision_of(
            question, {option: _number(prob, option) for option, prob in probs.items()}
        )
