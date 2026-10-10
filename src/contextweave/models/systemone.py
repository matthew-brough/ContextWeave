from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

from contextweave.core.types import JSON
from contextweave.models.decision import (
    Decision,
    DecisionClient,
    Question,
    QuestionKind,
    decision_of,
    noul_distribution,
    score_distribution,
)
from contextweave.models.device_http import DeviceHttp, device_table
from contextweave.models.errors import ModelResponseError
from contextweave.models.http import JsonClient
from contextweave.models.providers import BuildContext
from contextweave.models.wire import as_object, as_str, field

SYSTEMONE_PATH = "/v1/systemone"
PROBABILITIES = "probabilities"


def _number(value: JSON, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelResponseError(f"{what}: expected number")
    return float(value)


def question_payload(question: Question) -> JSON:
    """One question on the System One wire: criteria map for choice, ordered list for score."""
    payload: dict[str, JSON] = {"type": question.kind.value, "instructions": question.text}
    if question.kind is QuestionKind.CHOICE:
        payload["criteria"] = {
            option: question.description(index) for index, option in enumerate(question.options)
        }
    elif question.kind is QuestionKind.SCORE:
        payload["criteria"] = [
            question.description(index) for index in range(len(question.options))
        ]
    return payload


def request_body(model: str, state: str, questions: Sequence[Question]) -> dict[str, JSON]:
    """One state's System One request."""
    return {
        "model": model,
        "state": state,
        "questions": {question.name: question_payload(question) for question in questions},
    }


def _score_probs(question: Question, raw: Mapping[str, JSON]) -> dict[str, float]:
    keys = {question.description(index): option for index, option in enumerate(question.options)}
    keys.update({option: option for option in question.options})
    keys.update({str(index): option for index, option in enumerate(question.options)})
    probs: dict[str, float] = {}
    for key, value in raw.items():
        option = keys.get(key)
        if option is None:
            raise ModelResponseError(f"question '{question.name}': unknown score level '{key}'")
        probs[option] = probs.get(option, 0.0) + _number(value, key)
    return probs


def decision_from(question: Question, answer: Mapping[str, JSON]) -> Decision:
    """Read one answer object into a decision over the question's options."""
    kind = as_str(field(answer, "type", question.name), "type")
    if kind != question.kind.value:
        raise ModelResponseError(f"question '{question.name}': answered as '{kind}'")
    match question.kind:
        case QuestionKind.NOUL:
            yes = _number(field(answer, "noul", question.name), question.name)
            return decision_of(question, noul_distribution(yes))
        case QuestionKind.SCORE:
            raw = answer.get(PROBABILITIES)
            if isinstance(raw, dict) and raw:
                return decision_of(question, _score_probs(question, raw))
            score = _number(field(answer, "score", question.name), question.name)
            return decision_of(question, score_distribution(question.options, score))
        case QuestionKind.CHOICE:
            probs = as_object(field(answer, PROBABILITIES, question.name), question.name)
            return decision_of(
                question, {option: _number(prob, option) for option, prob in probs.items()}
            )


def parse_reply(reply: Mapping[str, JSON], questions: Sequence[Question]) -> list[Decision]:
    """One state's decisions in question order."""
    answers = as_object(field(reply, "answers", SYSTEMONE_PATH), "answers")
    row: list[Decision] = []
    for question in questions:
        answer = answers.get(question.name)
        if answer is None:
            raise ModelResponseError(f"no answer for '{question.name}'")
        row.append(decision_from(question, as_object(answer, question.name)))
    return row


class SystemOneClient(DeviceHttp):
    """Decision protocol over the System One wire; one request per state, sent concurrently."""

    def __init__(self, ctx: BuildContext) -> None:
        config = ctx.models.classify
        super().__init__(
            "classify", config.devices, device_table(config.gpu, config.cpu), ctx.transport
        )
        self._model = config.model
        self._parallel = max(config.parallel, 1)
        self._deadline_s = ctx.deadline_s

    def decide(self, states: Sequence[str], questions: Sequence[Question]) -> list[list[Decision]]:
        if not states:
            return []
        http = self.active()
        if len(states) == 1:
            return [self._one(http, states[0], questions)]
        workers = min(self._parallel, len(states))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cw-decide") as pool:
            futures = [pool.submit(self._one, http, state, questions) for state in states]
            return [future.result() for future in futures]

    def _one(self, http: JsonClient, state: str, questions: Sequence[Question]) -> list[Decision]:
        body = request_body(self._model, state, questions)
        reply = http.post(SYSTEMONE_PATH, body, timeout_s=self._deadline_s)
        return parse_reply(reply, questions)


def build_systemone(ctx: BuildContext) -> DecisionClient:
    """Factory for the systemone classify provider."""
    return SystemOneClient(ctx)
