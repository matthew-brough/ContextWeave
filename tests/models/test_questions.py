import json
from collections.abc import Callable, Mapping
from pathlib import Path

import httpx
import pytest
from models_support import recorder, transport_of

from contextweave.core.config import ClassifyConfig, DeviceEndpointConfig, ModelsConfig
from contextweave.core.types import JSON
from contextweave.models.decision import CHOICE_OPTION_LIMIT, Question, QuestionKind
from contextweave.models.errors import ModelResponseError
from contextweave.models.providers import BuildContext, Endpoints
from contextweave.models.questions import QuestionSetError, load_question_set
from contextweave.models.systemone import SystemOneClient, build_systemone


@pytest.mark.parametrize("stage", ["plan", "gate", "coverage"])
def test_shipped_sets_load(stage: str) -> None:
    assert load_question_set(stage, 2).build({"scope_dirs": ["src"]})


def test_version_mismatch_raises() -> None:
    with pytest.raises(QuestionSetError, match="version"):
        load_question_set("gate", 1)


def test_missing_set_raises() -> None:
    with pytest.raises(QuestionSetError):
        load_question_set("nope", 2)


def test_scope_options_come_from_dynamic_values() -> None:
    questions = {
        q.name: q for q in load_question_set("plan", 2).build({"scope_dirs": ["src", "tests"]})
    }
    assert questions["scope"].options == ("any", "src", "tests")


def test_malformed_file_raises(tmp_path: Path) -> None:
    (tmp_path / "bad.toml").write_text("question_set_version = 1\n", encoding="utf-8")
    with pytest.raises(QuestionSetError, match="no"):
        load_question_set("bad", 1, tmp_path)


def test_no_question_literals_in_python() -> None:
    root = Path(__file__).parents[2] / "src" / "contextweave"
    texts = [
        question.text
        for stage in ("plan", "gate", "coverage")
        for question in load_question_set(stage, 2).build({})
    ]
    code = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert not [text for text in texts if text in code]


def models_for(parallel: int = 8) -> ModelsConfig:
    endpoint = DeviceEndpointConfig(endpoint="http://gpu.test")
    classify = ClassifyConfig(devices=("gpu",), gpu=endpoint, parallel=parallel)
    return ModelsConfig(classify=classify)


def client_for(
    reply: Callable[[dict[str, JSON]], Mapping[str, JSON]], parallel: int = 8
) -> tuple[SystemOneClient, list[httpx.Request]]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=dict(reply(json.loads(request.content))))

    rec = recorder(handler)
    ctx = BuildContext(models_for(parallel), Endpoints(), transport_of(rec))
    client = build_systemone(ctx)
    assert isinstance(client, SystemOneClient)
    return client, rec.requests


SCORE = Question(
    "relevance",
    "r",
    QuestionKind.SCORE,
    ("off_topic", "related", "partial", "full"),
    ("Unrelated.", "Same area.", "Part of it.", "All of it."),
)
NOUL = Question("answers_query", "a", QuestionKind.NOUL, ("yes", "no"))
CHOICE = Question(
    "role", "c", QuestionKind.CHOICE, ("impl", "test"), ("Implements it.", "Tests it.")
)


def answers(**items: JSON) -> Callable[[dict[str, JSON]], Mapping[str, JSON]]:
    return lambda body: {"model": "m", "answers": dict(items)}


def test_request_body_is_one_state_with_keyed_typed_questions() -> None:
    reply = answers(
        answers_query={"type": "noul", "noul": 0.9, "confidence": 0.9},
        relevance={"type": "score", "score": 1.0},
        role={"type": "choice", "choice": "impl", "probabilities": {"impl": 0.7, "test": 0.3}},
    )
    client, requests = client_for(reply)
    client.decide(["s"], [NOUL, SCORE, CHOICE])
    sent = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/systemone"
    assert sent == {
        "model": "kev-latest",
        "state": "s",
        "questions": {
            "answers_query": {"type": "noul", "instructions": "a"},
            "relevance": {
                "type": "score",
                "instructions": "r",
                "criteria": ["Unrelated.", "Same area.", "Part of it.", "All of it."],
            },
            "role": {
                "type": "choice",
                "instructions": "c",
                "criteria": {"impl": "Implements it.", "test": "Tests it."},
            },
        },
    }


def test_reply_parses_noul_score_and_choice() -> None:
    reply = answers(
        answers_query={"type": "noul", "noul": 0.9},
        relevance={"type": "score", "score": 0.5},
        role={"type": "choice", "probabilities": {"impl": 0.2, "test": 0.8}},
    )
    client, _ = client_for(reply)
    row = client.decide(["s"], [NOUL, SCORE, CHOICE])[0]
    assert row[0].probs == pytest.approx({"yes": 0.9, "no": 0.1})
    assert row[1].probs["related"] == pytest.approx(0.5)
    assert row[1].probs["partial"] == pytest.approx(0.5)
    assert row[2].top == "test"


@pytest.mark.parametrize(
    "probabilities",
    [
        {"Unrelated.": 0.1, "Same area.": 0.1, "Part of it.": 0.1, "All of it.": 0.7},
        {"off_topic": 0.1, "related": 0.1, "partial": 0.1, "full": 0.7},
        {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.7},
    ],
)
def test_score_probabilities_keyed_by_description_label_or_index(
    probabilities: dict[str, JSON],
) -> None:
    client, _ = client_for(answers(relevance={"type": "score", "probabilities": probabilities}))
    decision = client.decide(["s"], [SCORE])[0][0]
    assert decision.top == "full"
    assert decision.probs["full"] == pytest.approx(0.7)


def test_unknown_score_level_is_a_response_error() -> None:
    reply = answers(relevance={"type": "score", "probabilities": {"huge": 1.0}})
    client, _ = client_for(reply)
    with pytest.raises(ModelResponseError):
        client.decide(["s"], [SCORE])


def test_missing_answer_is_a_response_error() -> None:
    client, _ = client_for(answers())
    with pytest.raises(ModelResponseError):
        client.decide(["s"], [NOUL])


def test_wrong_answer_type_is_a_response_error() -> None:
    client, _ = client_for(answers(answers_query={"type": "score", "score": 0.5}))
    with pytest.raises(ModelResponseError):
        client.decide(["s"], [NOUL])


def test_states_sent_one_per_request_and_rows_keep_state_order() -> None:
    def reply(body: dict[str, JSON]) -> Mapping[str, JSON]:
        yes = 0.9 if body["state"] == "good" else 0.1
        return {"answers": {"answers_query": {"type": "noul", "noul": yes}}}

    client, requests = client_for(reply, parallel=3)
    states = ["good", "bad", "good", "bad", "bad"]
    rows = client.decide(states, [NOUL])
    assert len(requests) == len(states)
    assert [row[0].top for row in rows] == ["yes", "no", "yes", "no", "no"]
    assert sorted(json.loads(r.content)["state"] for r in requests) == sorted(states)


def test_no_states_sends_nothing() -> None:
    client, requests = client_for(answers())
    assert client.decide([], [NOUL]) == []
    assert requests == []


def write_set(tmp_path: Path, body: str) -> Path:
    (tmp_path / "x.toml").write_text(f"question_set_version = 2\n{body}", encoding="utf-8")
    return tmp_path


def test_choice_over_option_cap_rejected(tmp_path: Path) -> None:
    criteria = ", ".join(f'{{ label = "o{index}" }}' for index in range(CHOICE_OPTION_LIMIT + 1))
    root = write_set(
        tmp_path,
        f'[[question]]\nname = "q"\ntext = "t"\nkind = "choice"\ncriteria = [{criteria}]\n',
    )
    with pytest.raises(QuestionSetError):
        load_question_set("x", 2, root)


def test_dynamic_options_truncate_at_cap() -> None:
    dirs = [f"d{index}" for index in range(40)]
    scope = {q.name: q for q in load_question_set("plan", 2).build({"scope_dirs": dirs})}["scope"]
    assert len(scope.options) == CHOICE_OPTION_LIMIT
    assert scope.description(1) == "Code under the top-level directory d0/."


@pytest.mark.parametrize("levels", [1, 11])
def test_score_level_count_bounded(tmp_path: Path, levels: int) -> None:
    criteria = ", ".join(f'{{ label = "l{index}" }}' for index in range(levels))
    root = write_set(
        tmp_path, f'[[question]]\nname = "q"\ntext = "t"\nkind = "score"\ncriteria = [{criteria}]\n'
    )
    with pytest.raises(QuestionSetError):
        load_question_set("x", 2, root)


def test_noul_with_criteria_rejected(tmp_path: Path) -> None:
    root = write_set(
        tmp_path,
        '[[question]]\nname = "q"\ntext = "t"\nkind = "noul"\ncriteria = [{ label = "a" }]\n',
    )
    with pytest.raises(QuestionSetError):
        load_question_set("x", 2, root)


def test_duplicate_labels_rejected(tmp_path: Path) -> None:
    root = write_set(
        tmp_path,
        '[[question]]\nname = "q"\ntext = "t"\nkind = "choice"\ncriteria = [{ label = "a" }, { label = "a" }]\n',
    )
    with pytest.raises(QuestionSetError):
        load_question_set("x", 2, root)


def test_shipped_choice_and_score_options_are_described() -> None:
    for stage in ("plan", "gate", "coverage"):
        for question in load_question_set(stage, 2).build({}):
            if question.kind is QuestionKind.NOUL:
                assert question.options == ("yes", "no")
                continue
            assert len(question.descriptions) == len(question.options), question.name
