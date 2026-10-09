from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest
from models_support import recorder, transport_of

from contextweave.core.config import ClassifyConfig, DeviceEndpointConfig, ModelsConfig
from contextweave.core.types import JSON
from contextweave.models.clef import build_clef
from contextweave.models.decision import Question, QuestionKind, ensure_decision_client
from contextweave.models.errors import ModelResponseError
from contextweave.models.kev import build_kev
from contextweave.models.providers import BuildContext, Endpoints
from contextweave.models.questions import QuestionSetError, load_question_set
from contextweave.models.slots import default_registries


def test_registries_hold_both_providers() -> None:
    classify = default_registries().classify
    assert classify.get("kev") is build_kev
    assert classify.get("clef") is build_clef


@pytest.mark.parametrize("stage", ["plan", "gate", "coverage"])
def test_shipped_sets_load(stage: str) -> None:
    assert load_question_set(stage, 1).build({"scope_dirs": ["src"]})


def test_version_mismatch_raises() -> None:
    with pytest.raises(QuestionSetError, match="version"):
        load_question_set("gate", 2)


def test_missing_set_raises() -> None:
    with pytest.raises(QuestionSetError):
        load_question_set("nope", 1)


def test_scope_options_come_from_dynamic_values() -> None:
    questions = {
        q.name: q for q in load_question_set("plan", 1).build({"scope_dirs": ["src", "tests"]})
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
        for question in load_question_set(stage, 1).build({})
    ]
    code = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))
    assert not [text for text in texts if text in code]


def models_for(provider: str) -> ModelsConfig:
    endpoint = DeviceEndpointConfig(endpoint="http://gpu.test")
    return ModelsConfig(classify=ClassifyConfig(provider=provider, devices=("gpu",), gpu=endpoint))


def client_for(provider: str, reply: Mapping[str, JSON]) -> tuple[object, list[httpx.Request]]:
    rec = recorder(lambda request: httpx.Response(200, json=reply))
    ctx = BuildContext(models_for(provider), Endpoints(), transport_of(rec))
    build = build_kev if provider == "kev" else build_clef
    return build(ctx), rec.requests


SCORE = Question("relevance", "r", QuestionKind.SCORE, ("0", "0.25", "0.5", "0.75", "1"))
BINARY = Question("answers_query", "a", QuestionKind.BINARY, ("yes", "no"))


def test_kev_sends_everything_as_choice_and_reads_probs() -> None:
    reply: dict[str, JSON] = {
        "results": [
            [
                {
                    "name": "relevance",
                    "probs": {"0": 0.0, "0.25": 0.0, "0.5": 0.0, "0.75": 0.0, "1": 1.0},
                },
                {"name": "answers_query", "probs": {"yes": 0.9, "no": 0.1}},
            ]
        ]
    }
    client, requests = client_for("kev", reply)
    decision = ensure_decision_client(client)
    rows = decision.decide(["s"], [SCORE, BINARY])
    assert rows[0][0].top == "1"
    assert rows[0][1].probs["yes"] == pytest.approx(0.9)
    sent = requests[0]
    assert sent.url.path == "/v1/decide"
    assert b'"type":"choice"' in sent.content.replace(b" ", b"")
    assert b"binary" not in sent.content


def test_clef_reads_native_score_value() -> None:
    reply: dict[str, JSON] = {"results": [[{"name": "relevance", "value": 0.5}]]}
    client, requests = client_for("clef", reply)
    rows = ensure_decision_client(client).decide(["s"], [SCORE])
    assert rows[0][0].probs["0.5"] == pytest.approx(1.0)
    assert requests[0].url.path == "/classify"
    assert b'"type":"score"' in requests[0].content.replace(b" ", b"")


def test_wrong_row_count_is_a_response_error() -> None:
    client, _ = client_for("clef", {"results": []})
    with pytest.raises(ModelResponseError):
        ensure_decision_client(client).decide(["s"], [BINARY])


def test_out_of_order_answer_is_a_response_error() -> None:
    client, _ = client_for("kev", {"results": [[{"name": "other", "probs": {"yes": 1, "no": 0}}]]})
    with pytest.raises(ModelResponseError):
        ensure_decision_client(client).decide(["s"], [BINARY])
