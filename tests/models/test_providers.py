import math
from dataclasses import replace

import httpx
import pytest
from models_support import Recorder, recorder, transport_of

from contextweave.core.config import ModelsConfig
from contextweave.core.types import JSON
from contextweave.models.decision import (
    NullDecisionClient,
    Question,
    QuestionKind,
    ensure_decision_client,
)
from contextweave.models.errors import (
    ModelDeadlineError,
    ModelDisabledError,
    ModelParseError,
    ModelTransportError,
    ProviderSlotError,
)
from contextweave.models.llama_server import LlamaAnnotate, LlamaEmbed
from contextweave.models.null import NullAnnotate, NullEmbed
from contextweave.models.openai_compat import OpenAIAnnotate, OpenAIEmbed
from contextweave.models.prompt_store import PromptError, load_prompt
from contextweave.models.providers import (
    BuildContext,
    Device,
    EmbedKind,
    Endpoints,
    Message,
    Supervised,
)

SCHEMA: dict[str, JSON] = {"type": "object", "required": ["purpose", "tags"]}
MESSAGES = [Message("system", "s"), Message("user", "u")]


def ok(payload: object) -> httpx.Response:
    return httpx.Response(200, json=payload)


def ctx_of(models: ModelsConfig, rec: Recorder) -> BuildContext:
    return BuildContext(models, Endpoints(openai_api_key="k"), transport_of(rec))


def embeddings(*rows: list[float]) -> dict[str, object]:
    return {"data": [{"index": index, "embedding": row} for index, row in enumerate(rows)]}


def test_llama_embed_request_prefix_and_truncation(models: ModelsConfig) -> None:
    rec = recorder(lambda r: ok(embeddings([3.0, 4.0, 12.0, 99.0])))
    client = LlamaEmbed(ctx_of(models, rec))
    out = client.embed(["x"], kind=EmbedKind.QUERY)
    body = rec.bodies()[0]
    assert [str(request.url) for request in rec.requests] == ["http://127.0.0.1:8770/v1/embeddings"]
    assert body == {"model": models.embed.model, "input": ["Q:x"]}
    assert len(out[0]) == 3
    assert math.isclose(sum(v * v for v in out[0]), 1.0)


def test_embed_kinds_use_their_prefix(models: ModelsConfig) -> None:
    rec = recorder(lambda r: ok(embeddings([1.0, 0.0, 0.0])))
    client = LlamaEmbed(ctx_of(models, rec))
    client.embed(["a"], kind=EmbedKind.DOC)
    client.embed(["a"], kind=EmbedKind.MEMORY)
    assert [b["input"] for b in rec.bodies()] == [["D:a"], ["M:a"]]


def test_embed_short_vector_is_error(models: ModelsConfig) -> None:
    rec = recorder(lambda r: ok(embeddings([1.0, 0.0])))
    with pytest.raises(Exception, match="below configured"):
        LlamaEmbed(ctx_of(models, rec)).embed(["a"], kind=EmbedKind.DOC)


def test_llama_chat_request_shape(models: ModelsConfig) -> None:
    payload = {"choices": [{"message": {"content": '{"purpose": "p", "tags": []}'}}]}
    rec = recorder(lambda r: ok(payload))
    result = LlamaAnnotate(ctx_of(models, rec)).annotate(MESSAGES, SCHEMA)
    body = rec.bodies()[0]
    assert rec.paths() == ["/v1/chat/completions"]
    assert rec.requests[0].url.port == 8772
    assert result == {"purpose": "p", "tags": []}
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "annotation", "strict": True, "schema": SCHEMA},
    }
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert (body["temperature"], body["seed"]) == (0, 7)
    assert body["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


@pytest.mark.parametrize("content", ["not json", "[1]", '{"purpose": "p"}'])
def test_annotate_parse_failure_is_typed(models: ModelsConfig, content: str) -> None:
    rec = recorder(lambda r: ok({"choices": [{"message": {"content": content}}]}))
    with pytest.raises(ModelParseError):
        LlamaAnnotate(ctx_of(models, rec)).annotate(MESSAGES, SCHEMA)


def test_llama_set_device_switches_server_without_requests(models: ModelsConfig) -> None:
    rec = recorder(lambda r: ok(embeddings([1.0, 0.0, 0.0])))
    client = LlamaEmbed(ctx_of(models, rec))
    client.set_device(Device.CPU)
    assert rec.requests == []
    client.embed(["a"], kind=EmbedKind.DOC)
    assert rec.requests[0].url.port == 8771
    assert client.device is Device.CPU


def test_llama_servers_follow_configured_devices(models: ModelsConfig) -> None:
    cpu_only = replace(models, embed=replace(models.embed, devices=("cpu",)))
    client = LlamaEmbed(ctx_of(cpu_only, recorder(lambda r: ok({}))))
    assert client.devices == (Device.CPU,)
    assert set(client.servers) == {Device.CPU}
    assert isinstance(client, Supervised)


def test_llama_rejects_remote_device(models: ModelsConfig) -> None:
    rec = recorder(lambda r: ok({}))
    with pytest.raises(ProviderSlotError):
        LlamaEmbed(ctx_of(models, rec)).set_device(Device.REMOTE)


def test_annotate_uses_annotate_deadline(models: ModelsConfig) -> None:
    payload = {"choices": [{"message": {"content": '{"purpose": "p", "tags": []}'}}]}
    rec = recorder(lambda r: ok(payload))
    LlamaAnnotate(ctx_of(models, rec)).annotate(MESSAGES, SCHEMA)
    assert rec.requests[0].extensions["timeout"]["read"] == models.annotate.deadline_ms / 1000


def test_openai_embed_shape_and_ordering(models: ModelsConfig) -> None:
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0, 0.0]},
            {"index": 0, "embedding": [1.0, 0.0, 0.0]},
        ]
    }
    rec = recorder(lambda r: ok(payload))
    out = OpenAIEmbed(ctx_of(models, rec)).embed(["a", "b"], kind=EmbedKind.DOC)
    assert rec.paths() == ["/v1/embeddings"]
    assert rec.requests[0].headers["authorization"] == "Bearer k"
    assert rec.bodies()[0]["input"] == ["D:a", "D:b"]
    assert out == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_openai_chat_shape(models: ModelsConfig) -> None:
    payload = {"choices": [{"message": {"content": '{"purpose": "p", "tags": ["a"]}'}}]}
    rec = recorder(lambda r: ok(payload))
    client = OpenAIAnnotate(ctx_of(models, rec))
    assert client.annotate(MESSAGES, SCHEMA) == {"purpose": "p", "tags": ["a"]}
    body = rec.bodies()[0]
    assert rec.paths() == ["/v1/chat/completions"]
    assert body["temperature"] == 0
    assert body["seed"] == 7
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "annotation", "strict": True, "schema": SCHEMA},
    }
    assert client.devices == (Device.REMOTE,)


def test_timeout_maps_to_deadline_error(models: ModelsConfig) -> None:
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModelDeadlineError):
        LlamaEmbed(ctx_of(models, recorder(boom))).embed(["a"], kind=EmbedKind.DOC)


def test_deadline_is_http_timeout(models: ModelsConfig) -> None:
    rec = recorder(lambda r: ok(embeddings([1.0, 0.0, 0.0])))
    LlamaEmbed(ctx_of(models, rec)).embed(["a"], kind=EmbedKind.DOC)
    assert rec.requests[0].extensions["timeout"]["read"] == models.deadline_ms / 1000


def test_error_status_is_transport_error(models: ModelsConfig) -> None:
    rec = recorder(lambda r: httpx.Response(500))
    with pytest.raises(ModelTransportError):
        LlamaEmbed(ctx_of(models, rec)).embed(["a"], kind=EmbedKind.DOC)


def test_null_providers_raise_disabled() -> None:
    with pytest.raises(ModelDisabledError):
        NullEmbed().embed(["a"], kind=EmbedKind.DOC)
    with pytest.raises(ModelDisabledError):
        NullAnnotate().annotate(MESSAGES, SCHEMA)
    with pytest.raises(ModelDisabledError):
        NullDecisionClient().decide(["s"], [Question("q", "t", QuestionKind.NOUL, ("yes", "no"))])


def test_classify_rejects_chat_client(models: ModelsConfig) -> None:
    chat = LlamaAnnotate(ctx_of(models, recorder(lambda r: ok({}))))
    with pytest.raises(ProviderSlotError):
        ensure_decision_client(chat)
    assert isinstance(ensure_decision_client(NullDecisionClient()), NullDecisionClient)


def test_annotate_prompt_loads_and_renders() -> None:
    prompt = load_prompt("annotate", 1)
    messages = prompt.render(path="a.py", kind="function", name="f", code="x = {1}")
    assert prompt.version == 1
    assert "x = {1}" in messages[-1].content
    assert prompt.schema["required"] == ["purpose", "tags"]
    with pytest.raises(PromptError):
        load_prompt("annotate", 2)
