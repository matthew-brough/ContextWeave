import asyncio
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from contextweave.cli_commands.indexing import full_index
from contextweave.core.budget import bytes35, serialize_structured
from contextweave.core.config import load_config
from contextweave.core.paths import user_state_dir
from contextweave.daemon.rebuild import build_shadow, swap_shadow
from contextweave.eval.adapters import Adapter, AdapterContext, ScoreSample, Served, degraded
from contextweave.eval.jsonvalue import JSON
from contextweave.eval.keywords import keywords
from contextweave.index.indexer import index_project
from contextweave.index.projects import ProjectRegistry
from contextweave.index.store import SchemaMismatchError
from contextweave.models.calibration import CALIBRATION_FILE_ENV, calibration_path

AST_MCP_COMMAND = "ast-mcp"
CW_COMMAND = "cw"
DISABLE_STAGE_ENV = "CW_EVAL_DISABLE_STAGE"
STRUCTURED_ENV = "CW_EVAL_STRUCTURED"
PROFILE_ENV = "CW_EVAL_PROFILE"
FULL_PROFILE = "full"
READONLY_FLAG = "--readonly"
CLASSIFY_STAGES = frozenset({"plan", "gate", "expand", "coverage"})
GATE_QUESTIONS = ("answers_query", "relevance", "structural_value")
SUFFICIENT_QUESTION = "sufficient"
OMITTED_KEY = "omitted"
SYMBOLS_KEY = "symbols"
STATE_HOME_ENV = "XDG_STATE_HOME"
CALL_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class ToolTarget:
    """How to launch an MCP server, which tool answers a query, and whether agents see only its text."""

    command: str
    args: tuple[str, ...]
    tool: str
    query_args: Mapping[str, JSON]
    indexed: bool = False
    text_only: bool = False
    per_keyword: bool = False


def _walk(value: JSON, found: list[dict[str, JSON]]) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("qualified_name"), str) or isinstance(value.get(SYMBOLS_KEY), list):
            found.append(value)
        for key, child in value.items():
            if key != OMITTED_KEY:
                _walk(child, found)
    elif isinstance(value, list):
        for child in value:
            _walk(child, found)


def extract_items(payload: JSON) -> tuple[str, ...]:
    """Qualified names of every served symbol entry, in order, deduplicated; omitted entries are skipped."""
    entries: list[dict[str, JSON]] = []
    _walk(payload, entries)
    names: list[JSON] = []
    for entry in entries:
        members = entry.get(SYMBOLS_KEY)
        names.extend([entry.get("qualified_name"), *(members if isinstance(members, list) else [])])
    return tuple(dict.fromkeys(name for name in names if isinstance(name, str)))


def extract_covered(payload: JSON) -> bool | None:
    """Whether the pack reports coverage.sufficient, or None when the payload reports none."""
    if not isinstance(payload, dict):
        return None
    coverage = payload.get("coverage")
    sufficient = coverage.get("sufficient") if isinstance(coverage, dict) else None
    return sufficient if isinstance(sufficient, bool) else None


def extract_model_calls(payload: JSON) -> int:
    """The reported model call count, or zero."""
    if not isinstance(payload, dict):
        return 0
    calls = payload.get("model_calls")
    return calls if isinstance(calls, int) and not isinstance(calls, bool) else 0


def _number(value: JSON) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def classify_active(payload: JSON) -> bool:
    """True when the pack was scored by a live classifier: a device is declared and no classify stage skipped."""
    if not isinstance(payload, dict):
        return False
    plan = payload.get("plan")
    devices = plan.get("devices") if isinstance(plan, dict) else None
    device = devices.get("classify") if isinstance(devices, dict) else None
    return isinstance(device, str) and device != "none" and count_skips(payload) == 0


def count_skips(payload: JSON) -> int:
    """Degraded entries raised by classify-backed stages."""
    entries = payload.get("degraded") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return 0
    return sum(
        1 for entry in entries if isinstance(entry, dict) and entry.get("stage") in CLASSIFY_STAGES
    )


def extract_samples(payload: JSON, expected: Sequence[str]) -> tuple[ScoreSample, ...]:
    """Per-item gate probabilities and the pack's sufficiency, labelled by membership in expected."""
    if not classify_active(payload) or not isinstance(payload, dict):
        return ()
    wanted = set(expected)
    entries: list[dict[str, JSON]] = []
    _walk(payload, entries)
    samples: list[ScoreSample] = []
    served: set[str] = set()
    for item in entries:
        scores = item.get("scores")
        name = item.get("qualified_name")
        if not isinstance(scores, dict) or not isinstance(name, str):
            continue
        served.add(name)
        for question in GATE_QUESTIONS:
            prob = _number(scores.get(question))
            if prob is not None:
                samples.append(ScoreSample(question, prob, 1.0 if name in wanted else 0.0))
    coverage = payload.get("coverage")
    score = _number(coverage.get("score")) if isinstance(coverage, dict) else None
    if score is not None and wanted:
        samples.append(ScoreSample(SUFFICIENT_QUESTION, score, 1.0 if wanted <= served else 0.0))
    return tuple(samples)


def _payload(structured: Mapping[str, object] | None, text: str) -> JSON:
    if structured is not None:
        return json.loads(json.dumps(structured))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def _query(
    session: ClientSession, target: ToolTarget, query: str, expected: Sequence[str]
) -> Served:
    started = time.perf_counter()
    arguments: dict[str, JSON] = {**target.query_args, "query": query}
    result = await session.call_tool(
        target.tool, arguments, read_timeout_seconds=CALL_TIMEOUT_SECONDS
    )
    elapsed = (time.perf_counter() - started) * 1000
    text = "\n".join(block.text for block in result.content if block.type == "text")
    structured = result.structured_content
    payload = _payload(structured, text)
    shown = text if target.text_only or structured is None else serialize_structured(structured)
    tokens = bytes35(shown)
    error = (text or "tool error") if result.is_error else None
    return Served(
        extract_items(payload),
        tokens,
        elapsed,
        extract_model_calls(payload),
        extract_covered(payload),
        error,
        extract_samples(payload, expected),
        count_skips(payload),
    )


def _state_home(root: Path) -> Path:
    return root.parent / f"{root.name}.state"


def prepare_checkout(
    root: Path, env: Mapping[str, str], profile: str = "nomodel"
) -> Mapping[str, str]:
    """Index the checkout and register it in a state dir private to the eval; returns the env to serve with.

    The full profile also annotates and embeds, so semantic retrieval has vectors to search.
    """
    isolated = {**os.environ, **env, STATE_HOME_ENV: str(_state_home(root))}
    config = load_config(root, strict=False, env=isolated).config
    try:
        index_project(root, config)
    except SchemaMismatchError:
        build_shadow(root, config)
        swap_shadow(root)
    if profile == FULL_PROFILE:
        full_index(root, config, isolated)
    ProjectRegistry(user_state_dir(isolated)).register(root)
    return {**env, STATE_HOME_ENV: isolated[STATE_HOME_ENV]}


async def _drive(
    target: ToolTarget, context: AdapterContext, queries: Sequence[str]
) -> tuple[Served, ...]:
    base = (
        prepare_checkout(context.root, context.env, context.profile)
        if target.indexed and not context.live
        else context.env
    )
    env = {
        **base,
        PROFILE_ENV: context.profile,
        CALIBRATION_FILE_ENV: str(calibration_path(context.env)),
    }
    if context.ablate is not None:
        env[DISABLE_STAGE_ENV] = context.ablate
    if target.text_only:
        env[STRUCTURED_ENV] = "1"
    live_index = target.indexed and context.live
    args = [*target.args, READONLY_FLAG] if live_index else list(target.args)
    params = StdioServerParameters(command=target.command, args=args, env=env, cwd=context.root)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        if target.tool not in tools:
            return degraded(len(queries), f"tool missing: {target.tool}")
        return tuple(
            [
                await _query_target(session, target, query, context.expected.get(query, ()))
                for query in queries
            ]
        )


async def _query_target(
    session: ClientSession, target: ToolTarget, query: str, expected: Sequence[str]
) -> Served:
    words = keywords(query) if target.per_keyword else ()
    if not words:
        return await _query(session, target, query, expected)
    parts = [await _query(session, target, word, expected) for word in words]
    return merge_served(parts)


def merge_served(parts: Sequence[Served]) -> Served:
    """One result for several tool calls answering the same query."""
    items = tuple(dict.fromkeys(item for part in parts for item in part.items))
    errors = [part.error for part in parts if part.error is not None]
    return Served(
        items,
        sum(part.tokens for part in parts),
        sum(part.latency_ms for part in parts),
        sum(part.model_calls for part in parts),
        None,
        errors[0] if errors else None,
        tuple(sample for part in parts for sample in part.samples),
        sum(part.skips for part in parts),
    )


class McpAdapter:
    """Drives one MCP stdio server; missing command, tool or crash yields degraded results."""

    def __init__(self, target: ToolTarget, context: AdapterContext) -> None:
        self._target = target
        self._context = context

    def serve_all(self, queries: Sequence[str]) -> tuple[Served, ...]:
        if shutil.which(self._target.command) is None:
            return degraded(len(queries), f"command not found: {self._target.command}")
        try:
            return asyncio.run(_drive(self._target, self._context, queries))
        except Exception as exc:
            return degraded(len(queries), f"{type(exc).__name__}: {exc}")


def ast_mcp_adapter(context: AdapterContext) -> Adapter:
    """ast-mcp search_symbols over stdio, one call per query keyword as an agent would name them."""
    target = ToolTarget(AST_MCP_COMMAND, ("serve",), "search_symbols", {}, per_keyword=True)
    return McpAdapter(target, context)


def cw_adapter(context: AdapterContext) -> Adapter:
    """cw serve --inproc context_search over stdio."""
    target = ToolTarget(
        CW_COMMAND, ("serve", "--inproc"), "context_search", {}, indexed=True, text_only=True
    )
    return McpAdapter(target, context)
