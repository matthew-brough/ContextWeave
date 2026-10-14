import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Self

from contextweave.agents.jsonio import JSONObject, parse_object
from contextweave.agents.transport import HookTransport, InprocTransport
from contextweave.core.config import HooksConfig, LoadedConfig, load_config
from contextweave.core.paths import user_state_dir
from contextweave.core.project import ResolvedProject, resolve_project
from contextweave.core.types import JSON
from contextweave.index.projects import ProjectRegistry

PAYLOAD_CAP_CHARS = 16_384
SUMMARY_CHARS = 160
HINT_KEYS = ("file_path", "path", "command", "pattern", "query", "url")
NO_OUTPUT = b""


class HookEvent(StrEnum):
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SESSION_END = "SessionEnd"


@dataclass(frozen=True, slots=True)
class HookLimits:
    """Wall-clock ceilings in seconds (V17); SessionStart waits less than the rest."""

    total_s: float
    session_start_s: float

    @classmethod
    def from_config(cls, hooks: HooksConfig) -> Self:
        return cls(hooks.deadline_ms / 1000, hooks.session_start_deadline_ms / 1000)

    def for_event(self, event: HookEvent) -> float:
        return self.session_start_s if event is HookEvent.SESSION_START else self.total_s


@dataclass(frozen=True, slots=True)
class HookPayload:
    """The fields of Claude Code's hook JSON this adapter uses."""

    session_id: str
    cwd: Path
    prompt: str | None
    tool_name: str | None
    tool_input: JSON
    tool_response: JSON
    reason: str | None


def parse_event(name: str) -> HookEvent | None:
    """Match an event name ignoring case, dashes and underscores."""
    folded = name.replace("-", "").replace("_", "").lower()
    for event in HookEvent:
        if event.value.lower() == folded:
            return event
    return None


def _string(data: JSONObject, key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def parse_payload(stdin: bytes) -> HookPayload | None:
    """Decode the hook payload; None when it lacks a session id or cwd."""
    data = parse_object(stdin.decode("utf-8"), Path("<stdin>"))
    session_id, cwd = _string(data, "session_id"), _string(data, "cwd")
    if session_id is None or cwd is None:
        return None
    return HookPayload(
        session_id=session_id,
        cwd=Path(cwd),
        prompt=_string(data, "prompt"),
        tool_name=_string(data, "tool_name"),
        tool_input=data.get("tool_input"),
        tool_response=data.get("tool_response"),
        reason=_string(data, "reason"),
    )


def _render(value: JSON) -> str:
    text = (
        value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    )
    return text[:PAYLOAD_CAP_CHARS]


def _summary(tool_name: str, tool_input: JSON) -> str:
    if isinstance(tool_input, dict):
        for key in HINT_KEYS:
            hint = tool_input.get(key)
            if isinstance(hint, str) and hint:
                return f"{tool_name}: {' '.join(hint.split())}"[:SUMMARY_CHARS]
    return tool_name


def _resolve(payload: HookPayload, env: Mapping[str, str]) -> ResolvedProject | None:
    known = ProjectRegistry(user_state_dir(env)).known_roots()
    resolution = resolve_project({}, [], payload.cwd, known)
    return resolution if isinstance(resolution, ResolvedProject) else None


def _session_start_output(context: str) -> bytes:
    body = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": context}}
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _Prepared:
    payload: HookPayload
    project: ResolvedProject
    loaded: LoadedConfig


def _prepare(stdin: bytes, env: Mapping[str, str]) -> _Prepared | None:
    payload = parse_payload(stdin)
    if payload is None:
        return None
    project = _resolve(payload, env)
    if project is None:
        return None
    return _Prepared(payload, project, load_config(project.root, strict=False, env=env))


def _handle(event: HookEvent, prepared: _Prepared, transport: HookTransport) -> bytes:
    payload, project = prepared.payload, prepared.project
    match event:
        case HookEvent.SESSION_START:
            context = transport.session_start(project, payload.session_id, prepared.loaded)
            return _session_start_output(context) if context else NO_OUTPUT
        case HookEvent.USER_PROMPT_SUBMIT:
            if payload.prompt is not None:
                transport.prompt(project, payload.session_id, payload.prompt)
        case HookEvent.POST_TOOL_USE:
            if payload.tool_name is not None:
                transport.tool_event(
                    project,
                    payload.session_id,
                    payload.tool_name,
                    _summary(payload.tool_name, payload.tool_input),
                    _render(payload.tool_input),
                    _render(payload.tool_response),
                )
        case HookEvent.STOP:
            transport.stop(project, payload.session_id)
        case HookEvent.SESSION_END:
            transport.session_end(project, payload.session_id, payload.reason)
    return NO_OUTPUT


def _within[T](limit_s: float, work: Callable[[], T]) -> T | None:
    box: list[T] = []

    def target() -> None:
        try:
            box.append(work())
        except Exception:
            return

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(max(limit_s, 0.0))
    return box[0] if box and not thread.is_alive() else None


def run_hook(
    event: str,
    stdin: bytes,
    *,
    env: Mapping[str, str],
    transport: HookTransport | None = None,
    connect: Callable[[HooksConfig], HookTransport] | None = None,
    limits: HookLimits | None = None,
) -> tuple[bytes, int]:
    """Run one hook fail-open: stdout bytes plus exit code 0, whatever happens (V17).

    Ceilings come from the project's hooks config unless limits overrides them; connect
    builds the transport from that config when no transport is given.
    """
    parsed = parse_event(event)
    if parsed is None:
        return NO_OUTPUT, 0
    started = time.monotonic()
    fallback = limits or HookLimits.from_config(HooksConfig())
    prepared = _within(fallback.for_event(parsed), lambda: _prepare(stdin, env))
    if prepared is None:
        return NO_OUTPUT, 0
    hooks = prepared.loaded.config.hooks
    ceiling = (limits or HookLimits.from_config(hooks)).for_event(parsed)

    def work() -> bytes:
        if transport is not None:
            backend = transport
        else:
            backend = connect(hooks) if connect is not None else InprocTransport(env)
        return _handle(parsed, prepared, backend)

    output = _within(ceiling - (time.monotonic() - started), work)
    return output or NO_OUTPUT, 0
