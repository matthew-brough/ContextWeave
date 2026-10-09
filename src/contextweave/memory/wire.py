from typing import Literal, NotRequired, TypedDict

from contextweave.core.errors import DegradedEntry, ErrorEntry

type RememberKind = Literal["decision", "code_area", "note"]
type RecallKind = Literal["decision", "code_area", "note", "turn_summary"]
type SummaryTier = Literal["extractive", "truncated", "synthesized"]
type RecallScope = Literal["project", "global"]
type RecallMode = Literal["event", "timeline", "topic"]
type PayloadState = Literal["present", "none", "aged_out"]

REMEMBER_KINDS: tuple[RememberKind, ...] = ("decision", "code_area", "note")
RECALL_KINDS: tuple[RecallKind, ...] = ("decision", "code_area", "note", "turn_summary")


class RecallHit(TypedDict):
    kind: RecallKind
    id: int
    project: str
    session_id: str | None
    text: str
    score: float
    created_at: int
    reason: NotRequired[str]
    path: NotRequired[str]
    prompt_number: NotRequired[int]
    tier: NotRequired[SummaryTier]
    synthesized: NotRequired[bool]
    model: NotRequired[str]
    prompt_version: NotRequired[int]


class TimelineEvent(TypedDict):
    id: int
    tool_name: str
    summary: str | None


class TimelineTurn(TypedDict):
    prompt_number: int
    summary: str
    tier: SummaryTier
    synthesized: bool
    events: list[TimelineEvent]
    model: NotRequired[str]
    prompt_version: NotRequired[int]


class Timeline(TypedDict):
    session_id: str
    project: str
    status: str
    started_at: int
    ended_at: int | None
    prompt_count: int
    turns: list[TimelineTurn]


class EventDetail(TypedDict):
    id: int
    project: str
    session_id: str
    prompt_number: int
    tool_name: str
    created_at: int
    summary: str | None
    payload: PayloadState
    raw_input: str | None
    raw_output: str | None
    truncated: bool


class RecallResult(TypedDict):
    mode: RecallMode
    scope: RecallScope
    hits: list[RecallHit]
    timeline: Timeline | None
    event: EventDetail | None
    errors: list[ErrorEntry]
    degraded: list[DegradedEntry]
    truncated: bool


class RememberResult(TypedDict):
    stored: bool
    kind: str
    id: int | None
    errors: list[ErrorEntry]
    degraded: list[DegradedEntry]


class SavingsBucketTotals(TypedDict):
    baseline: int
    served: int
    saved: int
    calls: int
