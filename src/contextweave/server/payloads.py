from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import Field, TypeAdapter

from contextweave.core.errors import DegradedEntry, ErrorEntry, OmittedEntry, OverflowEntry
from contextweave.core.project import ProjectEcho
from contextweave.core.types import JSON
from contextweave.index.records import FileProfile
from contextweave.lang.registry import Group
from contextweave.retrieve.context import Mode

type SymbolView = Literal["source", "signature", "doc", "annotation"]
type ExpandView = Literal["source", "signature", "doc"]
type OutlineMode = Literal["full", "names"]
type ExpandStatus = Literal["ok", "moved", "gone"]

MaxTokens = Annotated[int, Field(ge=1, description="Token ceiling for the whole response")]
ProjectArg = Annotated[str, Field(description="Project root path or 12-hex key")]


class BaseArgs(TypedDict):
    project: NotRequired[ProjectArg]
    max_tokens: NotRequired[MaxTokens]


class GetSymbolArgs(BaseArgs):
    name: Annotated[str, Field(min_length=1, description="Bare or qualified name, key path, slug")]
    path: NotRequired[Annotated[str, Field(description="Narrow to one file")]]
    line: NotRequired[Annotated[int, Field(ge=1, description="start_line of one candidate")]]
    view: NotRequired[SymbolView]


class FileOutlineArgs(BaseArgs):
    path: Annotated[str, Field(min_length=1, description="File path, project relative or absolute")]
    max_depth: NotRequired[Annotated[int, Field(ge=1)]]
    mode: NotRequired[OutlineMode]
    include: NotRequired[Annotated[str, Field(description="Comma list of docs, imports")]]


class SearchSymbolsArgs(BaseArgs):
    query: Annotated[str, Field(min_length=1, description="Case-insensitive substring")]
    kind: NotRequired[str]
    lang: NotRequired[str]
    group: NotRequired[Group]
    path_glob: NotRequired[str]
    limit: NotRequired[Annotated[int, Field(ge=1)]]


class AstQueryArgs(BaseArgs):
    path: Annotated[str, Field(min_length=1)]
    query: Annotated[str, Field(min_length=1, description="Tree-sitter S-expression")]
    captures: NotRequired[Annotated[list[str], Field(description="Capture names to return")]]


class ExpandArgs(BaseArgs):
    ref: Annotated[str, Field(min_length=1, description="cw:// ref from an earlier response")]
    view: NotRequired[ExpandView]


class SymbolItem(TypedDict):
    ref: str
    name: str
    qualified_name: str
    kind: str
    lang: str | None
    group: Group | None
    profile: FileProfile
    path: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    signature: NotRequired[str | None]
    docstring: NotRequired[str | None]
    source: NotRequired[str]
    annotation: NotRequired[None]


class ContextSearchArgs(BaseArgs):
    query: Annotated[str, Field(min_length=1, description="What you are looking for, in words")]
    scope: NotRequired[Annotated[str, Field(description="Path prefix or glob, overrides the plan")]]
    intent: NotRequired[Annotated[str, Field(description="Overrides the planner intent")]]
    mode: NotRequired[Mode]


class Candidate(TypedDict):
    ref: str
    qualified_name: str
    kind: str
    profile: FileProfile
    group: Group | None
    path: str
    start_line: int


class Hit(TypedDict):
    ref: str
    name: str
    qualified_name: str
    kind: str
    lang: str | None
    group: Group | None
    profile: FileProfile
    path: str
    start_line: int
    end_line: int
    signature: NotRequired[str | None]


class OutlineNode(TypedDict):
    ref: str
    qualified_name: str
    start_line: int
    kind: NotRequired[str]
    end_line: NotRequired[int]
    name: NotRequired[str]
    signature: NotRequired[str | None]
    docstring: NotRequired[str | None]
    level: NotRequired[int]
    children_count: NotRequired[int]
    uniformity: NotRequired[str]
    truncated_subtree: NotRequired[bool]
    children: NotRequired[list[OutlineNode]]


class ImportItem(TypedDict):
    module: str
    names: list[str]
    alias: str | None
    line: int
    kind: str


class Match(TypedDict):
    capture: str
    node_type: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    text: str


class Common(TypedDict):
    project: ProjectEcho | None
    errors: list[ErrorEntry]
    degraded: list[DegradedEntry]
    truncated: bool
    hint: NotRequired[str]
    omitted: NotRequired[list[OmittedEntry | OverflowEntry]]


class GetSymbolPayload(Common):
    found: bool
    ambiguous: bool
    symbols: list[SymbolItem]
    total_matches: NotRequired[int]
    candidates: NotRequired[list[Candidate]]


class FileOutlinePayload(Common):
    path: str
    lang: str | None
    group: Group | None
    profile: FileProfile | None
    line_count: int
    symbols: NotRequired[list[OutlineNode]]
    schema: NotRequired[list[OutlineNode]]
    outline: NotRequired[list[OutlineNode]]
    imports: NotRequired[list[ImportItem]]


class SearchSymbolsPayload(Common):
    hits: list[Hit]
    total_matches: int


class AstQueryPayload(Common):
    path: str
    lang: str | None
    group: Group | None
    profile: FileProfile | None
    matches: list[Match]


class ExpandPayload(Common):
    ref: str
    status: NotRequired[ExpandStatus]
    current: NotRequired[str]
    path: NotRequired[str]
    qualified_name: NotRequired[str]
    kind: NotRequired[str]
    profile: NotRequired[FileProfile]
    group: NotRequired[Group | None]
    start_line: NotRequired[int]
    end_line: NotRequired[int]
    start_byte: NotRequired[int]
    end_byte: NotRequired[int]
    signature: NotRequired[str | None]
    docstring: NotRequired[str | None]
    source: NotRequired[str]


def _strip_titles(node: JSON, in_properties: bool = False) -> JSON:
    if isinstance(node, dict):
        return {
            key: _strip_titles(value, key == "properties")
            for key, value in node.items()
            if in_properties or key != "title"
        }
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    return node


def json_schema[T](tp: type[T], *, mode: Literal["validation", "serialization"]) -> dict[str, JSON]:
    """JSON Schema of a TypedDict with pydantic's per-node titles removed."""
    stripped = _strip_titles(TypeAdapter(tp).json_schema(mode=mode))
    if not isinstance(stripped, dict):
        raise TypeError("schema root is not an object")
    return stripped


type Rel = Literal[
    "callers",
    "callees",
    "imports",
    "importers",
    "inherits",
    "implements",
    "tests",
    "config",
    "emitters",
    "listeners",
]


class RelatedArgs(BaseArgs):
    target: Annotated[str, Field(min_length=1, description="cw:// ref, qualified name or path")]
    rel: Rel
    depth: NotRequired[Annotated[int, Field(ge=1, le=4, description="Hops to follow")]]
    include_name_edges: NotRequired[bool]


class RelatedEdge(TypedDict):
    rel: str
    resolution: str
    hops: int
    name: str
    ref: NotRequired[str]
    path: NotRequired[str]


class RelatedPayload(Common):
    target: str
    found: bool
    ambiguous: bool
    edges: list[RelatedEdge]
    candidates: NotRequired[list[Candidate]]
