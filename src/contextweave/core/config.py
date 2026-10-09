import dataclasses
import hashlib
import json
import os
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast, get_origin, get_type_hints

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

CONFIG_INVALID = "config_invalid"
ENV_PREFIX = "CW_"
ENV_SEPARATOR = "__"
LIST_MERGE_KEYS = frozenset({"index.ignore", "edges.events.registrars", "edges.events.emitters"})

type Scalar = bool | int | float | str | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IndexConfig:
    ignore: tuple[str, ...] = ()
    test_globs: tuple[str, ...] = (
        "**/test_*.py",
        "**/*_test.go",
        "**/*.test.*",
        "**/*.spec.*",
        "tests/**",
    )
    vector_quant: str = "none"


@dataclass(frozen=True, slots=True)
class ChunkConfig:
    max_tokens: int = 800
    min_tokens: int = 40


@dataclass(frozen=True, slots=True)
class FuseConfig:
    keep: int = 100


@dataclass(frozen=True, slots=True)
class GateConfig:
    window: int = 30
    expand_window: int = 20
    card_tokens: int = 256
    critical: float = 0.8


@dataclass(frozen=True, slots=True)
class ExpandConfig:
    min_score: float = 0.6
    max_depth: int = 2


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    max_iterations: int = 2


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    default_max_tokens: int = 4000
    search_max_tokens: int = 6000
    omitted_max: int = 50


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    gate_answers_query: float = 0.40
    gate_relevance: float = 0.20
    gate_role_implementation: float = 0.15
    gate_lexical_match: float = 0.10
    gate_structural_value: float = 0.05
    gate_norm: float = 0.90
    final_gate: float = 0.90
    final_graph: float = 0.10
    final_redundancy: float = -0.20
    redundancy_floor: float = 0.85
    redundancy_span: float = 0.12


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    plan: bool = True
    retrieve: bool = True
    fuse: bool = True
    gate: bool = True
    expand: bool = True
    dedupe: bool = True
    compact: bool = True
    coverage: bool = True
    assemble: bool = True


@dataclass(frozen=True, slots=True)
class EmbedConfig:
    provider: str = "ollama"
    model: str = "jina-code-embeddings-1.5b"
    dim: int = 1536
    query_prefix: str = "Find the most relevant code snippet given the following query:\n"
    doc_prefix: str = "Candidate code snippet:\n"
    memory_prefix: str = ""
    query_device: str = "cpu"


@dataclass(frozen=True, slots=True)
class AnnotateConfig:
    provider: str = "ollama"
    model: str = "qwen3.5:4b"
    think: bool = False
    prompt_version: int = 1


@dataclass(frozen=True, slots=True)
class DeviceEndpointConfig:
    endpoint: str = ""
    launch: str = ""


def _gpu_endpoint() -> DeviceEndpointConfig:
    return DeviceEndpointConfig(endpoint="http://127.0.0.1:8765")


def _cpu_endpoint() -> DeviceEndpointConfig:
    return DeviceEndpointConfig(endpoint="http://127.0.0.1:8766")


@dataclass(frozen=True, slots=True)
class ClassifyConfig:
    provider: str = "kev"
    model: str = "jaredpalmer/kev-4b"
    devices: tuple[str, ...] = ("gpu", "cpu")
    question_set_version: int = 1
    gpu: DeviceEndpointConfig = field(default_factory=_gpu_endpoint)
    cpu: DeviceEndpointConfig = field(default_factory=_cpu_endpoint)


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    deadline_ms: int = 1500
    retry_after_s: int = 60
    skip_warn: float = 0.05
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    annotate: AnnotateConfig = field(default_factory=AnnotateConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)


@dataclass(frozen=True, slots=True)
class DaemonConfig:
    idle_minutes: int = 30
    project_idle_minutes: int = 30


@dataclass(frozen=True, slots=True)
class ServerConfig:
    pack_cache: int = 20


@dataclass(frozen=True, slots=True)
class HooksConfig:
    session_start_tokens: int = 400


@dataclass(frozen=True, slots=True)
class EventsConfig:
    registrars: tuple[str, ...] = (
        "RegisterNetEvent",
        "AddEventHandler",
        "RegisterServerEvent",
        "EventHandlers",
    )
    emitters: tuple[str, ...] = (
        "TriggerEvent",
        "TriggerServerEvent",
        "TriggerClientEvent",
        "TriggerLatentServerEvent",
        "TriggerLatentClientEvent",
    )


@dataclass(frozen=True, slots=True)
class EdgesConfig:
    events: EventsConfig = field(default_factory=EventsConfig)


@dataclass(frozen=True, slots=True)
class Config:
    index: IndexConfig = field(default_factory=IndexConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    fuse: FuseConfig = field(default_factory=FuseConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    expand: ExpandConfig = field(default_factory=ExpandConfig)
    coverage: CoverageConfig = field(default_factory=CoverageConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    daemon: DaemonConfig = field(default_factory=DaemonConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    hooks: HooksConfig = field(default_factory=HooksConfig)
    edges: EdgesConfig = field(default_factory=EdgesConfig)


class Layer(StrEnum):
    DEFAULT = "default"
    USER = "user"
    PROJECT = "project"
    LOCAL = "local"
    ENV = "env"
    CLI = "cli"


@dataclass(frozen=True, slots=True)
class Origin:
    layer: Layer
    source: str

    def __str__(self) -> str:
        return f"{self.layer}" if self.layer is Layer.DEFAULT else f"{self.layer} ({self.source})"


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    code: str
    key: str
    file: str
    message: str


class ConfigError(Exception):
    """Strict-mode config failure naming the offending key and file."""

    def __init__(self, issue: ConfigIssue) -> None:
        super().__init__(f"{issue.message}: key '{issue.key}' in {issue.file}")
        self.issue = issue


@dataclass(frozen=True, slots=True)
class LoadedConfig:
    config: Config
    origins: Mapping[str, Origin]
    issues: tuple[ConfigIssue, ...]


def _is_section(tp: object) -> bool:
    return isinstance(tp, type) and dataclasses.is_dataclass(tp)


def _collect_leaves(cls: type, prefix: str) -> tuple[dict[str, object], set[str]]:
    hints: dict[str, object] = get_type_hints(cls)
    leaves: dict[str, object] = {}
    sections: set[str] = set()
    for item in dataclasses.fields(cls):
        key = f"{prefix}{item.name}"
        tp = hints[item.name]
        if isinstance(tp, type) and _is_section(tp):
            sections.add(key)
            nested_leaves, nested_sections = _collect_leaves(tp, f"{key}.")
            leaves.update(nested_leaves)
            sections.update(nested_sections)
        else:
            leaves[key] = tp
    return leaves, sections


_LEAVES, _SECTIONS = _collect_leaves(Config, "")


def leaf_keys() -> tuple[str, ...]:
    """Every dotted leaf key of the config schema, in declaration order."""
    return tuple(_LEAVES)


def flatten(data: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    """Flatten nested mappings into dotted keys; non-mapping values are leaves."""
    flat: dict[str, object] = {}
    for name, value in data.items():
        key = f"{prefix}{name}"
        if isinstance(value, Mapping):
            nested = cast(Mapping[str, object], value)  # TOML tables always have str keys
            flat.update(flatten(nested, f"{key}."))
        else:
            flat[key] = value
    return flat


def unflatten(flat: Mapping[str, object]) -> dict[str, object]:
    """Inverse of flatten: dotted keys back into nested dicts."""
    root: dict[str, object] = {}
    for key, value in flat.items():
        *parents, leaf = key.split(".")
        node = root
        for part in parents:
            child = node.setdefault(part, {})
            node = cast(dict[str, object], child)  # only dicts are ever inserted by this function
        node[leaf] = value
    return root


def flatten_config(config: Config) -> dict[str, object]:
    """Dotted-key view of every effective value in a Config."""
    return {key: _read_leaf(config, key) for key in _LEAVES}


def _read_leaf(config: Config, key: str) -> object:
    node: object = config
    for part in key.split("."):
        node = getattr(node, part)
    return node


def config_hash(config: Config) -> str:
    """Stable sha256 hex digest of the effective config."""
    payload = json.dumps(flatten_config(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class _BadValue(Exception):
    pass


def _coerce(value: object, tp: object) -> Scalar:
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif tp is str:
        if isinstance(value, str):
            return value
    elif get_origin(tp) is tuple and isinstance(value, (list, tuple)):
        items = cast(Sequence[object], value)  # list/tuple of unknown element type, checked next
        strings = tuple(item for item in items if isinstance(item, str))
        if len(strings) == len(items):
            return strings
    raise _BadValue


def _parse_text(raw: str, tp: object) -> object:
    if tp is str:
        return raw
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw


@dataclass(frozen=True, slots=True)
class _LayerSource:
    layer: Layer
    source: str
    values: Mapping[str, object]
    parse_text: bool = False


def _validate_layer(
    source: _LayerSource,
) -> tuple[dict[str, Scalar], list[ConfigIssue]]:
    valid: dict[str, Scalar] = {}
    issues: list[ConfigIssue] = []
    for key, value in source.values.items():
        if key not in _LEAVES:
            message = "expected a scalar" if key in _SECTIONS else "unknown key"
            issues.append(ConfigIssue(CONFIG_INVALID, key, source.source, message))
            continue
        tp = _LEAVES[key]
        candidate = (
            _parse_text(value, tp) if source.parse_text and isinstance(value, str) else value
        )
        try:
            valid[key] = _coerce(candidate, tp)
        except _BadValue:
            issues.append(ConfigIssue(CONFIG_INVALID, key, source.source, "invalid value type"))
    return valid, issues


def _merge_lists(base: Scalar, addition: tuple[str, ...]) -> tuple[str, ...]:
    existing = base if isinstance(base, tuple) else ()
    return tuple(dict.fromkeys((*existing, *addition)))


def _default_values() -> dict[str, Scalar]:
    defaults = flatten_config(Config())
    return {key: _coerce(value, _LEAVES[key]) for key, value in defaults.items()}


def _read_file(path: Path, layer: Layer) -> tuple[_LayerSource | None, list[ConfigIssue]]:
    if not path.is_file():
        return None, []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        return None, [ConfigIssue(CONFIG_INVALID, "", str(path), f"unreadable config: {exc}")]
    return _LayerSource(layer, str(path), flatten(data)), []


def _env_values(env: Mapping[str, str]) -> dict[str, object]:
    values: dict[str, object] = {}
    for name, raw in env.items():
        if name.startswith(ENV_PREFIX) and ENV_SEPARATOR in name:
            key = name.removeprefix(ENV_PREFIX).lower().replace(ENV_SEPARATOR, ".")
            values[key] = raw
    return values


def user_config_path(env: Mapping[str, str]) -> Path:
    """Location of the user-level config file honouring XDG_CONFIG_HOME."""
    base = env.get("XDG_CONFIG_HOME")
    config_home = Path(base) if base else Path.home() / ".config"
    return config_home / "contextweave" / "config.toml"


def _build[T: DataclassInstance](cls: type[T], flat: Mapping[str, object], prefix: str = "") -> T:
    hints: dict[str, object] = get_type_hints(cls)
    changes: dict[str, object] = {}
    for item in dataclasses.fields(cls):
        key = f"{prefix}{item.name}"
        tp = hints[item.name]
        if isinstance(tp, type) and _is_section(tp):
            if any(name.startswith(f"{key}.") for name in flat):
                changes[item.name] = _build(tp, flat, f"{key}.")
        elif key in flat:
            changes[item.name] = flat[key]
    factory: Callable[..., T] = cls
    return factory(**changes)


def _file_layers(
    root: Path | None, env: Mapping[str, str]
) -> tuple[list[_LayerSource], list[ConfigIssue]]:
    candidates = [(user_config_path(env), Layer.USER)]
    if root is not None:
        candidates.append((root / ".contextweave.toml", Layer.PROJECT))
        candidates.append((root / ".contextweave" / "local.toml", Layer.LOCAL))
    sources: list[_LayerSource] = []
    issues: list[ConfigIssue] = []
    for path, layer in candidates:
        source, read_issues = _read_file(path, layer)
        issues.extend(read_issues)
        if source is not None:
            sources.append(source)
    return sources, issues


def load_config(
    root: Path | None,
    *,
    cli: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    strict: bool,
) -> LoadedConfig:
    """Resolve layered config; strict raises ConfigError, lenient collects issues.

    Args:
        root: project root for project/local layers, or None.
        cli: dotted-key overrides, highest precedence.
        env: environment mapping, defaults to os.environ.
        strict: raise on the first invalid key instead of dropping it.
    """
    environment = os.environ if env is None else env
    sources, issues = _file_layers(root, environment)
    if strict and issues:
        raise ConfigError(issues[0])
    sources.append(_LayerSource(Layer.ENV, "env", _env_values(environment), parse_text=True))
    sources.append(_LayerSource(Layer.CLI, "cli", dict(cli or {}), parse_text=True))

    effective = _default_values()
    origins = {key: Origin(Layer.DEFAULT, "built-in") for key in effective}
    for source in sources:
        valid, layer_issues = _validate_layer(source)
        if strict and layer_issues:
            raise ConfigError(layer_issues[0])
        issues.extend(layer_issues)
        for key, value in valid.items():
            if key in LIST_MERGE_KEYS and isinstance(value, tuple):
                effective[key] = _merge_lists(effective[key], value)
            else:
                effective[key] = value
            origins[key] = Origin(source.layer, source.source)
    return LoadedConfig(_build(Config, effective), origins, tuple(issues))
