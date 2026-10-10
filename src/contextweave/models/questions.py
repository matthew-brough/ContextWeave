import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from contextweave.core.registry import Registry, UnknownKeyError
from contextweave.core.types import JSON
from contextweave.models.decision import (
    CHOICE_OPTION_LIMIT,
    SCORE_LEVELS,
    Question,
    QuestionKind,
    kind_options,
)
from contextweave.models.errors import ModelError
from contextweave.models.wire import to_json

QUESTIONS_DIR = Path(__file__).parent / "questions"
OPTION_PLACEHOLDER = "{option}"

type OptionSource = Callable[[Sequence[str]], tuple[str, ...]]

OPTION_SOURCES: Registry[str, OptionSource] = Registry()
OPTION_SOURCES.register("scope_dirs", lambda values: tuple(values))


class QuestionSetError(ModelError):
    """A question set file is missing, malformed, or at a different version than requested."""

    code_value = "question_set"


@dataclass(frozen=True, slots=True)
class QuestionTemplate:
    name: str
    text: str
    kind: QuestionKind
    options: tuple[str, ...]
    descriptions: tuple[str, ...]
    options_from: str | None = None
    option_description: str = OPTION_PLACEHOLDER


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """A stage's versioned questions; dynamic options are filled per query."""

    stage: str
    version: int
    templates: tuple[QuestionTemplate, ...]

    def build(self, dynamic: Mapping[str, Sequence[str]] | None = None) -> tuple[Question, ...]:
        """Questions with option sources resolved against the query-time values."""
        values = dynamic or {}
        return tuple(_resolve(template, values) for template in self.templates)


def _resolve(template: QuestionTemplate, values: Mapping[str, Sequence[str]]) -> Question:
    options, descriptions = list(template.options), list(template.descriptions)
    if template.options_from is not None:
        try:
            source = OPTION_SOURCES.get(template.options_from)
        except UnknownKeyError as exc:
            raise QuestionSetError(f"unknown option source: {exc}") from exc
        for option in source(values.get(template.options_from, ())):
            if option not in options:
                options.append(option)
                descriptions.append(template.option_description.replace(OPTION_PLACEHOLDER, option))
    return Question(
        template.name,
        template.text,
        template.kind,
        tuple(options[:CHOICE_OPTION_LIMIT]),
        tuple(descriptions[:CHOICE_OPTION_LIMIT]),
    )


def _criteria(name: str, raw: JSON) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(raw, list):
        raise QuestionSetError(f"question '{name}' criteria must be a list")
    labels: list[str] = []
    descriptions: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise QuestionSetError(f"question '{name}' criteria entries must be tables")
        label, description = entry.get("label"), entry.get("description")
        if not (isinstance(label, str) and isinstance(description, str)):
            raise QuestionSetError(f"question '{name}' criteria need label and description")
        labels.append(label)
        descriptions.append(description)
    if len(set(labels)) != len(labels):
        raise QuestionSetError(f"question '{name}' has duplicate criteria labels")
    return tuple(labels), tuple(descriptions)


def _check_counts(name: str, kind: QuestionKind, count: int, dynamic: bool) -> None:
    match kind:
        case QuestionKind.NOUL:
            if count:
                raise QuestionSetError(f"noul question '{name}' takes no criteria")
        case QuestionKind.SCORE:
            low, high = SCORE_LEVELS
            if not low <= count <= high:
                raise QuestionSetError(f"score question '{name}' needs {low}-{high} levels")
        case QuestionKind.CHOICE:
            if count > CHOICE_OPTION_LIMIT or (count < 2 and not dynamic):
                raise QuestionSetError(
                    f"choice question '{name}' needs 2-{CHOICE_OPTION_LIMIT} options"
                )


def _template(raw: JSON) -> QuestionTemplate:
    if not isinstance(raw, dict):
        raise QuestionSetError("question entry is not a table")
    name, text, kind = raw.get("name"), raw.get("text"), raw.get("kind")
    source, described = raw.get("options_from"), raw.get("option_description", OPTION_PLACEHOLDER)
    if not (isinstance(name, str) and isinstance(text, str) and isinstance(kind, str)):
        raise QuestionSetError("question needs name, text and kind")
    if source is not None and not isinstance(source, str):
        raise QuestionSetError(f"question '{name}' options_from must be a string")
    if not isinstance(described, str):
        raise QuestionSetError(f"question '{name}' option_description must be a string")
    try:
        question_kind = QuestionKind(kind)
    except ValueError as exc:
        raise QuestionSetError(f"question '{name}' has unknown kind '{kind}'") from exc
    labels, descriptions = _criteria(name, raw.get("criteria", []))
    _check_counts(name, question_kind, len(labels), source is not None)
    if question_kind is QuestionKind.NOUL:
        return QuestionTemplate(name, text, question_kind, kind_options(question_kind, ()), ())
    return QuestionTemplate(name, text, question_kind, labels, descriptions, source, described)


@cache
def load_question_set(stage: str, version: int, directory: Path = QUESTIONS_DIR) -> QuestionSet:
    """Load questions/<stage>.toml, requiring question_set_version to equal version."""
    path = directory / f"{stage}.toml"
    try:
        raw = to_json(tomllib.loads(path.read_text(encoding="utf-8")))
    except (OSError, tomllib.TOMLDecodeError, ModelError) as exc:
        raise QuestionSetError(f"cannot load question set '{stage}': {exc}") from exc
    if not isinstance(raw, dict):
        raise QuestionSetError(f"question set '{stage}' is not a table")
    found = raw.get("question_set_version")
    if found != version:
        raise QuestionSetError(f"question set '{stage}' is version {found}, config wants {version}")
    entries = raw.get("question")
    if not isinstance(entries, list) or not entries:
        raise QuestionSetError(f"question set '{stage}' has no [[question]] entries")
    return QuestionSet(stage, version, tuple(_template(entry) for entry in entries))
