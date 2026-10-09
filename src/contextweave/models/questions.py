import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from contextweave.core.registry import Registry, UnknownKeyError
from contextweave.core.types import JSON
from contextweave.models.decision import Question, QuestionKind, kind_options
from contextweave.models.errors import ModelError
from contextweave.models.wire import to_json

QUESTIONS_DIR = Path(__file__).parent / "questions"
SCOPE_DIR_LIMIT = 30

type OptionSource = Callable[[Sequence[str]], tuple[str, ...]]

OPTION_SOURCES: Registry[str, OptionSource] = Registry()
OPTION_SOURCES.register("scope_dirs", lambda values: tuple(values[:SCOPE_DIR_LIMIT]))


class QuestionSetError(ModelError):
    """A question set file is missing, malformed, or at a different version than requested."""

    code_value = "question_set"


@dataclass(frozen=True, slots=True)
class QuestionTemplate:
    name: str
    text: str
    kind: QuestionKind
    options: tuple[str, ...]
    options_from: str | None


@dataclass(frozen=True, slots=True)
class QuestionSet:
    """A stage's versioned questions; dynamic options are filled per query."""

    stage: str
    version: int
    templates: tuple[QuestionTemplate, ...]

    def build(self, dynamic: Mapping[str, Sequence[str]] | None = None) -> tuple[Question, ...]:
        """Questions with option sources resolved against the query-time values."""
        values = dynamic or {}
        questions: list[Question] = []
        for template in self.templates:
            options = template.options
            if template.options_from is not None:
                try:
                    source = OPTION_SOURCES.get(template.options_from)
                except UnknownKeyError as exc:
                    raise QuestionSetError(f"unknown option source: {exc}") from exc
                options = (*options, *source(values.get(template.options_from, ())))
            questions.append(Question(template.name, template.text, template.kind, options))
        return tuple(questions)


def _template(raw: JSON) -> QuestionTemplate:
    if not isinstance(raw, dict):
        raise QuestionSetError("question entry is not a table")
    name, text, kind = raw.get("name"), raw.get("text"), raw.get("kind")
    options, source = raw.get("options", []), raw.get("options_from")
    if not (isinstance(name, str) and isinstance(text, str) and isinstance(kind, str)):
        raise QuestionSetError("question needs name, text and kind")
    if not isinstance(options, list):
        raise QuestionSetError(f"question '{name}' options must be a list")
    declared = tuple(option for option in options if isinstance(option, str))
    if len(declared) != len(options):
        raise QuestionSetError(f"question '{name}' options must be strings")
    if source is not None and not isinstance(source, str):
        raise QuestionSetError(f"question '{name}' options_from must be a string")
    try:
        question_kind = QuestionKind(kind)
    except ValueError as exc:
        raise QuestionSetError(f"question '{name}' has unknown kind '{kind}'") from exc
    return QuestionTemplate(
        name, text, question_kind, kind_options(question_kind, declared), source
    )


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
