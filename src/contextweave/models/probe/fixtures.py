import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from contextweave.models.decision import Question, QuestionKind, kind_options
from contextweave.models.providers import EmbedKind

FIXTURE_PATH = Path(__file__).parent / "probe_set.toml"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CardPrefixes(Frozen):
    query: str
    doc: str


class Pair(Frozen):
    query: str
    code: str


class ChunkSample(Frozen):
    path: str
    kind: str
    name: str
    code: str


class StateSample(Frozen):
    query: str
    card: str


class QuestionSample(Frozen):
    name: str
    text: str
    kind: QuestionKind
    options: tuple[str, ...] = ()
    descriptions: tuple[str, ...] = ()

    def question(self) -> Question:
        return Question(
            self.name,
            self.text,
            self.kind,
            kind_options(self.kind, self.options),
            self.descriptions,
        )


class EmbedReference(Frozen):
    text: str
    kind: EmbedKind
    vector: tuple[float, ...]


class ProbeSet(Frozen):
    card_prefixes: CardPrefixes
    pairs: tuple[Pair, ...]
    chunks: tuple[ChunkSample, ...]
    state: StateSample
    gate_questions: tuple[QuestionSample, ...]
    score_question: QuestionSample
    dynamic_question: QuestionSample
    embed_references: tuple[EmbedReference, ...] = ()


def load_probe_set(path: Path = FIXTURE_PATH) -> ProbeSet:
    """Load and validate the bundled probe samples."""
    return ProbeSet.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
