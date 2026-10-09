import re
from dataclasses import dataclass

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "not",
    ]
)
MAX_TERMS = 48
MIN_PREFIX_LENGTH = 3

_WORD = re.compile(r"\w+(?:\.\w+)*")
_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[^\W\d_]+")


@dataclass(frozen=True, slots=True)
class Token:
    """One FTS5 phrase; prefix tokens match as `"phrase"*`."""

    text: str
    prefix: bool = False

    def render(self) -> str:
        quoted = '"' + self.text.replace('"', '""') + '"'
        return f"{quoted}*" if self.prefix else quoted


def words(text: str) -> tuple[str, ...]:
    """Words and dotted names of free text, original case, containing an alphanumeric."""
    return tuple(word for word in _WORD.findall(text) if any(char.isalnum() for char in word))


def split_identifier(word: str) -> tuple[str, ...]:
    """Lowercased sub-tokens of a camelCase, snake_case or dotted identifier."""
    return tuple(part.lower() for part in _PART.findall(word))


def is_identifier(word: str) -> bool:
    return len(split_identifier(word)) > 1 or "_" in word or "." in word


def query_tokens(text: str) -> tuple[Token, ...]:
    """Search tokens: whole identifiers (prefix-matched) and their parts, minus stopwords."""
    plain: list[Token] = []
    identifiers: list[Token] = []
    stopped: list[Token] = []
    for word in words(text):
        lowered = word.lower()
        if is_identifier(word):
            identifiers.append(Token(lowered, prefix=len(lowered) >= MIN_PREFIX_LENGTH))
            plain.extend(
                Token(part)
                for part in split_identifier(word)
                if part not in STOPWORDS and (len(part) > 1 or part.isdigit())
            )
        elif lowered in STOPWORDS:
            stopped.append(Token(lowered))
        else:
            plain.append(Token(lowered))
    chosen = [*identifiers, *plain] or stopped
    return tuple(dict.fromkeys(chosen))[:MAX_TERMS]


def match_expression(text: str) -> str | None:
    """A safe FTS5 MATCH expression for free text, or None when nothing searchable remains."""
    tokens = query_tokens(text)
    if not tokens:
        return None
    return " OR ".join(token.render() for token in tokens)
