import re

WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
STOPWORDS = frozenset(
    {
        "about",
        "all",
        "and",
        "any",
        "are",
        "does",
        "for",
        "from",
        "get",
        "gets",
        "has",
        "have",
        "how",
        "into",
        "its",
        "logic",
        "not",
        "set",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "this",
        "via",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "work",
        "works",
        "code",
        "function",
        "handle",
        "handled",
        "handles",
    }
)


def keywords(query: str) -> tuple[str, ...]:
    """The distinct lowercase words of query an agent would grep for, in order."""
    words = (word.lower() for word in WORD.findall(query))
    return tuple(dict.fromkeys(word for word in words if word not in STOPWORDS))
