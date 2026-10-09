from contextweave.index.store import SqlValue

GLOB_CHARS = frozenset("*?[")


def is_glob(scope: str) -> bool:
    return any(char in GLOB_CHARS for char in scope)


def normalize_scope(scope: str | None) -> str | None:
    if scope is None:
        return None
    text = scope.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    text = text if is_glob(text) else text.strip("/")
    return text or None


def scope_clause(column: str, scope: str | None) -> tuple[str, list[SqlValue]]:
    """SQL predicate and params restricting a path column to a prefix or glob scope."""
    text = normalize_scope(scope)
    if text is None:
        return "1", []
    if is_glob(text):
        patterns = list(dict.fromkeys([text, text.replace("**/", "")]))
        clause = " OR ".join(f"{column} GLOB ?" for _ in patterns)
        return f"({clause})", list(patterns)
    return f"({column} = ? OR substr({column}, 1, ?) = ?)", [text, len(text) + 1, f"{text}/"]
