from collections.abc import Sequence

from contextweave.core.budget import Estimator, bytes35


def build_session_context(
    title: str,
    health: str,
    warnings: Sequence[str],
    limit_tokens: int,
    estimator: Estimator = bytes35,
) -> str:
    """Assemble project-scoped context, dropping whole lines that would exceed the token limit."""
    lines: list[str] = []
    for line in [title, health, *warnings]:
        candidate = [*lines, line]
        if estimator("\n".join(candidate)) > limit_tokens:
            continue
        lines = candidate
    return "\n".join(lines)
