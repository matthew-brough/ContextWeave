from collections.abc import Callable, Sequence

from contextweave.core.budget import Estimator, bytes35
from contextweave.memory.wire import RecallHit

MAX_ITEMS = 5
ITEM_CHARS = 200


def _clip(text: str) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= ITEM_CHARS else flat[: ITEM_CHARS - 1] + "…"


def decision_line(hit: RecallHit) -> str:
    """One bullet for a decision hit."""
    reason = hit.get("reason")
    suffix = f" (because {_clip(reason)})" if reason else ""
    return f"- {_clip(hit['text'])}{suffix}"


def code_area_line(hit: RecallHit) -> str:
    """One bullet for a code area hit."""
    return f"- {hit.get('path', '?')}: {_clip(hit['text'])}"


def _group(
    heading: str, hits: Sequence[RecallHit], render: Callable[[RecallHit], str]
) -> list[str]:
    items = [render(hit) for hit in hits[:MAX_ITEMS]]
    return [heading, *items] if items else []


def build_session_context(
    title: str,
    health: str,
    warnings: Sequence[str],
    decisions: Sequence[RecallHit],
    code_areas: Sequence[RecallHit],
    limit_tokens: int,
    estimator: Estimator = bytes35,
) -> str:
    """Assemble project-scoped context, dropping whole lines that would exceed the token limit."""
    groups: list[list[str]] = [
        [title],
        [health],
        *([line] for line in warnings),
        _group("Recent decisions:", decisions, decision_line),
        _group("Active code areas:", code_areas, code_area_line),
    ]
    lines: list[str] = []
    for group in groups:
        for index, line in enumerate(group):
            candidate = [*lines, line]
            if estimator("\n".join(candidate)) > limit_tokens:
                if index == 0:
                    break
                continue
            lines = candidate
    return "\n".join(lines)
