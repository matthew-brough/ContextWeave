from collections.abc import Callable

from contextweave.retrieve.pack import ContextPack

type TextRender = Callable[[ContextPack], str]


def render_pack_text(pack: ContextPack) -> str:
    """Terse text fallback: a header per file, ref and body per item, then omitted refs."""
    lines: list[str] = []
    current: str | None = None
    for item in pack["items"]:
        if item["path"] != current:
            current = item["path"]
            lines.append(f"# {current}")
        name = item["qualified_name"] or ""
        lines.append(
            f"## {item['ref']} {item['rep']} L{item['start_line']}-{item['end_line']} {name}"
        )
        lines.append(item["text"])
    if pack["omitted"]:
        lines.append("omitted:")
        lines.extend(f"{entry['ref']} {entry['reason']}" for entry in pack["omitted"])
    return "\n".join(lines)
