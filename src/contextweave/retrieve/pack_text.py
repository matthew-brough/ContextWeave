from collections.abc import Callable

from contextweave.retrieve.pack import ContextPack, ItemWire

type TextRender = Callable[[ContextPack], str]

FRESH = "fresh"
SOURCE_REP = "source"


def item_lines(item: ItemWire) -> list[str]:
    """Header and body of one served item; the ref appears only when the body is not the full source."""
    head = [item["kind"]] if item["kind"] else []
    if name := item["qualified_name"]:
        head.append(name)
    elif item["symbols"]:
        head.append(f"{{{', '.join(item['symbols'])}}}")
    head.append(f"L{item['start_line']}-{item['end_line']}")
    if item["rep"] != SOURCE_REP:
        head.append(f"[{item['rep']}]")
    if item["freshness"] != FRESH:
        head.append(f"[{item['freshness']}]")
    if item["rep"] != SOURCE_REP or item["oversize"]:
        head.append(item["ref"])
    return [" ".join(head), item["text"]]


def render_pack_text(pack: ContextPack) -> str:
    """Agent-facing pack: a header per file, items with bodies, omitted refs, then problems and hints."""
    lines: list[str] = []
    current: str | None = None
    for item in pack["items"]:
        if item["path"] != current:
            current = item["path"]
            lines.append(f"== {current} ==")
        lines.extend(item_lines(item))
    if pack["omitted"]:
        lines.append("omitted:")
        for entry in pack["omitted"]:
            if "qualified_name" in entry:
                lines.append(f"{entry['ref']} {entry['qualified_name']}".rstrip())
            else:
                lines.append(f"+{entry['count']} more {entry['ref']}")
    coverage = pack["coverage"]
    if not coverage["sufficient"] and coverage["missing"]:
        lines.append(f"coverage missing: {', '.join(coverage['missing'])}")
    lines.extend(
        f"!{error['code']} {error.get('message', '')}".rstrip() for error in pack["errors"]
    )
    lines.extend(f"~{entry['stage']} {entry['reason']}" for entry in pack["degraded"])
    if hint := pack["budget"].get("hint"):
        lines.append(f"hint: {hint}")
    return "\n".join(lines)
