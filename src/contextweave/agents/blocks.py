import re
from dataclasses import dataclass

BEGIN = re.compile(r"<!-- contextweave:begin v=(\d+) -->")
END = "<!-- contextweave:end -->"


@dataclass(frozen=True, slots=True)
class BlockSpan:
    """Byte-exact location of a marked block: begin marker start to end marker end."""

    start: int
    end: int
    version: int


def find_block(text: str) -> BlockSpan | None:
    """Locate the first begin marker and the end marker that follows it."""
    begin = BEGIN.search(text)
    if begin is None:
        return None
    end = text.find(END, begin.end())
    if end < 0:
        return None
    return BlockSpan(begin.start(), end + len(END), int(begin.group(1)))


def block_version(block: str) -> int:
    """Version number declared by a block's begin marker."""
    span = find_block(block)
    if span is None:
        raise ValueError("block has no markers")
    return span.version


def upsert_block(text: str, block: str) -> str:
    """Replace the existing block in place, else append it after a blank line."""
    span = find_block(text)
    if span is not None:
        return text[: span.start] + block + text[span.end :]
    if not text:
        return block + "\n"
    separator = "\n" * (2 - min(2, len(text) - len(text.rstrip("\n"))))
    return text + separator + block + "\n"


def remove_block(text: str) -> str:
    """Delete the block and its line break; a blank line before a block at the end goes too."""
    span = find_block(text)
    if span is None:
        return text
    start, end = span.start, span.end
    if text.startswith("\r\n", end):
        end += 2
    elif text.startswith("\n", end):
        end += 1
    if end == len(text) and text.endswith("\n\n", 0, start):
        start -= 1
    return text[:start] + text[end:]
