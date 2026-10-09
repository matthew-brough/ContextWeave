from pathlib import Path

from contextweave.agents.base import FileChange, plan_change, read_exact
from contextweave.agents.blocks import find_block, remove_block, upsert_block
from contextweave.agents.routing import routing_block


def plan_block_install(path: Path) -> FileChange | None:
    """Plan inserting or refreshing the routing block in a markdown memory file."""
    return plan_change(path, upsert_block(read_exact(path) or "", routing_block()))


def plan_block_removal(path: Path) -> FileChange | None:
    """Plan stripping the routing block; a file left empty is deleted."""
    text = read_exact(path)
    if text is None or find_block(text) is None:
        return None
    return plan_change(path, remove_block(text) or None)
