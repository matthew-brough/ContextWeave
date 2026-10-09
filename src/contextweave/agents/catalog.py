from contextweave.agents.base import AdapterRegistry
from contextweave.agents.claude_code import ClaudeCodeAdapter
from contextweave.core.registry import Registry

DEFAULT_AGENT = "claude_code"


def default_adapters() -> AdapterRegistry:
    """A registry holding every shipped adapter, one row each (V10)."""
    registry: AdapterRegistry = Registry()
    claude = ClaudeCodeAdapter()
    registry.register(claude.name, claude)
    return registry
