from contextweave.agents.agents_md import AgentsMdAdapter
from contextweave.agents.base import AdapterRegistry
from contextweave.agents.claude_code import ClaudeCodeAdapter
from contextweave.agents.codex import CodexAdapter
from contextweave.core.registry import Registry

DEFAULT_AGENT = "claude_code"
NO_AGENT = "none"


def default_adapters() -> AdapterRegistry:
    """A registry holding every shipped adapter, one row each (V10)."""
    registry: AdapterRegistry = Registry()
    registry.register(ClaudeCodeAdapter.name, ClaudeCodeAdapter())
    registry.register(AgentsMdAdapter.name, AgentsMdAdapter())
    registry.register(CodexAdapter.name, CodexAdapter())
    return registry
