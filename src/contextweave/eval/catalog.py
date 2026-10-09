from contextweave.core.registry import Registry
from contextweave.eval.adapters import AdapterFactory, ReadAdapter
from contextweave.eval.mcp_adapters import ast_mcp_adapter, cw_adapter


def default_adapters() -> Registry[str, AdapterFactory]:
    """Registry holding read, ast_mcp and cw."""
    registry: Registry[str, AdapterFactory] = Registry()
    registry.register("read", ReadAdapter)
    registry.register("ast_mcp", ast_mcp_adapter)
    registry.register("cw", cw_adapter)
    return registry
