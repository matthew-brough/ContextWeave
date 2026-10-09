from importlib.resources import files


def routing_block() -> str:
    """The routing block shipped as package data, shared by every agent that writes one."""
    return files("contextweave.agents").joinpath("routing.md").read_text("utf-8").rstrip("\n")
