from collections.abc import Mapping
from pathlib import Path

from contextweave.cli_commands.common import open_models
from contextweave.core.config import Config
from contextweave.index.indexer import IndexReport, index_project


def full_index(root: Path, config: Config, env: Mapping[str, str]) -> IndexReport:
    """Reconcile the tree and embed what changed; without usable models only the structure."""
    models = open_models(config, env)
    if models is None:
        return index_project(root, config, env=env)
    try:
        return index_project(root, config, slots=models.slots, cache=models.cache, env=env)
    finally:
        models.close()
