import tomllib
from collections.abc import Mapping
from pathlib import Path

import tomli_w

from contextweave.core.types import JSON

SCORE_TABLE = "score"
LOCAL_NAME = "local.toml"


def local_path(root: Path) -> Path:
    return root / ".contextweave" / LOCAL_NAME


def merge_score(path: Path, values: Mapping[str, float]) -> None:
    """Set keys of the [score] table in a TOML file, leaving every other key as it was.

    Raises tomllib.TOMLDecodeError when an existing file is not valid TOML.
    """
    data: dict[str, JSON] = {}
    if path.is_file():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    table = data.get(SCORE_TABLE)
    merged: dict[str, JSON] = dict(table) if isinstance(table, dict) else {}
    merged.update(values)
    data[SCORE_TABLE] = merged
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.tmp")
    staging.write_text(tomli_w.dumps(data), encoding="utf-8")
    staging.replace(path)
