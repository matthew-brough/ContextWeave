import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from contextweave.core.config import Config
from contextweave.core.errors import OmittedEntry
from contextweave.eval.mcp_adapters import DISABLE_STAGE_ENV, FULL_PROFILE, PROFILE_ENV
from contextweave.index.projects import ProjectRegistry
from contextweave.index.store import Store
from contextweave.models.calibration import calibration_path, load_calibration
from contextweave.models.residency import Profile
from contextweave.models.slots import ModelSlots
from contextweave.retrieve.stages import STAGE_SLOTS

STAGE_NAMES = frozenset(slot.name for slot in STAGE_SLOTS)

type StoreOpener = Callable[[Path], Store]


def inproc_opener(config: Config, *, readonly: bool) -> StoreOpener:
    """Opener giving every call its own connection to the project db."""

    def open_store(root: Path) -> Store:
        return Store.open(root, readonly=readonly, chunk_config=config.chunk)

    return open_store


class PackCache:
    """Bounded cache of the omitted entries that overflowed a response."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(capacity, 1)
        self._packs: OrderedDict[tuple[str, str], tuple[OmittedEntry, ...]] = OrderedDict()

    @staticmethod
    def pack_id(entries: Sequence[OmittedEntry]) -> str:
        """Content-derived id so identical overflow lists share one pack."""
        digest = hashlib.sha256(json.dumps(list(entries), sort_keys=True).encode())
        return digest.hexdigest()[:12]

    def put(self, key: str, pack_id: str, entries: Sequence[OmittedEntry]) -> None:
        self._packs[(key, pack_id)] = tuple(entries)
        self._packs.move_to_end((key, pack_id))
        while len(self._packs) > self._capacity:
            self._packs.popitem(last=False)

    def get(self, key: str, pack_id: str) -> tuple[OmittedEntry, ...] | None:
        return self._packs.get((key, pack_id))


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a tool call needs besides its arguments."""

    config: Config
    registry: ProjectRegistry
    open_store: StoreOpener
    cwd: Path | None = None
    roots: tuple[Path, ...] = ()
    packs: PackCache = field(default_factory=lambda: PackCache(20))
    models: ModelSlots | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "config", eval_config(self.config, os.environ))
        if self.models is None:
            object.__setattr__(self, "models", eval_models(self.config, os.environ))


def eval_config(config: Config, env: Mapping[str, str]) -> Config:
    """The config with the pipeline stage named by CW_EVAL_DISABLE_STAGE switched off."""
    stage = env.get(DISABLE_STAGE_ENV)
    if stage not in STAGE_NAMES:
        return config
    return replace(config, pipeline=replace(config.pipeline, **{stage: False}))


def eval_models(config: Config, env: Mapping[str, str]) -> ModelSlots | None:
    """Model slots in the serve residency profile when CW_EVAL_PROFILE is full, else none."""
    if env.get(PROFILE_ENV) != FULL_PROFILE:
        return None
    slots = ModelSlots.from_config(
        config.models, calibration=load_calibration(calibration_path(env))
    )
    slots.apply_profile(Profile.SERVE)
    return slots
