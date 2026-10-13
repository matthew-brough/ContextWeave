import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from contextweave.core.config import ModelsConfig
from contextweave.core.paths import user_cache_dir
from contextweave.models.errors import ModelTransportError
from contextweave.models.providers import Slot

PLACEHOLDER = "{weights}"


class WeightsRefError(ValueError):
    """A models.<slot>.weights entry is not `org/repo` or `org/repo/path/in/repo`."""


@dataclass(frozen=True, slots=True)
class WeightRef:
    """One Hugging Face download: a single file into the weights dir, or a whole repo into the HF cache."""

    repo: str
    file: str | None

    @classmethod
    def parse(cls, ref: str) -> WeightRef:
        parts = [part for part in ref.strip().split("/") if part]
        if len(parts) < 2:
            raise WeightsRefError(f"weights entry '{ref}' is not org/repo[/file]")
        return cls("/".join(parts[:2]), "/".join(parts[2:]) or None)

    def __str__(self) -> str:
        return f"{self.repo}/{self.file}" if self.file else self.repo


def weights_root(models: ModelsConfig, env: Mapping[str, str]) -> Path:
    """Where single-file weights live: models.weights_dir, else <cache>/models."""
    if models.weights_dir:
        return Path(models.weights_dir).expanduser()
    return user_cache_dir(env) / "models"


def slot_weights(models: ModelsConfig) -> Mapping[Slot, tuple[str, ...]]:
    return {
        Slot.EMBED: models.embed.weights,
        Slot.ANNOTATE: models.annotate.weights,
        Slot.CLASSIFY: models.classify.weights,
    }


def local_path(ref: WeightRef, root: Path) -> Path | None:
    """The weights on disk, or None when they still need fetching."""
    if ref.file is not None:
        path = root / ref.file
        return path if path.is_file() else None
    try:
        return Path(snapshot_download(ref.repo, local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def fetch(ref: WeightRef, root: Path) -> Path:
    """Download the weights; a file lands in root, a repo snapshot in the HF cache."""
    if ref.file is not None:
        return Path(hf_hub_download(ref.repo, ref.file, local_dir=root))
    return Path(snapshot_download(ref.repo))


def launch_expander(refs: Sequence[str], root: Path) -> Callable[[str], str]:
    """Expand {weights} in a launch command to the first entry's local path; refuse when any is missing."""

    def expand(command: str) -> str:
        paths: list[Path] = []
        for raw in refs:
            ref = WeightRef.parse(raw)
            path = local_path(ref, root)
            if path is None:
                raise ModelTransportError(f"weights {ref} not downloaded; run `cw models fetch`")
            paths.append(path)
        if PLACEHOLDER not in command:
            return command
        if not paths:
            raise ModelTransportError(f"launch uses {PLACEHOLDER} but no weights are configured")
        return command.replace(PLACEHOLDER, shlex.quote(str(paths[0])))

    return expand


@dataclass(frozen=True, slots=True)
class FetchResult:
    slot: Slot
    ref: str
    path: Path | None
    fetched: bool
    error: str | None = None


def ensure_weights(
    models: ModelsConfig,
    env: Mapping[str, str],
    on_fetch: Callable[[Slot, str], None] = lambda slot, ref: None,
) -> list[FetchResult]:
    """Download every configured weights entry that is not on disk yet."""
    root = weights_root(models, env)
    results: list[FetchResult] = []
    for slot, refs in slot_weights(models).items():
        for raw in refs:
            try:
                ref = WeightRef.parse(raw)
                present = local_path(ref, root)
                if present is not None:
                    results.append(FetchResult(slot, raw, present, fetched=False))
                    continue
                on_fetch(slot, raw)
                results.append(FetchResult(slot, raw, fetch(ref, root), fetched=True))
            except Exception as exc:
                results.append(FetchResult(slot, raw, None, False, str(exc) or type(exc).__name__))
    return results
