import json
from dataclasses import dataclass
from pathlib import Path

from contextweave.eval.jsonvalue import JSON
from contextweave.eval.metrics import Metrics

BASELINES_DIR = Path(__file__).parent / "baselines"
PROFILES = ("nomodel", "full")
MAX_RECALL_DROP = 0.02
MAX_TOKEN_GROWTH = 0.10


class BaselineError(Exception):
    """Raised when a baseline file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Baseline:
    """Recorded recall, precision and mean tokens per case."""

    suite: str
    profile: str
    recall: float
    precision: float
    tokens: float


def baseline_path(suite: str, profile: str, directory: Path = BASELINES_DIR) -> Path:
    """eval/baselines/<suite>.<profile>.json."""
    return directory / f"{suite}.{profile}.json"


def baseline_from(suite: str, profile: str, metrics: Metrics) -> Baseline:
    return Baseline(suite, profile, metrics.recall, metrics.precision, metrics.tokens)


def write_baseline(baseline: Baseline, directory: Path = BASELINES_DIR) -> Path:
    """Write the baseline JSON and return its path."""
    path = baseline_path(baseline.suite, baseline.profile, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "suite": baseline.suite,
        "profile": baseline.profile,
        "recall": baseline.recall,
        "precision": baseline.precision,
        "tokens": baseline.tokens,
    }
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _number(body: dict[str, JSON], key: str) -> float:
    value = body.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineError(f"baseline field '{key}' is not a number")
    return float(value)


def read_baseline(suite: str, profile: str, directory: Path = BASELINES_DIR) -> Baseline:
    """Load a baseline."""
    path = baseline_path(suite, profile, directory)
    try:
        raw: JSON = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BaselineError(f"cannot read baseline {path.name}: {exc}") from exc
    if not isinstance(raw, dict):
        raise BaselineError(f"baseline {path.name} is not an object")
    return Baseline(
        suite,
        profile,
        _number(raw, "recall"),
        _number(raw, "precision"),
        _number(raw, "tokens"),
    )


def check_regression(baseline: Baseline, current: Metrics) -> tuple[str, ...]:
    """Failure messages: recall down more than 2 points or tokens up more than 10%."""
    failures: list[str] = []
    drop = baseline.recall - current.recall
    if drop > MAX_RECALL_DROP + 1e-9:
        failures.append(
            f"recall {current.recall:.3f} is {drop * 100:.1f} pts below baseline {baseline.recall:.3f}"
        )
    if baseline.tokens > 0:
        growth = current.tokens / baseline.tokens - 1
        if growth > MAX_TOKEN_GROWTH + 1e-9:
            failures.append(
                f"tokens {current.tokens:.0f} is {growth * 100:.1f}% above baseline {baseline.tokens:.0f}"
            )
    elif current.tokens > 0:
        failures.append(f"tokens {current.tokens:.0f} above zero baseline")
    return tuple(failures)
