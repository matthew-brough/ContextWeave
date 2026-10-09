import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from contextweave.core.paths import user_state_dir
from contextweave.core.types import JSON

PROB_FLOOR = 1e-6
ECE_BINS = 10
ECE_TARGET = 0.05
TEMPERATURE_MIN = 0.05
TEMPERATURE_MAX = 20.0
SEARCH_STEPS = 80
GOLDEN = (math.sqrt(5) - 1) / 2
CALIBRATION_FILE = "calibration.json"
CALIBRATION_FILE_ENV = "CW_CALIBRATION_FILE"


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One predicted probability and the observed outcome in [0, 1]."""

    prob: float
    label: float


@dataclass(frozen=True, slots=True)
class CalibrationKey:
    """What a fitted temperature is valid for."""

    model_id: str
    build_id: str
    question_set_version: int


def clip_prob(prob: float) -> float:
    return min(max(prob, PROB_FLOOR), 1 - PROB_FLOOR)


def logit(prob: float) -> float:
    clipped = clip_prob(prob)
    return math.log(clipped / (1 - clipped))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1 + exp)


def apply_temperature(prob: float, temperature: float) -> float:
    """Scale the logit of prob by 1/temperature; temperature 1 is the identity."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return prob
    return sigmoid(logit(prob) / temperature)


def expected_calibration_error(samples: Sequence[CalibrationSample], bins: int = ECE_BINS) -> float:
    """Equal-width binned ECE: sum over bins of weight * |mean label - mean prob|."""
    if not samples:
        return 0.0
    buckets: list[list[CalibrationSample]] = [[] for _ in range(bins)]
    for sample in samples:
        index = min(int(min(max(sample.prob, 0.0), 1.0) * bins), bins - 1)
        buckets[index].append(sample)
    total = len(samples)
    error = 0.0
    for bucket in buckets:
        if bucket:
            confidence = sum(s.prob for s in bucket) / len(bucket)
            accuracy = sum(s.label for s in bucket) / len(bucket)
            error += len(bucket) / total * abs(accuracy - confidence)
    return error


def negative_log_likelihood(samples: Sequence[CalibrationSample], temperature: float) -> float:
    """Mean binary cross entropy of the temperature-scaled probabilities."""
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        prob = clip_prob(apply_temperature(sample.prob, temperature))
        total -= sample.label * math.log(prob) + (1 - sample.label) * math.log(1 - prob)
    return total / len(samples)


def fit_temperature(samples: Sequence[CalibrationSample]) -> float:
    """Temperature minimising NLL, by golden-section search over log temperature."""
    if not samples:
        return 1.0
    low, high = math.log(TEMPERATURE_MIN), math.log(TEMPERATURE_MAX)
    left = high - GOLDEN * (high - low)
    right = low + GOLDEN * (high - low)
    f_left = negative_log_likelihood(samples, math.exp(left))
    f_right = negative_log_likelihood(samples, math.exp(right))
    for _ in range(SEARCH_STEPS):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - GOLDEN * (high - low)
            f_left = negative_log_likelihood(samples, math.exp(left))
        else:
            low, left, f_left = left, right, f_right
            right = low + GOLDEN * (high - low)
            f_right = negative_log_likelihood(samples, math.exp(right))
    return math.exp((low + high) / 2)


def calibrated(samples: Sequence[CalibrationSample], temperature: float) -> list[CalibrationSample]:
    return [replace(sample, prob=apply_temperature(sample.prob, temperature)) for sample in samples]


@dataclass(frozen=True, slots=True)
class QuestionCalibration:
    """ECE before and after fitting one question's temperature."""

    question: str
    samples: int
    ece: float
    temperature: float
    ece_after: float

    @property
    def meets_target(self) -> bool:
        return self.ece <= ECE_TARGET


def report_question(question: str, samples: Sequence[CalibrationSample]) -> QuestionCalibration:
    """ECE as served and after the NLL-fitted temperature."""
    temperature = fit_temperature(samples)
    return QuestionCalibration(
        question,
        len(samples),
        expected_calibration_error(samples),
        temperature,
        expected_calibration_error(calibrated(samples, temperature)),
    )


@dataclass(frozen=True, slots=True)
class Calibration:
    """Per-question temperatures keyed by (model_id, build_id, question_set_version)."""

    entries: Mapping[CalibrationKey, Mapping[str, float]] = field(
        default_factory=dict[CalibrationKey, Mapping[str, float]]
    )

    def temperatures(self, key: CalibrationKey) -> Mapping[str, float]:
        return self.entries.get(key, {})

    def with_entry(self, key: CalibrationKey, temperatures: Mapping[str, float]) -> Calibration:
        """Replace the key's temperatures."""
        return Calibration({**self.entries, key: dict(temperatures)})

    def to_json(self) -> dict[str, JSON]:
        rows: list[JSON] = [
            {
                "model_id": key.model_id,
                "build_id": key.build_id,
                "question_set_version": key.question_set_version,
                "temperatures": dict(temperatures),
            }
            for key, temperatures in self.entries.items()
        ]
        return {"entries": rows}


def calibration_path(env: Mapping[str, str]) -> Path:
    """CW_CALIBRATION_FILE when set, else calibration.json in the user state dir."""
    override = env.get(CALIBRATION_FILE_ENV)
    return Path(override) if override else user_state_dir(env) / CALIBRATION_FILE


def _entry(raw: JSON) -> tuple[CalibrationKey, dict[str, float]] | None:
    if not isinstance(raw, dict):
        return None
    model_id, build_id = raw.get("model_id"), raw.get("build_id")
    version, table = raw.get("question_set_version"), raw.get("temperatures")
    if not (
        isinstance(model_id, str)
        and isinstance(build_id, str)
        and isinstance(version, int)
        and isinstance(table, dict)
    ):
        return None
    temperatures = {
        name: float(value)
        for name, value in table.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
    }
    return CalibrationKey(model_id, build_id, version), temperatures


def load_calibration(path: Path) -> Calibration:
    """Read the calibration file; missing or malformed content is an empty calibration."""
    try:
        decoded: JSON = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return Calibration()
    rows = decoded.get("entries") if isinstance(decoded, dict) else None
    found = [_entry(row) for row in rows] if isinstance(rows, list) else []
    return Calibration({entry[0]: entry[1] for entry in found if entry is not None})


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration.to_json(), indent=2, sort_keys=True), encoding="utf-8")
