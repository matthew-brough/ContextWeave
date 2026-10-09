import math
from dataclasses import dataclass, replace

from contextweave.core.config import ScoreConfig
from contextweave.feedback.served import FEATURE_NAMES, Features

MIN_SAMPLES = 50
MIN_PER_CLASS = 10
ITERATIONS = 800
LEARNING_RATE = 4.0
L2 = 0.001
VARIANCE_FLOOR = 1e-9
LOGIT_CLIP = 30.0
GATE_FEATURES = 5
GRAPH_INDEX = 5
REDUNDANCY_INDEX = 6


@dataclass(frozen=True, slots=True)
class Sample:
    """One served item: its score components and whether the agent went on to use it."""

    features: Features
    used: bool


@dataclass(frozen=True, slots=True)
class Fit:
    """Tuned weights plus the pooled ranking AUC of the old and new weights on the same data."""

    weights: ScoreConfig
    samples: int
    positives: int
    auc_before: float
    auc_after: float


class InsufficientDataError(Exception):
    """Too few labelled samples (or one class too small) to fit weights."""

    def __init__(self, samples: int, positives: int, required: int) -> None:
        super().__init__(
            f"{samples} labelled items ({positives} used, {samples - positives} unused); "
            f"need at least {required} items with {MIN_PER_CLASS} of each kind"
        )
        self.samples = samples
        self.positives = positives
        self.required = required


def effective(weights: ScoreConfig) -> tuple[float, ...]:
    """Per-feature multipliers of the final score, in FEATURE_NAMES order."""
    scale = weights.final_gate / weights.gate_norm
    return (
        weights.gate_answers_query * scale,
        weights.gate_relevance * scale,
        weights.gate_role_implementation * scale,
        weights.gate_lexical_match * scale,
        weights.gate_structural_value * scale,
        weights.final_graph,
        weights.final_redundancy,
    )


def with_effective(base: ScoreConfig, values: tuple[float, ...]) -> ScoreConfig:
    """The inverse of effective(): gate_norm and final_gate stay, the rest follow the values."""
    unscale = base.gate_norm / base.final_gate
    return replace(
        base,
        gate_answers_query=values[0] * unscale,
        gate_relevance=values[1] * unscale,
        gate_role_implementation=values[2] * unscale,
        gate_lexical_match=values[3] * unscale,
        gate_structural_value=values[4] * unscale,
        final_graph=values[GRAPH_INDEX],
        final_redundancy=values[REDUNDANCY_INDEX],
    )


def auc(scores: list[float], used: list[bool]) -> float:
    """Probability a used item outscores an unused one (ties count half)."""
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and scores[order[stop + 1]] == scores[order[start]]:
            stop += 1
        for position in range(start, stop + 1):
            ranks[order[position]] = (start + stop) / 2 + 1
        start = stop + 1
    positives = sum(used)
    negatives = len(used) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    rank_sum = sum(rank for rank, flag in zip(ranks, used, strict=True) if flag)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _sigmoid(z: float) -> float:
    clipped = max(-LOGIT_CLIP, min(LOGIT_CLIP, z))
    return 1.0 / (1.0 + math.exp(-clipped))


def _logistic(rows: list[list[float]], labels: list[bool]) -> list[float]:
    positives = sum(labels)
    sample_weights = [
        0.5 / positives if label else 0.5 / (len(labels) - positives) for label in labels
    ]
    width = len(rows[0])
    coefficients = [0.0] * width
    bias = 0.0
    for _ in range(ITERATIONS):
        gradient = [L2 * value for value in coefficients]
        bias_gradient = 0.0
        for row, label, weight in zip(rows, labels, sample_weights, strict=True):
            z = bias + sum(c * x for c, x in zip(coefficients, row, strict=True))
            error = weight * (_sigmoid(z) - (1.0 if label else 0.0))
            bias_gradient += error
            for column in range(width):
                gradient[column] += error * row[column]
        bias -= LEARNING_RATE * bias_gradient
        coefficients = [c - LEARNING_RATE * g for c, g in zip(coefficients, gradient, strict=True)]
    return coefficients


def _active_columns(samples: list[Sample]) -> list[int]:
    active: list[int] = []
    for column in range(len(FEATURE_NAMES)):
        values = [sample.features[column] for sample in samples]
        mean = sum(values) / len(values)
        if sum((value - mean) ** 2 for value in values) / len(values) > VARIANCE_FLOOR:
            active.append(column)
    return active


def _signed(column: int, value: float) -> float:
    return min(value, 0.0) if column == REDUNDANCY_INDEX else max(value, 0.0)


def fit_weights(samples: list[Sample], current: ScoreConfig, min_samples: int = MIN_SAMPLES) -> Fit:
    """Fit [score] weights by balanced L2 logistic regression, keeping the old weight scale.

    Features without variance keep their current weight; fitted weights are clamped to their
    sign (redundancy <= 0, the rest >= 0) and rescaled so the L1 mass of the active weights
    matches the old one, which keeps absolute thresholds such as gate.critical meaningful.
    """
    positives = sum(sample.used for sample in samples)
    if (
        len(samples) < min_samples
        or positives < MIN_PER_CLASS
        or len(samples) - positives < MIN_PER_CLASS
    ):
        raise InsufficientDataError(len(samples), positives, min_samples)
    active = _active_columns(samples)
    before = effective(current)
    labels = [sample.used for sample in samples]
    rows = [[sample.features[column] for column in active] for sample in samples]
    fitted = [
        _signed(column, value)
        for column, value in zip(active, _logistic(rows, labels), strict=True)
    ]
    old_mass = sum(abs(before[column]) for column in active)
    new_mass = sum(abs(value) for value in fitted)
    after = list(before)
    if new_mass > 0:
        for column, value in zip(active, fitted, strict=True):
            after[column] = value * old_mass / new_mass
    weights = with_effective(current, tuple(after))
    return Fit(
        weights,
        len(samples),
        positives,
        _pooled_auc(samples, before),
        _pooled_auc(samples, effective(weights)),
    )


def _pooled_auc(samples: list[Sample], multipliers: tuple[float, ...]) -> float:
    scores = [
        sum(m * x for m, x in zip(multipliers, sample.features, strict=True)) for sample in samples
    ]
    return auc(scores, [sample.used for sample in samples])
