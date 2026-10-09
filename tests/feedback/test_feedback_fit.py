import pytest
from feedback_support import synthetic_samples

from contextweave.core.config import ScoreConfig
from contextweave.feedback.fit import (
    InsufficientDataError,
    Sample,
    auc,
    effective,
    fit_weights,
)


def test_fit_moves_weight_to_the_predictive_feature_and_improves_ranking() -> None:
    current = ScoreConfig()
    fit = fit_weights(synthetic_samples(), current)
    assert fit.weights.gate_lexical_match > current.gate_lexical_match * 3
    assert fit.auc_after > fit.auc_before
    assert fit.auc_after > 0.95


def test_fit_is_deterministic() -> None:
    first = fit_weights(synthetic_samples(), ScoreConfig())
    assert first == fit_weights(synthetic_samples(), ScoreConfig())


def test_fit_keeps_signs_scale_and_unvarying_features() -> None:
    current = ScoreConfig()
    fit = fit_weights(synthetic_samples(), current)
    assert fit.weights.final_redundancy <= 0
    assert fit.weights.final_graph >= 0
    assert fit.weights.gate_answers_query == current.gate_answers_query
    assert fit.weights.gate_norm == current.gate_norm
    assert fit.weights.final_gate == current.final_gate
    kept = (0, 2)
    assert sum(abs(effective(fit.weights)[i]) for i in range(7) if i not in kept) == pytest.approx(
        sum(abs(effective(current)[i]) for i in range(7) if i not in kept)
    )


def test_fit_refuses_too_little_data() -> None:
    with pytest.raises(InsufficientDataError) as caught:
        fit_weights(synthetic_samples(20), ScoreConfig())
    assert caught.value.samples == 20
    assert caught.value.required == 50


def test_fit_refuses_when_one_class_is_tiny() -> None:
    samples = [Sample((0.0, 0.1, 1.0, 0.2, 0.0, 0.5, 0.0), False)] * 80
    with pytest.raises(InsufficientDataError):
        fit_weights(samples, ScoreConfig())


def test_auc_handles_ties() -> None:
    assert auc([1.0, 1.0, 2.0, 0.0], [True, False, True, False]) == pytest.approx(0.875)
    assert auc([1.0, 2.0], [True, True]) == 0.5
