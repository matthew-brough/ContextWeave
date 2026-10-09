import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contextweave.models.calibration import (
    ECE_TARGET,
    Calibration,
    CalibrationKey,
    CalibrationSample,
    apply_temperature,
    calibrated,
    calibration_path,
    expected_calibration_error,
    fit_temperature,
    load_calibration,
    negative_log_likelihood,
    report_question,
    save_calibration,
)

probs = st.floats(min_value=0.001, max_value=0.999)
temps = st.floats(min_value=0.1, max_value=10.0)
samples = st.lists(
    st.builds(CalibrationSample, probs, st.sampled_from([0.0, 1.0])), min_size=1, max_size=60
)


def test_perfectly_calibrated_samples_have_zero_ece() -> None:
    data = [CalibrationSample(1.0, 1.0)] * 5 + [CalibrationSample(0.0, 0.0)] * 5
    assert expected_calibration_error(data) == 0.0


def test_overconfident_wrong_samples_have_ece_near_one() -> None:
    assert expected_calibration_error([CalibrationSample(0.95, 0.0)] * 10) == pytest.approx(0.95)


def test_empty_samples_are_neutral() -> None:
    assert expected_calibration_error([]) == 0.0
    assert fit_temperature([]) == 1.0


@given(samples)
def test_ece_is_in_unit_interval(data: list[CalibrationSample]) -> None:
    assert 0.0 <= expected_calibration_error(data) <= 1.0


@given(probs)
def test_temperature_one_is_identity(prob: float) -> None:
    assert apply_temperature(prob, 1.0) == prob


@given(probs, temps)
def test_temperature_preserves_order_and_unit_range(prob: float, temperature: float) -> None:
    scaled = apply_temperature(prob, temperature)
    assert 0.0 <= scaled <= 1.0
    assert (scaled >= 0.5) == (prob >= 0.5) or math.isclose(prob, 0.5)


@given(probs, temps)
def test_high_temperature_pulls_towards_half(prob: float, temperature: float) -> None:
    if temperature >= 1:
        assert abs(apply_temperature(prob, temperature) - 0.5) <= abs(prob - 0.5) + 1e-9


def test_non_positive_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(0.5, 0.0)


@given(samples)
def test_fitted_temperature_never_worse_than_identity(data: list[CalibrationSample]) -> None:
    fitted = fit_temperature(data)
    assert negative_log_likelihood(data, fitted) <= negative_log_likelihood(data, 1.0) + 1e-6


def test_fit_recovers_overconfidence() -> None:
    data = [CalibrationSample(0.95, 1.0)] * 6 + [CalibrationSample(0.95, 0.0)] * 4
    fitted = fit_temperature(data)
    assert fitted > 1.0
    assert expected_calibration_error(calibrated(data, fitted)) < expected_calibration_error(data)


def test_fit_recovers_underconfidence() -> None:
    data = [CalibrationSample(0.6, 1.0)] * 20 + [CalibrationSample(0.4, 0.0)] * 20
    assert fit_temperature(data) < 1.0


def test_report_flags_target_misses() -> None:
    bad = report_question("q", [CalibrationSample(0.95, 0.0)] * 10)
    good = report_question("q", [CalibrationSample(1.0, 1.0)] * 10)
    assert not bad.meets_target
    assert good.meets_target
    assert good.ece <= ECE_TARGET
    assert bad.samples == 10


def test_calibration_round_trips_through_file(tmp_path: Path) -> None:
    key = CalibrationKey("m", "b1", 1)
    path = tmp_path / "state" / "calibration.json"
    save_calibration(path, Calibration().with_entry(key, {"relevance": 1.5}))
    loaded = load_calibration(path)
    assert loaded.temperatures(key) == {"relevance": 1.5}
    assert loaded.temperatures(CalibrationKey("m", "b2", 1)) == {}


def test_missing_or_corrupt_file_is_empty(tmp_path: Path) -> None:
    assert load_calibration(tmp_path / "none.json").entries == {}
    broken = tmp_path / "broken.json"
    broken.write_text("{nope", encoding="utf-8")
    assert load_calibration(broken).entries == {}


def test_path_honours_override() -> None:
    assert calibration_path({"CW_CALIBRATION_FILE": "/x/c.json"}) == Path("/x/c.json")
