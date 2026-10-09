import json

from contextweave.core.budget import (
    DEFAULT_ESTIMATOR,
    Ceiling,
    bytes35,
    estimators,
    measure_payload,
)


def test_bytes35_math() -> None:
    assert bytes35("") == 0
    assert bytes35("a") == 1
    assert bytes35("a" * 7) == 2
    assert bytes35("a" * 8) == 3
    assert bytes35("é" * 7) == 4


def test_registry_default() -> None:
    assert estimators.get(DEFAULT_ESTIMATOR) is bytes35


def test_envelope_counted() -> None:
    structured = {"b": 1, "a": ["x"]}
    compact = json.dumps({"a": ["x"], "b": 1}, separators=(",", ":"))
    text = "hello world"
    assert measure_payload(structured, text) == bytes35(compact) + bytes35(text)
    assert measure_payload(structured, text) > measure_payload(structured, "")


def test_custom_estimator() -> None:
    assert measure_payload({}, "abc", len) == len("{}") + 3


def test_ceiling() -> None:
    ceiling = Ceiling(100)
    assert ceiling.fits(100)
    assert not ceiling.fits(101)
    assert ceiling.over_by(101) == 1
    assert ceiling.over_by(50) == 0
