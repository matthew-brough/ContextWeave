from contextweave.core.budget import (
    DEFAULT_ESTIMATOR,
    Ceiling,
    bytes35,
    estimators,
)


def test_bytes35_math() -> None:
    assert bytes35("") == 0
    assert bytes35("a") == 1
    assert bytes35("a" * 7) == 2
    assert bytes35("a" * 8) == 3
    assert bytes35("é" * 7) == 4


def test_registry_default() -> None:
    assert estimators.get(DEFAULT_ESTIMATOR) is bytes35


def test_ceiling() -> None:
    ceiling = Ceiling(100)
    assert ceiling.fits(100)
    assert not ceiling.fits(101)
    assert ceiling.over_by(101) == 1
    assert ceiling.over_by(50) == 0
