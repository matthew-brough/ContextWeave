import pytest

from contextweave.core.registry import DuplicateKeyError, Registry, UnknownKeyError


def test_order_and_lookup() -> None:
    reg: Registry[str, int] = Registry()
    reg.register("b", 2)
    reg.register("a", 1)
    assert reg.names() == ("b", "a")
    assert list(reg) == ["b", "a"]
    assert reg.values() == (2, 1)
    assert reg.get("a") == 1
    assert "a" in reg
    assert "z" not in reg
    assert len(reg) == 2


def test_duplicate_rejected() -> None:
    reg: Registry[str, int] = Registry()
    reg.register("a", 1)
    with pytest.raises(DuplicateKeyError):
        reg.register("a", 2)


def test_unknown_key() -> None:
    reg: Registry[str, int] = Registry()
    with pytest.raises(UnknownKeyError):
        reg.get("missing")
