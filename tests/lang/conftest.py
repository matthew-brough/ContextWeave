from collections.abc import Callable
from pathlib import Path

import pytest

from contextweave.lang.parse import ParsedFile, parse_bytes

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "lang"
GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "lang"
STORAGE_SUFFIX = ".txt"


def fixture_names() -> list[str]:
    """Relative fixture paths as the extractor sees them (storage suffix removed)."""
    return sorted(
        path.relative_to(FIXTURES).as_posix().removesuffix(STORAGE_SUFFIX)
        for path in FIXTURES.rglob("*")
        if path.is_file() and path.name != "ruff.toml"
    )


def read_fixture(name: str) -> bytes:
    """Bytes of a fixture, found under its plain or storage-suffixed name."""
    plain = FIXTURES / name
    return (plain if plain.is_file() else plain.with_name(plain.name + STORAGE_SUFFIX)).read_bytes()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "fixture_name" in metafunc.fixturenames:
        metafunc.parametrize("fixture_name", fixture_names())


@pytest.fixture
def golden_dir() -> Path:
    return GOLDEN


@pytest.fixture
def all_fixture_names() -> list[str]:
    return fixture_names()


@pytest.fixture
def parse_fixture() -> Callable[[str], ParsedFile]:
    def load(name: str) -> ParsedFile:
        result = parse_bytes(name, read_fixture(name))
        assert result.parsed is not None, result.errors
        return result.parsed

    return load


@pytest.fixture
def parse_text() -> Callable[[str, str], ParsedFile]:
    def load(name: str, text: str) -> ParsedFile:
        result = parse_bytes(name, text.encode("utf-8"))
        assert result.parsed is not None, result.errors
        return result.parsed

    return load
