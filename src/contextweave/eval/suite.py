import tomllib
from dataclasses import dataclass
from pathlib import Path

from contextweave.eval.jsonvalue import JSON

SUITES_DIR = Path(__file__).parent / "suites"
LIVE_COMMIT = "live"


class SuiteError(Exception):
    """Raised when a suite file is missing or malformed."""


@dataclass(frozen=True, slots=True)
class Case:
    """One query with the qualified names it must surface and those it must not."""

    query: str
    expect: tuple[str, ...]
    must_not: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Suite:
    """A pinned repo checkout and its cases."""

    name: str
    repo: str
    commit: str
    cases: tuple[Case, ...]

    @property
    def live(self) -> bool:
        """Served from the registered index of repo in place, not a pinned checkout."""
        return self.commit == LIVE_COMMIT


def _strings(value: JSON, where: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SuiteError(f"{where}: expected a list of strings")
    items = tuple(item for item in value if isinstance(item, str))
    if len(items) != len(value):
        raise SuiteError(f"{where}: expected a list of strings")
    return items


def _required_string(table: dict[str, JSON], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise SuiteError(f"{where}: missing string '{key}'")
    return value


def _case(raw: JSON, index: int) -> Case:
    where = f"case {index + 1}"
    if not isinstance(raw, dict):
        raise SuiteError(f"{where}: expected a table")
    table = raw
    expect = _strings(table.get("expect"), f"{where} expect")
    if not expect:
        raise SuiteError(f"{where}: expect must not be empty")
    must_not = _strings(table.get("must_not", []), f"{where} must_not")
    return Case(_required_string(table, "query", where), expect, must_not)


def parse_suite(name: str, text: str) -> Suite:
    """Parse suite TOML text."""
    try:
        header: dict[str, JSON] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise SuiteError(f"{name}: {exc}") from exc
    raw_cases = header.get("case")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise SuiteError(f"{name}: no [[case]] entries")
    cases = tuple(_case(raw, index) for index, raw in enumerate(raw_cases))
    if len({case.query for case in cases}) != len(cases):
        raise SuiteError(f"{name}: duplicate case queries")
    return Suite(
        name,
        _required_string(header, "repo", name),
        _required_string(header, "commit", name),
        cases,
    )


def load_suite(name: str, directory: Path = SUITES_DIR) -> Suite:
    """Load suites/<name>.toml, or name itself when it is a path to a .toml file."""
    given = Path(name)
    path = given if given.suffix == ".toml" else directory / f"{name}.toml"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SuiteError(f"suite not found: {name}") from exc
    return parse_suite(path.stem, text)
