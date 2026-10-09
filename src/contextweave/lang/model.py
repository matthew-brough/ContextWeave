from dataclasses import dataclass

from contextweave.lang.registry import Group, Profile


@dataclass(frozen=True, slots=True)
class LangError:
    """A non-fatal problem carried in a payload instead of raised."""

    code: str
    path: str | None = None
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class Symbol:
    """A definition from the symbols or defs profile; bytes are authoritative."""

    name: str
    qualified_name: str
    kind: str
    lang: str
    group: Group
    profile: Profile
    path: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    signature: str
    docstring: str | None
    parent: str | None


@dataclass(frozen=True, slots=True)
class Import:
    """A module reference; the range covers the statement, or the link target for markdown."""

    module: str
    names: tuple[str, ...]
    alias: str | None
    kind: str
    lang: str
    group: Group
    profile: Profile
    start_byte: int
    end_byte: int
    line: int


@dataclass(frozen=True, slots=True)
class Export:
    """A symbol covered by an export statement."""

    name: str
    kind: str
    lang: str
    group: Group
    profile: Profile
    start_byte: int
    end_byte: int
    line: int
