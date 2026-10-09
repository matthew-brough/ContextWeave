from contextweave.index.records import FileProfile
from contextweave.lang.registry import GROUPS, PROFILES, Group


def as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"expected int, got {type(value).__name__}")
    return value


def as_str(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")
    return value


def as_opt_str(value: object) -> str | None:
    return None if value is None else as_str(value)


def as_opt_int(value: object) -> int | None:
    return None if value is None else as_int(value)


def as_group(value: object) -> Group | None:
    if value is None:
        return None
    text = as_str(value)
    for group in GROUPS:
        if group == text:
            return group
    raise ValueError(f"unknown group {text}")


def as_profile(value: object) -> FileProfile:
    text = as_str(value)
    if text == "text":
        return "text"
    for profile in PROFILES:
        if profile == text:
            return profile
    raise ValueError(f"unknown profile {text}")
