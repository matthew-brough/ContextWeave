from collections.abc import Hashable, Iterator


class RegistryError(Exception):
    """Base error for registry misuse."""


class DuplicateKeyError(RegistryError):
    """Raised when a key is registered twice."""


class UnknownKeyError(RegistryError):
    """Raised when a key is not registered."""


class Registry[K: Hashable, V]:
    """Insertion-ordered registry of values by key."""

    def __init__(self) -> None:
        self._entries: dict[K, V] = {}

    def register(self, key: K, value: V) -> V:
        if key in self._entries:
            raise DuplicateKeyError(f"already registered: {key!r}")
        self._entries[key] = value
        return value

    def get(self, key: K) -> V:
        try:
            return self._entries[key]
        except KeyError:
            raise UnknownKeyError(f"not registered: {key!r}") from None

    def names(self) -> tuple[K, ...]:
        return tuple(self._entries)

    def values(self) -> tuple[V, ...]:
        return tuple(self._entries.values())

    def items(self) -> tuple[tuple[K, V], ...]:
        return tuple(self._entries.items())

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[K]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
