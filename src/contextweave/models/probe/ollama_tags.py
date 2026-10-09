from dataclasses import dataclass

import httpx

from contextweave.models.errors import ModelError
from contextweave.models.wire import as_list, as_object, as_str, field, parse_json_text

TAGS_PATH = "/api/tags"
TAGS_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class TagCatalog:
    names: frozenset[str]

    def has(self, tag: str) -> bool:
        """Exact tag, or the implicit :latest when the tag names no version."""
        return tag in self.names or (":" not in tag and f"{tag}:latest" in self.names)


def fetch_tags(base_url: str, transport: httpx.BaseTransport | None) -> TagCatalog | None:
    """Installed Ollama tags via the read-only list endpoint, or None when unreachable."""
    try:
        with httpx.Client(base_url=base_url, transport=transport, timeout=TAGS_TIMEOUT_S) as client:
            response = client.get(TAGS_PATH)
        if response.status_code >= 400:
            return None
        body = as_object(parse_json_text(response.text), TAGS_PATH)
        rows = as_list(field(body, "models", TAGS_PATH), "models")
        names = [as_str(field(as_object(row, "model"), "name", "model"), "name") for row in rows]
    except httpx.HTTPError, ModelError:
        return None
    return TagCatalog(frozenset(names))
