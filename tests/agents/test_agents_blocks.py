from hypothesis import given
from hypothesis import strategies as st

from contextweave.agents.blocks import find_block, remove_block, upsert_block
from contextweave.agents.routing import routing_block

BLOCK = routing_block()
OTHER = "<!-- contextweave:begin v=2 -->\nnew\n<!-- contextweave:end -->"
SURROUND = st.text(alphabet=st.characters(blacklist_characters="<", blacklist_categories=["Cs"]))
TERMINATED = SURROUND.map(lambda text: text.rstrip("\n") + "\n" if text.strip("\n") else "")


def test_routing_block_has_markers_and_routes() -> None:
    span = find_block(BLOCK)
    assert span is not None and span.version == 1
    assert BLOCK.startswith("<!-- contextweave:begin v=1 -->")
    assert BLOCK.endswith("<!-- contextweave:end -->")
    for name in ("get_symbol", "context_search", "Read", "recall", "remember"):
        assert name in BLOCK


@given(SURROUND)
def test_insert_keeps_original_as_prefix(text: str) -> None:
    result = upsert_block(text, BLOCK)
    assert result.startswith(text)
    assert BLOCK in result


@given(SURROUND, SURROUND)
def test_replace_touches_only_span(before: str, after: str) -> None:
    text = before + BLOCK + after
    result = upsert_block(text, OTHER)
    assert result == before + OTHER + after
    assert upsert_block(result, OTHER) == result


@given(TERMINATED)
def test_insert_then_remove_roundtrips(text: str) -> None:
    assert remove_block(upsert_block(text, BLOCK)) == text


@given(SURROUND, SURROUND.filter(bool))
def test_remove_in_middle_leaves_outside_bytes(before: str, after: str) -> None:
    text = before + BLOCK + "\n" + after
    assert remove_block(text) == before + after


@given(SURROUND)
def test_insert_is_idempotent(text: str) -> None:
    once = upsert_block(text, BLOCK)
    assert upsert_block(once, BLOCK) == once


def test_remove_without_block_is_identity() -> None:
    assert remove_block("# Title\nbody\n") == "# Title\nbody\n"
