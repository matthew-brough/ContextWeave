from hypothesis import given
from hypothesis import strategies as st

from contextweave.retrieve.terms import (
    MAX_TERMS,
    match_expression,
    query_tokens,
    split_identifier,
)


def rendered(text: str) -> list[str]:
    return [token.render() for token in query_tokens(text)]


def test_camel_case_split_and_whole_kept() -> None:
    assert split_identifier("getUserHTTPName2") == ("get", "user", "http", "name", "2")
    assert rendered("getUserName") == ['"getusername"*', '"get"', '"user"', '"name"']


def test_snake_and_dotted_split() -> None:
    assert rendered("parse_config") == ['"parse_config"*', '"parse"', '"config"']
    assert rendered("pkg.mod.func") == ['"pkg.mod.func"*', '"pkg"', '"mod"', '"func"']


def test_stopwords_dropped() -> None:
    assert rendered("where is the retry logic") == ['"retry"', '"logic"']


def test_all_stopwords_fall_back() -> None:
    assert rendered("what is the") == ['"what"', '"is"', '"the"']


def test_plain_words_not_prefixed() -> None:
    assert all(not token.prefix for token in query_tokens("retry logic"))


def test_empty_and_symbols_only() -> None:
    assert match_expression("") is None
    assert match_expression("   ()*-\"'") is None


def test_dedupes_and_caps() -> None:
    assert rendered("retry retry RETRY") == ['"retry"']
    many = " ".join(f"word{index}" for index in range(200))
    assert len(query_tokens(many)) == MAX_TERMS


def test_operators_are_quoted() -> None:
    expression = match_expression('NEAR(a b) OR "x" AND -y* col:z ^w')
    assert expression is not None
    assert "NEAR" not in expression.replace('"near"', "")
    assert all(part.strip().startswith('"') for part in expression.split(" OR "))


@given(st.text())
def test_any_text_yields_wellformed_expression(text: str) -> None:
    expression = match_expression(text)
    if expression is None:
        return
    for part in expression.split(" OR "):
        body = part.removesuffix("*")
        assert body.startswith('"') and body.endswith('"') and len(body) > 2
