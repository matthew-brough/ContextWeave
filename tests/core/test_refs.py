from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from contextweave.core.refs import (
    Anchor,
    ByteAnchor,
    PartAnchor,
    Ref,
    RefError,
    SymbolAnchor,
    encode_ref,
    parse_ref,
    project_key,
)

KEY = "0123456789ab"
HASH = "abcdef012345"

text = st.text(st.characters(exclude_categories=["Cs"]), min_size=1)
anchors: st.SearchStrategy[Anchor] = st.one_of(
    text.map(SymbolAnchor),
    st.builds(PartAnchor, text, st.integers(min_value=1, max_value=10**6)),
    st.integers(min_value=0, max_value=10**9).flatmap(
        lambda s: st.integers(min_value=s, max_value=10**9).map(lambda e: ByteAnchor(s, e))
    ),
)
refs = st.builds(
    Ref,
    st.just(KEY),
    text,
    anchors,
    st.text("0123456789abcdef", min_size=12, max_size=12),
)


@given(refs)
def test_round_trip(ref: Ref) -> None:
    assert parse_ref(encode_ref(ref)) == ref


def test_symbol_anchor() -> None:
    ref = Ref(KEY, "src/a.py", SymbolAnchor("A.b"), HASH)
    assert encode_ref(ref) == f"cw://{KEY}/src/a.py#A.b@{HASH}"


def test_part_anchor() -> None:
    ref = Ref(KEY, "src/a.py", PartAnchor("A.b", 2), HASH)
    assert encode_ref(ref) == f"cw://{KEY}/src/a.py#A.b~2@{HASH}"
    assert parse_ref(encode_ref(ref)) == ref


def test_byte_anchor() -> None:
    ref = Ref(KEY, "a.md", ByteAnchor(10, 20), HASH)
    assert encode_ref(ref) == f"cw://{KEY}/a.md#b10-20@{HASH}"
    assert parse_ref(encode_ref(ref)) == ref


def test_special_path_chars() -> None:
    ref = Ref(KEY, "dir #1/a@b c.py", SymbolAnchor("f"), HASH)
    assert parse_ref(encode_ref(ref)) == ref


def test_name_lookalikes_round_trip() -> None:
    for anchor in (SymbolAnchor("b1-2"), SymbolAnchor("a~1"), PartAnchor("a~b", 3)):
        ref = Ref(KEY, "x.py", anchor, HASH)
        assert parse_ref(encode_ref(ref)) == ref


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "http://x/y#a@abcdef012345",
        f"cw://{KEY}",
        f"cw://{KEY}/a.py#f",
        f"cw://{KEY}/a.py@{HASH}",
        f"cw://{KEY}/a.py#f@short",
        f"cw://{KEY}/a.py#f@ABCDEF012345",
        f"cw://short/a.py#f@{HASH}",
        f"cw://{KEY}/#f@{HASH}",
        f"cw://{KEY}/a.py#@{HASH}",
        f"cw://{KEY}/a.py#f~0@{HASH}",
        f"cw://{KEY}/a.py#f~x@{HASH}",
        f"cw://{KEY}/a.py#b20-10@{HASH}",
        f"cw://{KEY}/%ff.py#f@{HASH}",
    ],
)
def test_bad_input(bad: str) -> None:
    with pytest.raises(RefError):
        parse_ref(bad)


def test_invalid_ref_construction() -> None:
    with pytest.raises(RefError):
        Ref(KEY, "a.py", ByteAnchor(5, 1), HASH)


def test_project_key(tmp_path: Path) -> None:
    key = project_key(tmp_path)
    assert len(key) == 12
    assert key == project_key(tmp_path)
    link = tmp_path.parent / (tmp_path.name + "-link")
    link.symlink_to(tmp_path)
    assert project_key(link) == key
    assert project_key(tmp_path / "other") != key
