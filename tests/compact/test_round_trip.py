import tempfile
from dataclasses import replace
from pathlib import Path

from compact_support import SMALL, ingest, make_ctx, resolve_ref
from hypothesis import given, settings
from hypothesis import strategies as st

from contextweave.compact.rules import compact_candidates
from contextweave.compact.sources import SourceCache
from contextweave.core.refs import parse_ref
from contextweave.index.store import Store
from contextweave.retrieve.assemble import AssembleStage
from contextweave.retrieve.candidates import Candidate, CandidateSet, Scores
from contextweave.retrieve.hydrate import load_file_candidates
from contextweave.retrieve.pack import ContextPack

MODULES = ["os", "sys", "re", "json", "pathlib", "typing", "itertools", "functools"]
SEPARATORS = ["\n", "\n\n", "\n\n\n"]

IMPORT_BLOCK = st.lists(st.sampled_from(MODULES), min_size=1, max_size=8).map(
    lambda names: "".join(f"import {name}\n" for name in names)
)
FUNCTION = st.tuples(st.sampled_from(["alpha", "beta", "gamma"]), st.integers(1, 30)).map(
    lambda pair: f"def {pair[0]}():\n" + "    value = 0\n" * pair[1] + "    return value\n"
)
FILE_TEXT = st.tuples(
    IMPORT_BLOCK,
    st.lists(st.tuples(FUNCTION, st.sampled_from(SEPARATORS)), min_size=1, max_size=4),
).map(lambda parts: parts[0] + "\n" + "".join(body + sep for body, sep in parts[1]))
FILES = st.lists(FILE_TEXT, min_size=1, max_size=4)


def build(root: Path, texts: list[str]) -> Store:
    store = Store.open(root, chunk_config=SMALL)
    for number, text in enumerate(texts):
        ingest(store, f"pkg/mod{number}.py", text)
    return store


def all_candidates(store: Store, finals: list[float]) -> tuple[Candidate, ...]:
    found = [cand for file in store.files() for cand in load_file_candidates(store, file.path)]
    return tuple(
        replace(cand, scores=Scores(final=final))
        for cand, final in zip(found, finals * len(found), strict=False)
    )


def covered(original: Candidate, merged: tuple[Candidate, ...]) -> bool:
    return any(
        m.path == original.path
        and m.start_byte <= original.start_byte
        and original.end_byte <= m.end_byte
        for m in merged
    )


def recoverable(store: Store, pack: ContextPack, original: Candidate) -> bool:
    source = (store.root / original.path).read_bytes()[original.start_byte : original.end_byte]
    refs = [item["ref"] for item in pack["items"] if item["path"] == original.path]
    refs += [
        entry["ref"]
        for entry in pack["omitted"]
        if "path" in entry and entry["path"] == original.path
    ]
    return any((body := resolve_ref(store, ref)) is not None and source in body for ref in refs)


FINALS = st.lists(st.floats(0.0, 1.0), min_size=1, max_size=6)


@settings(max_examples=60, deadline=None)
@given(texts=FILES, finals=FINALS)
def test_candidate_rules_never_lose_a_byte(texts: list[str], finals: list[float]) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = build(Path(directory), texts)
        try:
            cands = all_candidates(store, finals)
            merged = compact_candidates(SourceCache(store.root), cands)
            assert all(covered(cand, merged) for cand in cands)
            for m in merged:
                source = (store.root / m.path).read_bytes()
                inside = [c for c in cands if c.path == m.path and c.start_byte < m.end_byte]
                inside = [c for c in inside if m.start_byte < c.end_byte]
                gaps = bytearray()
                position = m.start_byte
                for c in sorted(inside, key=lambda c: c.order_key[1:]):
                    gaps += source[position : max(position, c.start_byte)]
                    position = max(position, c.end_byte)
                assert not bytes(gaps).strip()
        finally:
            store.close()


@settings(max_examples=60, deadline=None)
@given(texts=FILES, finals=FINALS, budget=st.sampled_from([512, 700, 1200, 10**6]))
def test_every_elided_span_is_recoverable_through_its_ref(
    texts: list[str], finals: list[float], budget: int
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        store = build(Path(directory), texts)
        try:
            cands = all_candidates(store, finals)
            compacted = compact_candidates(SourceCache(store.root), cands)
            result = AssembleStage().assemble(
                make_ctx(store, max_tokens=budget), CandidateSet.of(compacted)
            )
            pack = result.pack
            for item in pack["items"]:
                body = resolve_ref(store, item["ref"])
                assert body is not None
                if item["rep"] == "source":
                    assert item["text"] == body.decode()
            for entry in result.omitted:
                assert resolve_ref(store, entry["ref"]) is not None
            if budget >= 10**6:
                assert all(recoverable(store, pack, cand) for cand in cands)
                for item in pack["items"]:
                    parse_ref(item["ref"])
        finally:
            store.close()


def test_duplicate_import_blocks_collapse_and_stay_recoverable(small_store: Store) -> None:
    imports = "".join(f"import module_{n}\n" for n in range(12))
    body = "def f():\n" + "    value = 0\n" * 30 + "    return value\n"
    ingest(small_store, "pkg/one.py", imports + "\n\n" + body)
    ingest(small_store, "pkg/two.py", imports + "\n\n" + body.replace("f()", "g()"))
    cands = tuple(
        replace(cand, scores=Scores(final=0.5))
        for path in ("pkg/one.py", "pkg/two.py")
        for cand in load_file_candidates(small_store, path)
    )
    compacted = compact_candidates(SourceCache(small_store.root), cands)
    result = AssembleStage().assemble(make_ctx(small_store), CandidateSet.of(compacted))
    collapsed = [item for item in result.pack["items"] if item["rep"] == "collapsed"]
    assert len(collapsed) == 1
    assert result.pack["budget"]["truncated"] is False
    assert resolve_ref(small_store, collapsed[0]["ref"]) is not None
    assert all(recoverable(small_store, result.pack, cand) for cand in cands)
