import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from compact_support import ingest, make_ctx, pack_item, resolve_ref, span
from hypothesis import given, settings
from hypothesis import strategies as st

from contextweave.compact.lossy import attach_annotations, prepare_lossy
from contextweave.core.budget import bytes35
from contextweave.core.config import AnnotateConfig, Config
from contextweave.core.refs import SymbolAnchor
from contextweave.index.store import Store
from contextweave.models.slots import ModelSlots
from contextweave.retrieve.assemble import MIN_MAX_TOKENS, AssembleStage
from contextweave.retrieve.candidates import Candidate, CandidateSet, Scores
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.ladder import next_rung
from contextweave.retrieve.pack import CachedAnnotation, PackItem
from contextweave.retrieve.search import SearchOptions, search

ANNOTATE = AnnotateConfig()
PURPOSE = "handles one request"
BODY = "    value = 0\n" * 60


def annotate_all(store: Store, config: AnnotateConfig = ANNOTATE, purpose: str = PURPOSE) -> None:
    store.execute(
        "INSERT OR REPLACE INTO annotations(chunk_id, content_hash, model_id, prompt_version,"
        " purpose, tags) SELECT id, content_hash, ?, ?, ?, '' FROM chunks",
        (config.model, config.prompt_version, purpose),
    )


def corpus(store: Store, files: int, lines: int = 60) -> None:
    for number in range(files):
        body = f"def handler_{number}():\n" + "    value = 0\n" * lines + "    return value\n"
        ingest(store, f"pkg/mod{number}.py", body)


def candidates(store: Store, path: str) -> list[Candidate]:
    ids = [chunk.id for chunk in store.chunks_in_file(path)]
    return list(load_candidates(store, ids).values())


def item_for(store: Store, path: str, **changes: object) -> PackItem:
    cand = candidates(store, path)[0]
    source = (store.root / path).read_bytes()
    text = source[cand.start_byte : cand.end_byte].decode()
    return replace(pack_item(cand, text, "cw://x"), signature="def handler_0()", **changes)


def with_note(item: PackItem) -> PackItem:
    return replace(item, annotation=CachedAnnotation(PURPOSE, ANNOTATE.model, 1))


def test_summary_rung_golden(store: Store) -> None:
    corpus(store, 1)
    item = with_note(item_for(store, "pkg/mod0.py"))
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "summary"
    after = rung.apply(item)
    assert after is not None
    assert after.rep == "summary"
    assert after.text == f"def handler_0()\n{PURPOSE}"
    wire = after.to_wire()
    assert wire["synthesized"] is True
    assert wire.get("model") == ANNOTATE.model
    assert wire.get("prompt_version") == ANNOTATE.prompt_version
    assert wire["ref"] == "cw://x"


def test_summary_then_signature_then_ref(store: Store) -> None:
    corpus(store, 1)
    item = with_note(item_for(store, "pkg/mod0.py"))
    names: list[str] = []
    current: PackItem | None = item
    while current is not None and (rung := next_rung(current, True)) is not None:
        names.append(rung.name)
        current = rung.apply(current)
    assert names == ["summary", "signature", "ref"]


def test_no_summary_rung_without_annotation(store: Store) -> None:
    corpus(store, 1)
    rung = next_rung(item_for(store, "pkg/mod0.py"), False)
    assert rung is not None and rung.name == "signature"


def test_boilerplate_folds_to_signature() -> None:
    cand = span(candidates_stub(), 0, 40)
    cand = replace(cand, qualified_name="Point.__repr__", kind="method")
    text = "def __repr__(self):\n    return f'Point({self.x})'\n"
    item = replace(pack_item(cand, text, "cw://x"), signature="def __repr__(self)")
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "fold_boilerplate"
    folded = rung.apply(item)
    assert folded is not None and folded.text == "def __repr__(self)"


def test_test_items_collapse_to_names() -> None:
    cand = replace(
        span(candidates_stub(), 0, 80),
        path="tests/test_x.py",
        qualified_name="test_adds",
        kind="function",
    )
    text = "def test_adds():\n    assert 1 + 1 == 2\n    assert 2 + 2 == 4\n"
    item = pack_item(cand, text, "cw://x")
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "names_only"
    named = rung.apply(item)
    assert named is not None and named.text == "test_adds" and named.rep == "signature"
    assert named.to_wire()["synthesized"] is False


def candidates_stub() -> Candidate:
    return Candidate(
        chunk_id=1,
        path="a.py",
        start_byte=0,
        end_byte=1,
        anchor=SymbolAnchor("x"),
        profile="symbols",
        group="core",
        scores=Scores(final=0.5),
    )


def test_attach_uses_current_annotation(store: Store) -> None:
    corpus(store, 1)
    annotate_all(store)
    item = item_for(store, "pkg/mod0.py")
    (attached,) = attach_annotations(store, ANNOTATE, [item])
    assert attached.annotation == CachedAnnotation(PURPOSE, ANNOTATE.model, ANNOTATE.prompt_version)


def test_attach_skips_missing(store: Store) -> None:
    corpus(store, 1)
    (attached,) = attach_annotations(store, ANNOTATE, [item_for(store, "pkg/mod0.py")])
    assert attached.annotation is None


def test_attach_skips_stale_hash(store: Store) -> None:
    corpus(store, 1)
    annotate_all(store)
    store.execute("UPDATE annotations SET content_hash = 'stale'")
    (attached,) = attach_annotations(store, ANNOTATE, [item_for(store, "pkg/mod0.py")])
    assert attached.annotation is None


def test_attach_skips_other_model_or_prompt(store: Store) -> None:
    corpus(store, 1)
    annotate_all(store, replace(ANNOTATE, model="other"))
    annotate_all(store, replace(ANNOTATE, prompt_version=ANNOTATE.prompt_version + 1))
    (attached,) = attach_annotations(store, ANNOTATE, [item_for(store, "pkg/mod0.py")])
    assert attached.annotation is None


def test_attach_skips_changed_span(store: Store) -> None:
    corpus(store, 1)
    annotate_all(store)
    item = item_for(store, "pkg/mod0.py")
    narrowed = replace(item, cand=replace(item.cand, end_byte=item.cand.end_byte - 1))
    (attached,) = attach_annotations(store, ANNOTATE, [narrowed])
    assert attached.annotation is None


def test_ceiling_with_annotations_uses_summaries(store: Store) -> None:
    corpus(store, 6)
    annotate_all(store)
    budget = 1000
    result = search(store, Config(), "handler value", SearchOptions(max_tokens=budget))
    assert bytes35(result.text) <= budget
    summaries = [item for item in result.pack["items"] if item["rep"] == "summary"]
    assert summaries
    assert result.pack["budget"]["truncated"] is True
    for item in summaries:
        assert item["synthesized"] is True
        assert PURPOSE in item["text"]
        assert resolve_ref(store, item["ref"]) is not None


def test_no_summaries_without_annotations(store: Store) -> None:
    corpus(store, 6)
    result = search(store, Config(), "handler value", SearchOptions(max_tokens=1000))
    assert all(item["rep"] != "summary" for item in result.pack["items"])


def test_no_summaries_when_lossless_fits(store: Store) -> None:
    corpus(store, 1, lines=2)
    annotate_all(store)
    result = search(store, Config(), "handler value", SearchOptions(max_tokens=6000))
    assert all(item["rep"] == "source" for item in result.pack["items"])
    assert result.pack["budget"]["truncated"] is False


def test_lossy_path_never_calls_a_model(store: Store, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("model called during compaction")

    for name in ("call", "embed", "annotate"):
        monkeypatch.setattr(ModelSlots, name, forbidden)
    corpus(store, 6)
    annotate_all(store)
    slots = ModelSlots.from_config(Config().models)
    ctx = replace(make_ctx(store, max_tokens=MIN_MAX_TOKENS + 400), models=slots)
    cands = [cand for number in range(6) for cand in candidates(store, f"pkg/mod{number}.py")]
    scored = [replace(cand, scores=Scores(final=0.1 * (n + 1))) for n, cand in enumerate(cands)]
    result = AssembleStage().assemble(ctx, CandidateSet.of(scored))
    assert any(item["rep"] == "summary" for item in result.pack["items"])


@settings(max_examples=20, deadline=None)
@given(
    files=st.integers(1, 6),
    lines=st.integers(5, 40),
    budget=st.integers(MIN_MAX_TOKENS, 1500),
    annotated=st.booleans(),
)
def test_synthesized_refs_resolve_to_exact_source(
    files: int, lines: int, budget: int, annotated: bool
) -> None:
    with tempfile.TemporaryDirectory() as directory, Store.open(Path(directory)) as store:
        corpus(store, files, lines)
        if annotated:
            annotate_all(store)
        result = search(store, Config(), "handler value", SearchOptions(max_tokens=budget))
        assert bytes35(result.text) <= budget
        for item in result.pack["items"]:
            if item["rep"] != "source":
                body = resolve_ref(store, item["ref"])
                source = (store.root / item["path"]).read_bytes()
                assert body is not None and body in source
            assert item["synthesized"] == (item["rep"] == "summary")
        for entry in result.pack["omitted"]:
            if "count" not in entry:
                assert resolve_ref(store, entry["ref"]) is not None


CONFIG_JSON = (
    '{"server": {"host": "a-very-long-host-name.internal.example.com", "port": 8080,'
    ' "tls": {"cert": "/etc/ssl/certs/a-long-certificate-name.pem",'
    ' "key": "/etc/ssl/private/a-long-key-name.key"}},'
    ' "limits": {"requests": 1000, "burst": 50}}\n'
)
BULKY_JSON = CONFIG_JSON.replace("a-very-long-host-name", "a-very-long-host-name" * 8)
UTILITY = (
    "class Point:\n"
    "    def __init__(self, x):\n"
    "        self.x = x\n"
    "        self.y = x\n"
    "        self.z = x\n\n"
    "    def __repr__(self):\n"
    "        return f'Point({self.x}, {self.y}, {self.z}, extra, padding, more)'\n\n"
    "    def __eq__(self, other):\n"
    "        return self.x == other.x and self.y == other.y and self.z == other.z\n\n"
    "    def __str__(self):\n"
    "        return f'Point at {self.x} then {self.y} then {self.z} in the plane'\n"
)
REEXPORTS = "from .alpha import Alpha\nfrom .beta import Beta\nfrom .gamma import Gamma\n"


def folded_item(store: Store, path: str, name: str | None = None) -> PackItem:
    item = item_for(store, path)
    cands = [c for c in candidates(store, path) if name is None or c.qualified_name == name]
    source = (store.root / path).read_bytes()
    cand = cands[0]
    base = replace(
        item, cand=cand, text=source[cand.start_byte : cand.end_byte].decode(), signature=None
    )
    (result,) = prepare_lossy(store, ANNOTATE, [base])
    return result


def test_key_paths_golden(store: Store) -> None:
    ingest(store, "conf/app.json", CONFIG_JSON)
    item = folded_item(store, "conf/app.json")
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "key_paths"
    after = rung.apply(item)
    assert after is not None and after.rep == "schema"
    assert "server.port number" in after.text
    assert "example.com" not in after.text and "8080" not in after.text
    assert after.to_wire()["synthesized"] is False


def test_member_list_golden(store: Store) -> None:
    ingest(store, "pkg/point.py", UTILITY)
    item = folded_item(store, "pkg/point.py", "Point")
    assert item.folds.members is not None
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "member_list"
    after = rung.apply(item)
    assert after is not None and after.rep == "signature"
    assert after.text.startswith("class Point")
    assert "__repr__" in after.text and "__eq__" in after.text
    assert "return" not in after.text


def test_reexports_fold(store: Store) -> None:
    ingest(store, "pkg/__init__.py", REEXPORTS)
    item = folded_item(store, "pkg/__init__.py")
    assert item.folds.reexports is not None
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "fold_reexports"
    after = rung.apply(item)
    assert after is not None and "alpha" in after.text and "Gamma" not in after.text


def test_folds_absent_for_ordinary_code(store: Store) -> None:
    corpus(store, 1)
    item = folded_item(store, "pkg/mod0.py")
    assert item.folds.key_paths is None
    assert item.folds.members is None
    assert item.folds.reexports is None


def test_structural_rungs_end_to_end_round_trip(store: Store) -> None:
    paths: list[str] = []
    for number in range(6):
        for template, text in (
            ("conf{}/app.json", BULKY_JSON),
            ("pkg{}/point.py", UTILITY),
            ("pkg{}/__init__.py", REEXPORTS),
        ):
            path = template.format(number)
            ingest(store, path, text)
            paths.append(path)
    cands = [
        replace(cand, scores=Scores(final=0.01 * (n + 1)))
        for n, cand in enumerate(c for path in paths for c in candidates(store, path))
    ]
    reps: set[str] = set()
    for budget in range(MIN_MAX_TOKENS, 1600, 40):
        result = AssembleStage().assemble(
            make_ctx(store, max_tokens=budget), CandidateSet.of(cands)
        )
        assert bytes35(result.text) <= budget
        for item in result.pack["items"]:
            reps.add(item["rep"])
            if item["rep"] != "source":
                assert resolve_ref(store, item["ref"]) is not None
    assert {"schema", "signature"} <= reps
