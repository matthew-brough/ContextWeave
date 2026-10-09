from collections.abc import Sequence
from dataclasses import replace

from pack_support import make_cand
from retrieve_support import ingest, make_ctx

from contextweave.core.budget import measure_payload
from contextweave.core.config import BudgetConfig, Config
from contextweave.core.refs import PartAnchor, SymbolAnchor, parse_ref
from contextweave.index.store import Store
from contextweave.retrieve.assemble import MIN_MAX_TOKENS, AssembleStage
from contextweave.retrieve.candidates import Candidate, CandidateSet, Scores
from contextweave.retrieve.hydrate import load_file_candidates
from contextweave.retrieve.ladder import LADDER, next_rung
from contextweave.retrieve.pack import PackItem

BODY = "    total = 0\n" * 40


def seed(store: Store) -> None:
    ingest(
        store, "a.py", f"def first():\n{BODY}    return 1\n\n\ndef second():\n{BODY}    return 2\n"
    )
    ingest(store, "b.py", f"def third():\n{BODY}    return 3\n")


def with_final(cands: Sequence[Candidate], finals: list[float]) -> CandidateSet:
    return CandidateSet.of(
        replace(cand, scores=Scores(final=final)) for cand, final in zip(cands, finals, strict=True)
    )


def all_cands(store: Store) -> tuple[Candidate, ...]:
    return (*load_file_candidates(store, "a.py"), *load_file_candidates(store, "b.py"))


def test_groups_by_file_ordered_by_max_final_then_start_byte(store: Store) -> None:
    seed(store)
    cands = all_cands(store)
    assert len(cands) == 3
    ranked = with_final(cands, [0.2, 0.3, 0.9])
    result = AssembleStage().assemble(make_ctx(store, "x"), ranked)
    paths = [item["path"] for item in result.pack["items"]]
    assert paths == ["b.py", "a.py", "a.py"]
    starts = [item["start_line"] for item in result.pack["items"] if item["path"] == "a.py"]
    assert starts == sorted(starts)


def test_item_text_is_source_slice_with_one_based_lines_and_parseable_ref(store: Store) -> None:
    seed(store)
    ranked = with_final(all_cands(store), [0.5, 0.5, 0.5])
    ctx = make_ctx(store, "x")
    result = AssembleStage().assemble(ctx, ranked)
    source = (store.root / "a.py").read_bytes()
    first = next(i for i in result.pack["items"] if i["qualified_name"] == "first")
    cand = next(c for c in ranked if c.qualified_name == "first")
    assert first["text"] == source[cand.start_byte : cand.end_byte].decode()
    assert first["start_line"] == 1
    assert first["end_line"] == source[: cand.end_byte - 1].count(b"\n") + 1
    ref = parse_ref(first["ref"])
    assert ref.key == ctx.project_key
    assert ref.anchor == SymbolAnchor("first")
    assert ref.path == "a.py"


def test_pack_shape_and_budget_counts_text(store: Store) -> None:
    seed(store)
    result = AssembleStage().assemble(make_ctx(store, "x"), with_final(all_cands(store), [1, 1, 1]))
    pack = result.pack
    assert set(pack) == {
        "pack_id",
        "plan",
        "coverage",
        "items",
        "omitted",
        "budget",
        "degraded",
        "errors",
    }
    assert pack["budget"]["ceiling"] == 6000
    assert pack["budget"]["truncated"] is False
    assert "hint" not in pack["budget"]
    assert pack["budget"]["served_tokens"] >= measure_payload(pack, result.text)
    assert pack["plan"]["devices"] == {"embed": "none", "classify": "none"}


def test_over_ceiling_trims_lowest_final_to_refs(store: Store) -> None:
    seed(store)
    ranked = with_final(all_cands(store), [0.9, 0.1, 0.5])
    ctx = replace(make_ctx(store, "x"), max_tokens=MIN_MAX_TOKENS)
    result = AssembleStage().assemble(ctx, ranked)
    pack = result.pack
    assert pack["budget"]["truncated"] is True
    assert "scope" in pack["budget"].get("hint", "")
    assert measure_payload(pack, result.text) <= MIN_MAX_TOKENS
    served = {item["ref"] for item in pack["items"]}
    omitted = {entry["ref"] for entry in result.omitted}
    assert omitted
    assert served.isdisjoint(omitted)
    for ref in omitted:
        parse_ref(ref)
    second = next(c for c in ranked if c.qualified_name == "second")
    assert any(entry["qualified_name"] == "second" for entry in result.omitted)
    assert second.scores.final == 0.1


def test_signature_rung_precedes_ref_and_is_flagged_by_rep(store: Store) -> None:
    seed(store)
    ranked = with_final(all_cands(store), [0.9, 0.1, 0.5])
    ctx = replace(make_ctx(store, "x"), max_tokens=1100)
    result = AssembleStage().assemble(ctx, ranked)
    reps = {item["qualified_name"]: item["rep"] for item in result.pack["items"]}
    assert "signature" in reps.values()
    sig = next(i for i in result.pack["items"] if i["rep"] == "signature")
    assert sig["text"].startswith("def ")
    assert sig["synthesized"] is False
    assert measure_payload(result.pack, result.text) <= 1100


def test_critical_items_stop_at_signature_before_ref(store: Store) -> None:
    seed(store)
    cands = all_cands(store)
    pinned = replace(cands[0], critical=True, scores=Scores(final=0.0))
    ranked = CandidateSet.of(
        [
            pinned,
            replace(cands[1], scores=Scores(final=0.5)),
            replace(cands[2], scores=Scores(final=0.5)),
        ]
    )
    ctx = replace(make_ctx(store, "x"), max_tokens=600)
    result = AssembleStage().assemble(ctx, ranked)
    kept = {item["qualified_name"]: item["rep"] for item in result.pack["items"]}
    assert kept.get("first") == "signature"
    assert result.pack["budget"]["truncated"] is True


def test_omitted_overflow_tail_and_full_list(store: Store) -> None:
    for index in range(12):
        ingest(store, f"m{index}.py", f"def f{index}():\n{BODY}    return {index}\n")
    cands = tuple(c for i in range(12) for c in load_file_candidates(store, f"m{i}.py"))
    ranked = with_final(cands, [i / 100 for i in range(12)])
    config = replace(Config(), budget=BudgetConfig(omitted_max=4))
    ctx = replace(make_ctx(store, "x", config=config), max_tokens=MIN_MAX_TOKENS)
    result = AssembleStage().assemble(ctx, ranked)
    entries = result.pack["omitted"]
    assert len(entries) <= 4
    assert len(result.omitted) + len(result.pack["items"]) == 12
    if len(result.omitted) > 4:
        tail = entries[-1]
        assert tail["reason"] == "overflow"
        assert tail["ref"].endswith(f"/pack/{result.pack['pack_id']}/omitted")
        assert "count" in tail
        assert tail["count"] == len(result.omitted) - (len(entries) - 1)


def test_tiny_max_tokens_is_raised_with_error(store: Store) -> None:
    seed(store)
    ctx = replace(make_ctx(store, "x"), max_tokens=5)
    result = AssembleStage().assemble(ctx, with_final(all_cands(store), [1, 1, 1]))
    assert result.pack["budget"]["ceiling"] == MIN_MAX_TOKENS
    assert [e["code"] for e in result.pack["errors"]] == ["max_tokens_raised"]
    assert measure_payload(result.pack, result.text) <= MIN_MAX_TOKENS


def test_vanished_source_is_degraded_not_raised(store: Store) -> None:
    seed(store)
    ranked = with_final(all_cands(store), [1, 1, 1])
    (store.root / "b.py").unlink()
    ctx = make_ctx(store, "x")
    result = AssembleStage().assemble(ctx, ranked)
    assert {i["path"] for i in result.pack["items"]} == {"a.py"}
    assert ctx.envelope.degraded[0]["reason"].startswith("source_unavailable")


def test_empty_candidates_give_valid_pack(store: Store) -> None:
    result = AssembleStage().assemble(make_ctx(store, "x"), CandidateSet())
    assert result.pack["items"] == []
    assert result.pack["omitted"] == []


def test_split_part_ref_uses_part_anchor(store: Store) -> None:
    ingest(store, "big.py", "def big():\n" + "    x = 1\n" * 400)
    cands = load_file_candidates(store, "big.py")
    parts = [c for c in cands if isinstance(c.anchor, PartAnchor)]
    if parts:
        ranked = with_final(parts, [0.5] * len(parts))
        result = AssembleStage().assemble(replace(make_ctx(store, "x"), max_tokens=100000), ranked)
        assert all(isinstance(parse_ref(i["ref"]).anchor, PartAnchor) for i in result.pack["items"])


def pack_item(**changes: object) -> PackItem:
    base = PackItem(
        cand=make_cand(1, name="f"),
        ref="cw://x",
        rep="source",
        text="def f():\n    return 1\n",
        signature="def f()",
        start_line=1,
        end_line=2,
        oversize=False,
    )
    return replace(base, **changes)


def test_ladder_order_and_critical_floor() -> None:
    assert [rung.name for rung in LADDER] == ["signature", "ref"]
    item = pack_item()
    rung = next_rung(item, False)
    assert rung is not None and rung.name == "signature"
    signed = item.with_signature_rep()
    assert signed.text == "def f()"
    ref_rung = next_rung(signed, False)
    assert ref_rung is not None and ref_rung.name == "ref"
    critical = replace(signed, cand=replace(signed.cand, critical=True))
    assert next_rung(critical, False) is None
    final = next_rung(critical, True)
    assert final is not None and final.apply(critical) is None


def test_ladder_skips_signature_without_gain() -> None:
    nameless = pack_item(signature=None)
    rung = next_rung(nameless, False)
    assert rung is not None and rung.name == "ref"
    longer = pack_item(signature="def f():\n    return 1\n  # longer than text")
    other = next_rung(longer, False)
    assert other is not None and other.name == "ref"
