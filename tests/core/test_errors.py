from contextweave.core.errors import Envelope, OmittedEntry, StageError, cap_omitted

KEY = "0123456789ab"


def entry(i: int) -> OmittedEntry:
    return OmittedEntry(ref=f"r{i}", qualified_name=f"q{i}", path="p", reason="budget")


def test_guard_converts_exception() -> None:
    env = Envelope()
    with env.guard("embed"):
        raise RuntimeError("model down")
    assert env.errors == [{"code": "stage_failed", "message": "model down", "stage": "embed"}]
    assert env.degraded == [{"stage": "embed", "reason": "model down"}]


def test_guard_stage_error_device() -> None:
    env = Envelope()
    with env.guard("classify"):
        raise StageError("no_device", "cuda missing", device_tried="cuda")
    assert env.errors[0]["code"] == "no_device"
    assert env.degraded[0].get("device_tried") == "cuda"


def test_guard_passes_clean_block() -> None:
    env = Envelope()
    with env.guard("ok"):
        pass
    assert env.errors == []
    assert env.degraded == []


def test_omitted_under_cap_unchanged() -> None:
    entries = [entry(i) for i in range(50)]
    assert cap_omitted(entries, omitted_max=50, key=KEY, pack_id="p1") == entries


def test_omitted_overflow_tail() -> None:
    entries = [entry(i) for i in range(60)]
    capped = cap_omitted(entries, omitted_max=50, key=KEY, pack_id="p1")
    assert len(capped) == 50
    assert capped[:49] == entries[:49]
    assert capped[-1] == {
        "reason": "overflow",
        "count": 11,
        "ref": f"cw://{KEY}/pack/p1/omitted",
    }
