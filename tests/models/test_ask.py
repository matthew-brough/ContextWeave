from fake_decision import FakeDecision, fake_slots
from models_support import Clock

from contextweave.models.calibration import (
    Calibration,
    CalibrationKey,
)
from contextweave.models.decision import Question, QuestionKind
from contextweave.models.providers import Device
from contextweave.models.slots import Done, Skipped, Slot

NOUL = Question("answers_query", "q", QuestionKind.NOUL, ("yes", "no"))


def test_ask_returns_answers_per_state() -> None:
    outcome = fake_slots(FakeDecision()).ask(["good", "bad"], [NOUL], stage="gate")
    assert isinstance(outcome, Done)
    assert [round(row.value("answers_query"), 2) for row in outcome.value] == [0.95, 0.05]


def test_ask_applies_fitted_temperature() -> None:
    key = CalibrationKey("kev-latest", "kev-0.8b-int8", 2)
    slots = fake_slots(FakeDecision(), calibration=Calibration({key: {"answers_query": 4.0}}))
    outcome = slots.ask(["good"], [NOUL], stage="gate")
    assert isinstance(outcome, Done)
    assert outcome.value[0].value("answers_query") < 0.95


def test_transport_failure_skips_and_marks_device_down() -> None:
    slots = fake_slots(FakeDecision(down=True))
    outcome = slots.ask(["good"], [NOUL], stage="gate")
    assert isinstance(outcome, Skipped)
    assert outcome.degraded["stage"] == "gate"
    assert slots.healths[Slot.CLASSIFY].active is not Device.GPU


def test_wrong_row_count_skips_without_raising() -> None:
    outcome = fake_slots(FakeDecision(short_rows=True)).ask(["good", "bad"], [NOUL], stage="gate")
    assert isinstance(outcome, Skipped)


def test_deadline_overrun_skips() -> None:
    clock = Clock()
    client = FakeDecision()
    slots = fake_slots(client, clock=clock)
    original = client.yes

    def slow(state: str) -> float:
        clock.now += 60
        return original(state)

    client.yes = slow
    assert isinstance(slots.ask(["good"], [NOUL], stage="gate"), Skipped)


def test_unhealthy_slot_skips_without_calling_the_client() -> None:
    clock = Clock()
    client = FakeDecision(down=True)
    slots = fake_slots(client, clock=clock)
    for _ in range(4):
        slots.ask(["good"], [NOUL], stage="gate")
    client.down = False
    client.calls.clear()
    outcome = slots.ask(["good"], [NOUL], stage="gate")
    assert isinstance(outcome, Skipped)
    assert client.calls == []
