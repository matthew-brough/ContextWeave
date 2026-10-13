import signal
import threading

import pytest

from contextweave.core.interrupt import deferred_interrupt


def test_ctrl_c_is_held_until_the_block_ends() -> None:
    finished: list[bool] = []
    with pytest.raises(KeyboardInterrupt), deferred_interrupt():
        signal.raise_signal(signal.SIGINT)
        finished.append(True)
    assert finished == [True]
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_quiet_block_raises_nothing() -> None:
    with deferred_interrupt():
        pass
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


def test_held_signal_goes_to_the_previous_handler() -> None:
    seen: list[int] = []
    previous = signal.signal(signal.SIGINT, lambda signum, _frame: seen.append(signum))
    try:
        with deferred_interrupt():
            signal.raise_signal(signal.SIGINT)
            assert seen == []
        assert seen == [signal.SIGINT]
    finally:
        signal.signal(signal.SIGINT, previous)


def test_off_main_thread_it_is_a_no_op() -> None:
    errors: list[BaseException] = []

    def work() -> None:
        try:
            with deferred_interrupt():
                pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert errors == []
