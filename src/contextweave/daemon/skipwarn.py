from collections.abc import Callable

from contextweave.daemon.models import ModelRuntime


def skip_rate_warnings(models: ModelRuntime | None) -> Callable[[], tuple[str, ...]]:
    """Warning lines for model slots whose rolling skip rate exceeds models.skip_warn (V26)."""

    def warnings() -> tuple[str, ...]:
        if models is None:
            return ()
        return tuple(
            f"warning: skip_rate slot {status.slot.value} {status.skip_rate:.0%}"
            f" over last {status.calls_in_window} calls"
            for status in models.slots.status()
            if status.warn
        )

    return warnings
