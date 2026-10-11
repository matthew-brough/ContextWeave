from collections.abc import Callable

from contextweave.daemon.models import ModelRuntime


def skip_rate_warnings(models: ModelRuntime | None) -> Callable[[], tuple[str, ...]]:
    """Warning lines for slots over models.skip_warn (V26) or models.slow_warn (calls past warn_ms)."""

    def warnings() -> tuple[str, ...]:
        if models is None:
            return ()
        lines: list[str] = []
        for status in models.slots.status():
            if status.warn:
                lines.append(
                    f"warning: skip_rate slot {status.slot.value} {status.skip_rate:.0%}"
                    f" over last {status.calls_in_window} calls"
                )
            if status.slow_warn:
                lines.append(
                    f"warning: slow_rate slot {status.slot.value} {status.slow_rate:.0%}"
                    " of calls past warn_ms"
                )
        return tuple(lines)

    return warnings
