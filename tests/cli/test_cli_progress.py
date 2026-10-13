import io
from collections.abc import Iterator

from contextweave.cli_commands.indexing import ProgressPrinter, progress_text
from contextweave.index.progress import Progress


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def ticks(*values: float) -> Iterator[float]:
    return iter(values)


def test_text_shows_count_percent_and_eta() -> None:
    assert progress_text(Progress("scan", 3, 12)) == "scanning files 3/12 (25%)"
    assert progress_text(Progress("annotate", 1, 4, 65.0)) == "annotating 1/4 (25%), eta 1m05s"
    assert progress_text(Progress("edges", 0, 1)) == "resolving edges..."
    assert progress_text(Progress("edges", 1, 1)) == "resolving edges done"


def test_piped_output_throttles_but_keeps_starts_and_finishes() -> None:
    stream = io.StringIO()
    clock = ticks(0.0, 1.0, 6.0, 7.0, 8.0, 9.0)
    printer = ProgressPrinter(stream, clock=lambda: next(clock))
    for done in (0, 1, 2, 3):
        printer(Progress("scan", done, 4))
    printer(Progress("scan", 4, 4))
    printer(Progress("edges", 0, 1))
    assert stream.getvalue().splitlines() == [
        "scanning files 0/4 (0%)",
        "scanning files 2/4 (50%)",
        "scanning files 4/4 (100%)",
        "resolving edges...",
    ]


def test_terminal_redraws_in_place_and_closes_open_line() -> None:
    stream = Tty()
    clock = ticks(0.0, 1.0)
    printer = ProgressPrinter(stream, clock=lambda: next(clock))
    printer(Progress("embed code", 0, 2))
    printer(Progress("embed code", 1, 2))
    printer.close()
    assert stream.getvalue() == (
        "\r\x1b[Kembedding code 0/2 (0%)\r\x1b[Kembedding code 1/2 (50%)\n"
    )


def test_empty_stage_prints_nothing() -> None:
    stream = io.StringIO()
    ProgressPrinter(stream)(Progress("embed annot", 0, 0))
    assert stream.getvalue() == ""
