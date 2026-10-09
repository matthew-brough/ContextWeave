from contextweave.cli_commands.common import EXIT_OK, SubparserRegistry, say
from contextweave.lang.registry import LANGS, load_language


def add_parser(commands: SubparserRegistry) -> None:
    commands.add_parser("languages", help="List supported languages")


def run() -> int:
    for spec in LANGS:
        available = "ok" if load_language(spec.lang) is not None else "no grammar"
        names = ",".join((*spec.exts, *spec.filenames))
        say(f"{spec.lang:<11} {spec.group:<10} {spec.profile:<8} {available:<10} {names}")
    return EXIT_OK
