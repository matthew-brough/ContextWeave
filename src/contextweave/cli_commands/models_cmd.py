import argparse
from datetime import date
from pathlib import Path

from contextweave.cli_commands.common import (
    EXIT_FAILED,
    EXIT_USAGE,
    CommandContext,
    SubparserRegistry,
    complain,
    say,
)
from contextweave.core.config import ConfigError, load_config
from contextweave.index.root import discover_root
from contextweave.models.probe.context import (
    ALL_DEVICES,
    ALL_SLOTS,
    DEFAULT_PIPELINE_BUDGET_MS,
    DEFAULT_PROBE_DEADLINE_MS,
    DEFAULT_VRAM_MIB,
    ClassifyCandidate,
    ProbeOptions,
)
from contextweave.models.probe.render import render_json, render_record, render_table
from contextweave.models.probe.runner import run_probe
from contextweave.models.providers import Device, Slot

DEFAULTS = ProbeOptions()
CANDIDATE_KEYS = ("family", "rung", "build", "model", "gpu", "cpu")


class CandidateSpecError(ValueError):
    """A --candidate value is malformed."""


def add_parser(commands: SubparserRegistry) -> None:
    parser = commands.add_parser("models", help="Model slot tooling")
    actions = parser.add_subparsers(dest="models_command", required=True)
    probe = actions.add_parser("probe", help="Time each model slot on this hardware, no downloads")
    probe.add_argument("--slot", action="append", choices=[slot.value for slot in ALL_SLOTS])
    probe.add_argument("--device", action="append", choices=[d.value for d in ALL_DEVICES])
    probe.add_argument("--json", action="store_true", dest="as_json")
    probe.add_argument(
        "--record", type=Path, metavar="PATH", help="write a markdown snippet of the run"
    )
    probe.add_argument("--runs", type=int, default=DEFAULTS.runs)
    probe.add_argument("--annotate-chunks", type=int, default=DEFAULTS.annotate_chunks)
    probe.add_argument("--classify-runs", type=int, default=DEFAULTS.classify_runs)
    probe.add_argument("--vram-mib", type=int, default=DEFAULT_VRAM_MIB)
    probe.add_argument("--pipeline-budget-ms", type=int, default=DEFAULT_PIPELINE_BUDGET_MS)
    probe.add_argument("--probe-deadline-ms", type=int, default=DEFAULT_PROBE_DEADLINE_MS)
    probe.add_argument(
        "--candidate",
        action="append",
        metavar="SPEC",
        help="classify ladder rung: family=kev,rung=1,build=q4_k_m,gpu=URL,cpu=URL[,model=ID]",
    )


def parse_candidate(spec: str, default_model: str) -> ClassifyCandidate:
    """Parse `key=value` pairs into a ladder candidate."""
    values: dict[str, str] = {}
    for part in spec.split(","):
        key, separator, value = part.partition("=")
        if not separator or key not in CANDIDATE_KEYS:
            raise CandidateSpecError(
                f"bad candidate field '{part}', keys: {', '.join(CANDIDATE_KEYS)}"
            )
        values[key] = value
    if "family" not in values or "rung" not in values:
        raise CandidateSpecError("candidate needs family= and rung=")
    try:
        rung = int(values["rung"])
    except ValueError as exc:
        raise CandidateSpecError(f"rung '{values['rung']}' is not an integer") from exc
    return ClassifyCandidate(
        values["family"],
        rung,
        values.get("build", "unnamed"),
        values.get("model", default_model),
        values.get("gpu", ""),
        values.get("cpu", ""),
    )


def run(args: argparse.Namespace, ctx: CommandContext) -> int:
    try:
        root = discover_root(ctx.cwd)
    except OSError, RuntimeError:
        root = None
    try:
        config = load_config(root, strict=False, env=ctx.env).config
    except ConfigError as exc:
        complain("models", str(exc))
        return EXIT_FAILED
    raw_specs: list[str] = args.candidate or []
    try:
        candidates = tuple(
            parse_candidate(spec, config.models.classify.model) for spec in raw_specs
        )
    except CandidateSpecError as exc:
        complain("models probe", str(exc))
        return EXIT_USAGE
    slots: list[str] = args.slot or [slot.value for slot in ALL_SLOTS]
    devices: list[str] = args.device or [device.value for device in ALL_DEVICES]
    options = ProbeOptions(
        slots=tuple(Slot(name) for name in slots),
        devices=tuple(Device(name) for name in devices),
        runs=args.runs,
        annotate_chunks=args.annotate_chunks,
        classify_runs=args.classify_runs,
        vram_mib=args.vram_mib,
        pipeline_budget_ms=args.pipeline_budget_ms,
        candidates=candidates,
    )
    report = run_probe(
        config,
        options,
        probe_deadline_ms=args.probe_deadline_ms,
    )
    say(render_json(report) if args.as_json else render_table(report))
    record: Path | None = args.record
    if record is not None:
        try:
            record.write_text(render_record(report, date.today().isoformat()), encoding="utf-8")
        except OSError as exc:
            complain("models probe", f"cannot write {record}: {exc}")
            return EXIT_FAILED
        if not args.as_json:
            say(f"wrote {record}")
    return report.exit_code
