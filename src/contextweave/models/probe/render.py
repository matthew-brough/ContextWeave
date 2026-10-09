import json

from contextweave.models.probe.report import Line, ProbeReport, report_json

HEADERS = ("probe", "device", "status", "result")


def render_table(report: ProbeReport) -> str:
    """Aligned human table, recommendations below."""
    rows = [
        HEADERS,
        *((line.probe, line.device, line.status.value, line.summary) for line in report.lines),
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    host = report.host
    gpu = host.gpu if host.gpu_total_mib is None else f"{host.gpu} ({host.gpu_total_mib:.0f} MiB)"
    out = [f"host: cpu {host.cpu} x{host.cores}; gpu {gpu}", ""]
    out += [
        "  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row[:3])) + "  " + row[3]
        for row in rows
    ]
    if report.recommendations:
        out.append("")
        out.append("decides:")
        out.extend(f"  - {item}" for item in report.recommendations)
    if not report.probed:
        out.append("")
        out.append("nothing could be probed")
    return "\n".join(out)


def render_json(report: ProbeReport) -> str:
    return json.dumps(report_json(report), indent=2, sort_keys=True, ensure_ascii=False)


def render_record(report: ProbeReport, date: str) -> str:
    """Markdown snippet for the user to paste under SPEC §B."""
    host = report.host
    gpu = host.gpu if host.gpu_total_mib is None else f"{host.gpu}, {host.gpu_total_mib:.0f} MiB"
    out = [
        f"### T10 model probe {date}",
        "",
        f"host: {host.cpu} x{host.cores}; gpu: {gpu}",
        "",
        "| probe | device | status | result |",
        "|---|---|---|---|",
    ]
    out.extend(_row(line) for line in report.lines)
    detail = [line for line in report.lines if line.facts or line.metrics]
    if detail:
        out.extend(["", "measurements:"])
        for line in detail:
            out.append(f"- `{line.probe}` {line.device}: {_measurements(line)}")
    out.extend(["", "decides:"])
    out.extend(f"- {item}" for item in report.recommendations)
    if not report.recommendations:
        out.append("- nothing measured")
    out.append("")
    return "\n".join(out)


def _row(line: Line) -> str:
    summary = line.summary.replace("|", "\\|")
    return f"| {line.probe} | {line.device} | {line.status.value} | {summary} |"


def _measurements(line: Line) -> str:
    metrics = [f"{key}={value:.4g}" for key, value in line.metrics.items()]
    facts = [f"{key}={value}" for key, value in line.facts.items()]
    return "; ".join([*metrics, *facts])
