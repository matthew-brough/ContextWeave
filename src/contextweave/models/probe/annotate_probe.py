from contextweave.models.errors import ModelParseError
from contextweave.models.probe.context import ProbeEnv
from contextweave.models.probe.ollama_tags import TagCatalog
from contextweave.models.probe.report import Line, Status, missing_summary
from contextweave.models.probe.timing import rate_per_s, summarize
from contextweave.models.probe.util import Failure, attempt, gpu_used_mib
from contextweave.models.prompt_store import PromptError, PromptTemplate, load_prompt
from contextweave.models.providers import AnnotateClient, Device, Message

PROMPT_NAME = "annotate"
OLLAMA = "ollama"


def probe_annotate(env: ProbeEnv, catalog: TagCatalog | None) -> list[Line]:
    """Tag presence and annotate rate (chunks/s) per device for the annotate slot."""
    config = env.models.annotate
    if config.provider == "none":
        return [Line("annotate", "-", Status.SKIPPED, "provider is none")]
    lines: list[Line] = []
    if config.provider == OLLAMA:
        if catalog is None:
            return [
                Line(
                    "annotate.tag",
                    "-",
                    Status.UNAVAILABLE,
                    f"ollama unreachable at {env.ollama_url}",
                )
            ]
        if not catalog.has(config.model):
            return [Line("annotate.tag", "-", Status.MISSING, missing_summary(config.model))]
        lines.append(Line("annotate.tag", "-", Status.OK, f"present: {config.model}"))
    try:
        prompt = load_prompt(PROMPT_NAME, config.prompt_version)
    except PromptError as exc:
        return [*lines, Line("annotate.rate", "-", Status.UNAVAILABLE, str(exc))]
    client = env.slots.clients.annotate
    for device in _devices(env, client):
        moved = attempt(lambda device=device: client.set_device(device))
        if isinstance(moved, Failure):
            lines.append(Line("annotate.rate", device.value, Status.UNAVAILABLE, moved.reason))
            continue
        lines.append(_rate_line(env, client, prompt, device))
    return lines


def _devices(env: ProbeEnv, client: AnnotateClient) -> list[Device]:
    wanted = [device for device in env.options.devices if device in client.devices]
    return wanted or [device for device in client.devices if device is not Device.NONE]


def _messages(env: ProbeEnv, prompt: PromptTemplate, index: int) -> tuple[Message, ...]:
    chunks = env.probe_set.chunks
    chunk = chunks[index % len(chunks)]
    return prompt.render(path=chunk.path, kind=chunk.kind, name=chunk.name, code=chunk.code)


def _annotate_once(
    env: ProbeEnv, client: AnnotateClient, prompt: PromptTemplate, index: int
) -> tuple[float, bool]:
    messages = _messages(env, prompt, index)
    started = env.clock()
    try:
        client.annotate(messages, prompt.schema)
    except ModelParseError:
        return (env.clock() - started) * 1000, True
    return (env.clock() - started) * 1000, False


def _rate_line(
    env: ProbeEnv, client: AnnotateClient, prompt: PromptTemplate, device: Device
) -> Line:
    name = device.value
    samples: list[float] = []
    parse_failures = 0
    for index in range(env.options.annotate_chunks):
        timed = attempt(lambda index=index: _annotate_once(env, client, prompt, index))
        if isinstance(timed, Failure):
            if not samples:
                return Line("annotate.rate", name, Status.UNAVAILABLE, timed.reason)
            break
        samples.append(timed[0])
        parse_failures += 1 if timed[1] else 0
    if not samples:
        return Line("annotate.rate", name, Status.UNAVAILABLE, "no chunks to annotate")
    latency = summarize(samples)
    total_ms = sum(samples)
    rate = rate_per_s(len(samples), total_ms)
    deadline = env.configured_deadline_ms
    over = sum(1 for sample in samples if sample > deadline)
    metrics = {
        "chunks": float(len(samples)),
        "chunks_per_s": rate,
        "p50_ms": latency.p50_ms,
        "p95_ms": latency.p95_ms,
        "max_ms": latency.max_ms,
        "parse_failures": float(parse_failures),
        "over_deadline": float(over),
        "deadline_ms": float(deadline),
    }
    gpu_used = gpu_used_mib(env.host)
    if gpu_used is not None:
        metrics["gpu_used_mib"] = gpu_used
    summary = (
        f"{rate:.2f} chunks/s, p50 {latency.p50_ms:.0f} ms, p95 {latency.p95_ms:.0f} ms, "
        f"{over}/{len(samples)} over deadline_ms {deadline}, {parse_failures} parse failures"
    )
    status = Status.WARN if over or parse_failures else Status.OK
    return Line("annotate.rate", name, status, summary, metrics)
