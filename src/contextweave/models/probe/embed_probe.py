import math
from collections.abc import Sequence

from contextweave.models.probe.context import ProbeEnv
from contextweave.models.probe.ollama_tags import TagCatalog
from contextweave.models.probe.report import Line, Status, missing_summary
from contextweave.models.probe.timing import rate_per_s, sample_ms, summarize, time_ms
from contextweave.models.probe.util import Failure, attempt, gpu_used_mib
from contextweave.models.providers import Device, EmbedClient, EmbedKind

DOC_BATCH = 16
QUERY_SAMPLE = "parse a toml config file into dataclasses"
OLLAMA = "ollama"


def probe_embed(env: ProbeEnv, catalog: TagCatalog | None) -> list[Line]:
    """Tag presence, prefixes, query latency and doc throughput per device for the embed slot."""
    config = env.models.embed
    if config.provider == "none":
        return [Line("embed", "-", Status.SKIPPED, "provider is none")]
    lines: list[Line] = []
    if config.provider == OLLAMA:
        tag_line = _tag_line(config.model, catalog, env.ollama_url)
        lines.append(tag_line)
        if tag_line.status is not Status.OK:
            return lines
    lines.append(_card_prefix_line(env))
    client = env.slots.clients.embed
    effect_done = False
    for device in _devices(env, client):
        moved = attempt(lambda device=device: client.set_device(device))
        if isinstance(moved, Failure):
            lines.append(Line("embed.query", device.value, Status.UNAVAILABLE, moved.reason))
            continue
        latency = _latency_lines(env, client, device)
        lines.extend(latency)
        if not effect_done and any(line.status in (Status.OK, Status.WARN) for line in latency):
            lines.append(_prefix_effect_line(env, client, device))
            effect_done = True
    return lines


def _devices(env: ProbeEnv, client: EmbedClient) -> list[Device]:
    wanted = [device for device in env.options.devices if device in client.devices]
    return wanted or [device for device in client.devices if device is not Device.NONE]


def _tag_line(tag: str, catalog: TagCatalog | None, url: str) -> Line:
    if catalog is None:
        return Line("embed.tag", "-", Status.UNAVAILABLE, f"ollama unreachable at {url}")
    if not catalog.has(tag):
        return Line("embed.tag", "-", Status.MISSING, missing_summary(tag))
    return Line("embed.tag", "-", Status.OK, f"present: {tag}")


def _card_prefix_line(env: ProbeEnv) -> Line:
    config = env.models.embed
    card = env.probe_set.card_prefixes
    query_ok = config.query_prefix == card.query
    doc_ok = config.doc_prefix == card.doc
    facts = {
        "query_prefix_matches_card": str(query_ok).lower(),
        "doc_prefix_matches_card": str(doc_ok).lower(),
        "memory_prefix_empty": str(config.memory_prefix == "").lower(),
    }
    if query_ok and doc_ok:
        return Line(
            "embed.prefix_vs_card",
            "-",
            Status.OK,
            "config prefixes equal card",
            {},
            facts,
            reached=False,
        )
    return Line(
        "embed.prefix_vs_card",
        "-",
        Status.WARN,
        "config prefixes differ from bundled card prefixes; confirm against the model card",
        {},
        facts,
        reached=False,
    )


def _latency_lines(env: ProbeEnv, client: EmbedClient, device: Device) -> list[Line]:
    name = device.value

    def query() -> list[list[float]]:
        return client.embed([QUERY_SAMPLE], kind=EmbedKind.QUERY)

    cold = attempt(lambda: time_ms(query, env.clock))
    if isinstance(cold, Failure):
        return [Line("embed.query", name, Status.UNAVAILABLE, cold.reason)]
    samples = attempt(
        lambda: sample_ms(query, env.clock, runs=env.options.runs, warmup=env.options.warmup)
    )
    if isinstance(samples, Failure):
        return [Line("embed.query", name, Status.UNAVAILABLE, samples.reason)]
    latency = summarize(samples)
    metrics = {
        "cold_ms": cold,
        "p50_ms": latency.p50_ms,
        "p95_ms": latency.p95_ms,
        "min_ms": latency.min_ms,
        "max_ms": latency.max_ms,
        "runs": float(latency.runs),
    }
    gpu_used = gpu_used_mib(env.host)
    if gpu_used is not None:
        metrics["gpu_used_mib"] = gpu_used
    over = latency.p95_ms > env.configured_deadline_ms
    summary = f"query p50 {latency.p50_ms:.0f} ms, p95 {latency.p95_ms:.0f} ms, cold {cold:.0f} ms"
    if over:
        summary += f" (p95 over deadline_ms {env.configured_deadline_ms})"
    lines = [Line("embed.query", name, Status.WARN if over else Status.OK, summary, metrics)]
    lines.append(_doc_batch_line(env, client, device))
    return lines


def _doc_batch_line(env: ProbeEnv, client: EmbedClient, device: Device) -> Line:
    codes = [pair.code for pair in env.probe_set.pairs]
    docs = [codes[index % len(codes)] for index in range(DOC_BATCH)]

    def batch() -> list[list[float]]:
        return client.embed(docs, kind=EmbedKind.DOC)

    samples = attempt(lambda: sample_ms(batch, env.clock, runs=3, warmup=1))
    if isinstance(samples, Failure):
        return Line("embed.doc_batch", device.value, Status.UNAVAILABLE, samples.reason)
    latency = summarize(samples)
    rate = rate_per_s(DOC_BATCH, latency.p50_ms)
    metrics = {"batch": float(DOC_BATCH), "p50_ms": latency.p50_ms, "docs_per_s": rate}
    return Line(
        "embed.doc_batch",
        device.value,
        Status.OK,
        f"{DOC_BATCH} docs p50 {latency.p50_ms:.0f} ms ({rate:.1f} docs/s)",
        metrics,
    )


def _prefix_effect_line(env: ProbeEnv, client: EmbedClient, device: Device) -> Line:
    pairs = env.probe_set.pairs
    queries = [pair.query for pair in pairs]
    codes = [pair.code for pair in pairs]
    variants = {
        "configured": (EmbedKind.QUERY, EmbedKind.DOC),
        "swapped": (EmbedKind.DOC, EmbedKind.QUERY),
        "memory": (EmbedKind.MEMORY, EmbedKind.MEMORY),
    }
    scores = attempt(
        lambda: {
            name: _top1(client, queries, codes, kinds[0], kinds[1])
            for name, kinds in variants.items()
        }
    )
    if isinstance(scores, Failure):
        return Line("embed.prefix_effect", device.value, Status.UNAVAILABLE, scores.reason)
    metrics: dict[str, float] = {}
    for name, (accuracy, margin) in scores.items():
        metrics[f"{name}_top1"] = accuracy
        metrics[f"{name}_margin"] = margin
    configured = scores["configured"][0]
    best_other = max(scores["swapped"][0], scores["memory"][0])
    status = Status.OK if configured >= best_other else Status.WARN
    summary = (
        f"top1 configured {configured:.2f}, swapped {scores['swapped'][0]:.2f}, "
        f"unprefixed {scores['memory'][0]:.2f} over {len(pairs)} pairs"
    )
    return Line("embed.prefix_effect", device.value, status, summary, metrics)


def _top1(
    client: EmbedClient,
    queries: Sequence[str],
    codes: Sequence[str],
    query_kind: EmbedKind,
    doc_kind: EmbedKind,
) -> tuple[float, float]:
    query_vectors = client.embed(queries, kind=query_kind)
    code_vectors = client.embed(codes, kind=doc_kind)
    hits = 0
    margins: list[float] = []
    for index, query_vector in enumerate(query_vectors):
        sims = [_cosine(query_vector, vector) for vector in code_vectors]
        wrong = max(sim for other, sim in enumerate(sims) if other != index)
        margins.append(sims[index] - wrong)
        hits += 1 if sims[index] > wrong else 0
    return hits / len(queries), sum(margins) / len(margins)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0
