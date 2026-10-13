# Model probe runbook

`cw models probe` measures the three model slots on the machine it runs on and
produces the numbers that decide the embed defaults, the annotate
deadline and the classify build. The probe never downloads, pulls, creates
or deletes a model. It reads `GET /v1/models` from each configured
llama-server and calls the embed / chat / `/v1/systemone` endpoints of models
you already serve. An unreachable server is reported as `unavailable` with
`start llama-server for <model> per docs/runbooks/model-probe.md yourself`, and
the probe carries on. The only network traffic is to the configured local
endpoints (ports below).

Reference hardware: Ryzen 7 5700X (8c/16t), RTX 3060 Ti 8 GB, 32 GB RAM, WSL2
capped near 23 GB.

## Prerequisites

1. `uv sync` in the repo. `nvidia-smi` working inside WSL2 (used only to read
   GPU memory; without it VRAM criteria stay `pending`).
2. llama.cpp `llama-server` build b11361 or newer for embed and annotate.
3. Classify: a checkout of the Kev fork with weight-only quantisation,
   [matthew-brough/kev `feat/weight-quantization`](https://github.com/matthew-brough/kev/tree/feat/weight-quantization),
   synced with `uv sync`. Its `kev.serve` speaks `/v1/systemone`.
4. Models, fetched by you. `-hf <repo>:<quant>` downloads on first start; that
   is your call, never the daemon's. Check each model page for the exact name.
5. Close other GPU users (browsers, games) so the VRAM reading is the server
   under test alone.

## Servers

One server process per slot per device, each on its own port. The device a
slot uses is the endpoint its calls go to.

| slot | device | command |
|---|---|---|
| embed | gpu | `llama-server -hf jinaai/jina-code-embeddings-1.5b-GGUF:Q8_0 --embeddings --pooling last -c 32768 -np 4 -b 8192 -ub 8192 -ngl 99 --cache-ram 0 --port 8770` |
| embed | cpu | same, `-ngl 0 --port 8771` |
| annotate | gpu | `llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --jinja -c 32768 -np 4 -ngl 99 --cache-ram 0 --port 8772` |
| annotate | cpu | same, `-ngl 0 --port 8773` |
| classify | gpu | `KEV_QUANT=int8 KEV_GRAPH_BANK_WIDTH=1024 KEV_GRAPH_TOKENS=16384 KEV_GRAPH_ROWS=16 KEV_GRAPH_STATES=8 uv run python -m kev.serve --run jaredpalmer/kev-0.8b --device cuda --port 8765` (in the fork checkout) |

Classify has no CPU server: Kev on the 5700X costs about 0.8 s per state even
at 0.8B, so the chain is GPU then skip. Kev-0.8B int8 holds about 2.2 GB of
GPU memory, leaving room for a GPU embed server alongside it.

- Embed needs `--pooling last` (Jina uses last-token pooling); the
  `embed.pooling` line catches a server started without it.
- Pass `--cache-ram 0`. The default host prompt cache is 8 GiB per server and every
  chunk is unique, so it only fills RAM (two servers reached ~20 GB under WSL2).
- Embed needs `-ub` >= the per-slot context (8192) so one doc fits a single ubatch, and
  `-c` = slots × 8192. With `-c 8192` the slots share one cache and a doc batch fails with
  a 500 "Context size has been exceeded" that marks the device down.
- Annotate `-np` should equal `models.annotate.parallel` (4), with `-c` = slots × 8192.
  With a smaller `-c` all slots share one KV cache and concurrent chunks fail with
  "Context size has been exceeded".
- `KEV_GRAPH_STATES` should equal `models.classify.parallel` (8); a
  `gate.window` of 24 runs as three batches.
- Pin the CPU servers to separate cores with `taskset -c` and set `-t`/`-tb`
  explicitly. llama-server's default thread counts oversubscribe the 5700X
  when embed and annotate run together.
- llama-server GGUF builds (`-hf ggml-org/Kev-4B-GGUF:Q8_0 --decision-type
  kev`, Clef-flash `--decision-type clef`) still work over the same wire, but
  measured 1.4 to 1.6 times slower per state than the fork.

Two ways to run them:

- **External** (default, `launch = ""`): start the servers yourself. The
  daemon only health-checks `GET /health`.
- **Supervised**: put the command in config and the daemon starts it on warm,
  stops the slot's other-device process once the new one answers `/health`,
  and stops everything on residency stop:

  ```toml
  [models.embed]
  gpu = { endpoint = "http://127.0.0.1:8770", launch = "llama-server -m /models/jina-code-embeddings-1.5b-Q8_0.gguf --embeddings --pooling last -ngl 99 --port 8770" }
  ```

  Use a local `-m <path.gguf>` instead of `-hf` once the file is downloaded,
  so a supervised start never fetches anything.

## Running

Embed and annotate, both devices (start the four servers first, or only the
ones for the device you pass):

```
uv run cw models probe --slot embed --slot annotate --record /tmp/probe-embed-annotate.md
```

GPU only or CPU only:

```
uv run cw models probe --slot embed --slot annotate --device gpu
uv run cw models probe --slot embed --slot annotate --device cpu
```

Run the CPU pass with nothing else heavy running on the 5700X; the numbers are
the reference for the incremental annotate and embed-query path.

Classify. Start the server for the rung you want to test, then describe it.
`--candidate` is repeatable and the probe walks rungs in order per family,
stopping at the first one that meets the measurable fit criteria. `family` is
a label only; every candidate is spoken to over `/v1/systemone`:

```
uv run cw models probe --slot classify \
  --candidate family=kev,rung=0,build=kev-0.8b-int8,gpu=http://127.0.0.1:8765,model=kev-latest \
  --record /tmp/probe-classify.md
```

One GPU holds one classify server at a time, so run the rungs as separate
invocations (start server for rung N, probe it, stop it) and record each. A
candidate whose endpoint does not answer is reported `unavailable` and the walk
continues. With no `--candidate`, the configured `[models.classify]` model and
endpoints are probed as rung 0 (`config`).

Options: `--json` machine output, `--runs`, `--annotate-chunks`,
`--classify-runs`, `--vram-mib` (default 8192), `--pipeline-budget-ms`
(default 1000), `--probe-deadline-ms` (default 120000, raised so slow CPU calls
finish and can be measured; results are still judged against the configured
`models.warn_ms` and `models.annotate.warn_ms`).

Exit code is 0 when at least one line was measured, 1 when nothing could be
probed. Lines for unreachable providers are `unavailable`, never an error.

## Reading the output

Status per line: `ok`, `warn` (measured but outside a limit), `unavailable`
(server unreachable or failed), `skipped`.

| line | meaning |
|---|---|
| `embed.served` / `annotate.served` | per device: server answers `/v1/models` and lists the configured model (`warn` when it serves another) |
| `embed.prefix_vs_card` | config prefixes equal the bundled Jina card prefixes |
| `embed.prefix_effect` | top-1 on 6 query/code pairs: configured prefixes vs swapped vs unprefixed. Configured should not lose |
| `embed.query` | single short query latency: cold, p50, p95 per device |
| `embed.pooling` | cosine vs bundled reference vectors >= 0.99; `warn` means the server pools differently (missing `--pooling last`); `skipped` until references are bundled |
| `embed.doc_batch` | 16-doc batch p50 and docs/s (bulk indexing speed) |
| `annotate.rate` | chunks/s, p50/p95, calls over `annotate.warn_ms`, parse failures |
| `classify.<family>.r<rung>.<build>` | gate latency p50 per window size (1, 5, 10, 20, `gate.window`) with `gate.card_tokens` states, GPU peak MiB |
| `classify.<...>.fit` | per-criterion verdicts (pass, fail, pending) |
| `classify.<...>.protocol` | `wire_source`, `served_models`, `serves_model`, `concurrent_states`, `dynamic_options` |

Fit criteria the probe can measure: gate p50 at `gate.window` (GPU) or at
window >= 10 (CPU, rung 6) within `min(warn_ms, 1000 ms)`, and peak GPU
memory <= 8192 MiB. Quality (top agreement >= 95%, recall drop <= 2 pts), ECE
and full pipeline p50 are always `pending`: they need the classify eval. A rung
reported as fitting is a candidate, not a winner. If every rung of every family
fails, the output says so; record the gaps in section B and raise it to the
user, do not ship fallback scoring as the default.

Protocol facts: `concurrent_states` is `yes` when the widest window costs under
0.8 x (window x single-state latency), i.e. the server overlaps requests up to
`-np`; `dynamic_options` is `yes` when unseen option names come back as a
distribution over exactly those options.

## Pooling reference vectors

`embed.pooling` compares the server against vectors from the reference
implementation. Generate them once on any machine that has the model weights
(CPU is fine), with the configured prefix applied to the text exactly as the
client sends it, and append them to
`src/contextweave/models/probe/probe_set.toml`:

```
uv run --with sentence-transformers python - <<'EOF'
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("jinaai/jina-code-embeddings-1.5b")
query = "Find the most relevant code snippet given the following query:\n"
doc = "Candidate code snippet:\n"
for kind, prefix, text in [
    ("query", query, "parse the config file"),
    ("doc", doc, "def load_config(path):\n    return tomllib.loads(path.read_text())"),
]:
    vector = model.encode(prefix + text, normalize_embeddings=True).tolist()
    print(f"[[embed_references]]\nkind = {kind!r}\ntext = {text!r}\nvector = {vector}\n")
EOF
```

The probe truncates the reference to `models.embed.dim`, so store the full
vector. Regenerate whenever the embed model or prefixes change.

## What to record

Use `--record PATH` on every run and keep the files (one per run) with the
change that tunes a default. Each snippet has: host line, result table, a
`measurements:` list (all numbers, criterion verdicts, protocol facts) and a
`decides:` list. Paste raw; do not round. Also paste the `nvidia-smi` total used
with only the classify server resident, and the exact `llama-server` command
and build id for each classify rung tried.

## Which defaults the results decide

- `models.embed.query_device`: CPU stays if `embed.query` cpu p50 is within
  250 ms (the probe prints the recommendation); otherwise `gpu` and classify
  must share it. `embed.prefix_effect` and `embed.prefix_vs_card` decide
  whether `query_prefix` and `doc_prefix` stay as shipped. Also
  confirm `dim` (1536 vs 512) against `embed.doc_batch` throughput and index
  size.
- `models.annotate.warn_ms` (default 30000, separate from the 1500 ms
  query-time `models.warn_ms`; hard caps are `deadline_ms`, loose by design): the probe prints the p95-based figure
  (p95 x 1.25 rounded up to 500 ms) from `annotate.rate` cpu. GPU
  `chunks_per_s` gives the `cw init` ETA.
- classify: decided 2026-10-12. Kev-0.8B on the fork's `kev.serve`, int8,
  GPU only, `gate.window` 24, `gate.card_tokens` 512. On the live eval suites it
  matched Kev-4B recall at 1.6 to 2.5 times lower p50, and Kev-4B cannot hold
  the widened window inside the query deadline.
