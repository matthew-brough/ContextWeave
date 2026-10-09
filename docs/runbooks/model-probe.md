# Model probe runbook (T10)

`cw models probe` measures the three model slots on the machine it runs on and
produces the numbers that decide the T12 embed defaults, the T19 annotate
deadline and the T20 classify build. The probe never downloads, pulls, creates
or deletes a model. It only reads `GET /api/tags` from Ollama and calls the
embed/chat endpoints of models you already have. A missing model is reported as
`missing: <tag> (run `ollama pull <tag>` yourself)` and the probe carries on.
The only network traffic is to the configured local endpoints (Ollama
`127.0.0.1:11434`, classify servers `127.0.0.1:8765` / `8766`).

Reference hardware: Ryzen 7 5700X (8c/16t), RTX 3060 Ti 8 GB, 32 GB RAM, WSL2
capped near 23 GB.

## Prerequisites

1. `uv sync` in the repo. `nvidia-smi` working inside WSL2 (used only to read
   GPU memory; without it VRAM criteria stay `pending`).
2. Ollama running locally: `ollama serve` (or the Windows service).
3. Models, pulled by you. Check each model page for the exact name before
   running; the commands below are the intended ones.

   Annotate (Qwen3.5 4B):

   ```
   ollama pull qwen3.5:4b
   ```

   Embed (Jina code embeddings 1.5B). The config tag is
   `jina-code-embeddings-1.5b`. Pull a GGUF build from Hugging Face and alias it
   to that tag (confirm the quant name on the model page first):

   ```
   ollama pull hf.co/jinaai/jina-code-embeddings-1.5b-GGUF:Q8_0
   ollama cp hf.co/jinaai/jina-code-embeddings-1.5b-GGUF:Q8_0 jina-code-embeddings-1.5b
   ```

   Classify (separate torch servers, not Ollama). Download the weights with the
   Hugging Face CLI and start the server per the model repo README
   (`kev.serve` for Kev). Clef-flash: use the repo id from its model card.

   ```
   hf download jaredpalmer/kev-4b
   hf download <clef-flash repo id from its model card>
   ```

   Ladder rungs 2 and 5 need quantised builds you create with the tooling in
   `tools/quantize/` (T20); the probe only measures builds that are already
   being served.

4. Close other GPU users (browsers, games). Stop idle Ollama models so the VRAM
   reading is the classify server alone: `ollama stop <model>`.

## Running

Embed and annotate, both devices (Ollama reloads the model with `num_gpu=0`
for CPU, so the CPU pass is slow the first time):

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
stopping at the first one that meets the measurable fit criteria:

```
uv run cw models probe --slot classify \
  --candidate family=kev,rung=1,build=published-q4,gpu=http://127.0.0.1:8765,cpu=http://127.0.0.1:8766 \
  --candidate family=kev,rung=3,build=ctx4k-nograph,gpu=http://127.0.0.1:8765 \
  --candidate family=clef,rung=4,build=textonly-int8,gpu=http://127.0.0.1:8765 \
  --record /tmp/probe-classify.md
```

Because one GPU holds one server at a time, run the rungs as separate
invocations (start server for rung N, probe it, stop it) and record each. A
candidate whose endpoint does not answer is reported `unavailable` and the walk
continues. With no `--candidate`, the configured `[models.classify]` model and
endpoints are probed as rung 0 (`config`). The `kev` and `clef` providers must
be registered (T20); until then the line reads `provider ... not registered`.

Options: `--json` machine output, `--runs`, `--annotate-chunks`,
`--classify-runs`, `--vram-mib` (default 8192), `--pipeline-budget-ms`
(default 1000), `--probe-deadline-ms` (default 120000, raised so slow CPU calls
finish and can be measured; the configured `deadline_ms` is still what the
results are judged against), `--ollama-url`.

Exit code is 0 when at least one line was measured, 1 when nothing could be
probed. Lines for unreachable providers are `unavailable`, never an error.

## Reading the output

Status per line: `ok`, `warn` (measured but outside a limit), `missing` (model
absent, pull it), `unavailable` (provider unreachable or failed), `skipped`.

| line | meaning |
|---|---|
| `embed.tag` / `annotate.tag` | model present in Ollama |
| `embed.prefix_vs_card` | config prefixes equal the bundled Jina card prefixes |
| `embed.prefix_effect` | top-1 on 6 query/code pairs: configured prefixes vs swapped vs unprefixed. Configured should not lose |
| `embed.query` | single short query latency: cold, p50, p95 per device |
| `embed.doc_batch` | 16-doc batch p50 and docs/s (bulk indexing speed) |
| `annotate.rate` | chunks/s, p50/p95, calls over configured `deadline_ms`, parse failures |
| `classify.<family>.r<rung>.<build>` | gate latency p50 per window size (1, 5, 10, 20, `gate.window`) with `gate.card_tokens` states, GPU peak MiB |
| `classify.<...>.fit` | per-criterion verdicts (pass, fail, pending) |
| `classify.<...>.protocol` | `score_native`, `multi_state_batching`, `dynamic_options`, plus the OpenAPI paths found |

Fit criteria the probe can measure: gate p50 at `gate.window` (GPU) or at
window >= 10 (CPU, rung 6) within `min(deadline_ms, 1000 ms)`, and peak GPU
memory <= 8192 MiB. Quality (top agreement >= 95%, recall drop <= 2 pts), ECE
and full pipeline p50 are always `pending`: they need the T20 eval. A rung
reported as fitting is a candidate, not a winner. If every rung of every family
fails, the output says so; record the gaps in section B and raise it to the
user, do not ship fallback scoring as the default.

Protocol checks are inferred from replies: `score_native` is `yes` when the
reply keys differ from the five bucket options or the OpenAPI document mentions
`score`, otherwise `no`; `multi_state_batching` is `yes` when the widest window
costs under 0.8 x (window x single-state latency) and returns one row per
state; `dynamic_options` is `yes` when unseen option names come back as a
distribution over exactly those options.

## What to paste into section B

Use `--record PATH` on every run and paste the files (one per run) under `§B` of
`docs/SPECS/SPEC.md`. Each snippet has: host line, result table, a
`measurements:` list (all numbers, criterion verdicts, protocol facts) and a
`decides:` list. Paste raw; do not round. Also paste the `nvidia-smi` total used
with only the classify server resident, and the exact server launch command and
build id for each classify rung tried.

## Which defaults the results decide

- T12 `models.embed.query_device`: CPU stays if `embed.query` cpu p50 is within
  250 ms (the probe prints the recommendation); otherwise `gpu` and classify
  must share it. `embed.prefix_effect` and `embed.prefix_vs_card` decide
  whether `query_prefix`, `doc_prefix` and `memory_prefix` stay as shipped. Also
  confirm `dim` (1536 vs 512) against `embed.doc_batch` throughput and index
  size.
- T19 annotate deadline: `models.deadline_ms = 1500` is a single value for every
  slot. Qwen3.5 4B on the 5700X will very likely exceed it per chunk
  (`annotate.rate` cpu `over_deadline`). The probe prints the p95-based figure
  (p95 x 1.25 rounded up to 500 ms). Raising the global value loosens the
  query-time embed and classify deadlines, so the real decision is whether to
  add a per-slot override for annotate. That is a spec change; decide it with
  the numbers, not before. GPU `chunks_per_s` gives the `cw init` ETA.
- T20 classify: the winning family, rung/build and device, the CPU
  `gate.window` the 5700X sustains (printed per CPU run as "largest gate.window
  within N ms"), and which of `score`, batching and dynamic options each
  server really supports, so the provider adapters and `options_from` handling
  are written to measured behaviour. Quality and calibration are finished in
  the T20 eval on the chosen build.
