# ContextWeave

Code-context MCP server for coding agents. It gives an agent structural, lexical
and semantic retrieval over a repository, so it can pull the definition it
needs instead of reading whole files.

- **Structural**: tree-sitter symbols, key paths and headings for 27 languages
  and data/doc formats (`cw languages`).
- **Lexical + semantic**: a SQLite index with full-text search and
  [sqlite-vec](https://github.com/asg017/sqlite-vec) embeddings, fused and
  packed into a token budget.
- **Graph**: callers, callees, imports, inheritance, tests and config edges.

Published on PyPI as **`contextweave-mcp`**. The import package is
`contextweave` and the commands are `cw` and `contextweave`.

## Install

Requires Python 3.14+.

```sh
uv tool install contextweave-mcp
cw --version
```

## Quickstart

From the root of a project:

```sh
cw init                 # index + wire up Claude Code (default agent)
cw status               # counts, freshness, daemon and model health
cw search "where are retries handled"
```

`cw init` indexes the project and installs the agent integration: MCP server
registration, hooks, and a routing block that tells the agent when to use which
tool. Remove it again with `cw uninstall` (the index is kept; `cw forget` drops
it).

| `--agent`     | Integration                                  |
| ------------- | -------------------------------------------- |
| `claude_code` | `.mcp.json`, `.claude/settings.json` hooks and a `CLAUDE.md` block (default) |
| `codex`       | One global entry in Codex's `config.toml` and an `AGENTS.md` block |
| `agents_md`   | `.mcp.json` and an `AGENTS.md` block, no hooks |
| `none`        | Index only                                   |

`cw init --plugin` writes a Claude Code plugin directory instead of editing
project files. `--dry-run` shows what would change.

To run the server by hand for any MCP client:

```sh
cw serve   # MCP over stdio
```

## MCP tools

| Tool             | Use it to                                                      |
| ---------------- | -------------------------------------------------------------- |
| `context_search` | Describe behaviour in words; get a budgeted pack of relevant code |
| `get_symbol`     | Fetch one definition, key path or heading by name             |
| `file_outline`   | See one file's shape: symbols, schema key paths or headings   |
| `search_symbols` | Find names across the project                                  |
| `ast_query`      | Run a raw tree-sitter query on one file                        |
| `expand`         | Resolve a `cw://` ref to its current bytes                     |
| `related`        | Graph neighbours of a symbol or file                           |

Every tool is read-only. Results are terse text; summarised or omitted items
carry `cw://` refs that `expand` resolves later, and report when the code moved
or was deleted.

## Models

Semantic search, chunk annotation and query gating use local models served by
[llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` (embed,
annotate) and a classifier endpoint. No model weights are bundled. Either run
the servers yourself and point the config at them, or give each slot a `launch`
command and the daemon starts and supervises them. Weights listed under
`models.<slot>.weights` are downloaded by `cw init` (or `cw models fetch`) when
missing, never by a server start.
Each model has its own licence: the default embed model,
jina-code-embeddings, is CC-BY-NC (non-commercial). Swap `models.embed`
for a permissively licensed model if that does not fit your use.

| Slot       | Default model                                 | Endpoints (gpu / cpu) |
| ---------- | --------------------------------------------- | --------------------- |
| `embed`    | `jinaai/jina-code-embeddings-1.5b-GGUF:Q8_0`  | `:8770` / `:8771`     |
| `annotate` | `unsloth/Qwen3.5-4B-GGUF:Q4_K_M`              | `:8772` / `:8773`     |
| `classify` | Kev-0.8B (`kev-latest`)                       | `:8765` (gpu only)    |

Example servers:

```sh
llama-server -hf jinaai/jina-code-embeddings-1.5b-GGUF:Q8_0 --embeddings --pooling last --port 8770
llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --port 8772
```

Supervised servers, downloaded on `cw init`:

```toml
[models]
weights_dir = "~/models"  # single-file weights; default ~/.cache/contextweave/models

[models.embed]
weights = ["jinaai/jina-code-embeddings-1.5b-GGUF/jina-code-embeddings-1.5b-Q8_0.gguf"]
gpu = { endpoint = "http://127.0.0.1:8770", launch = "llama-server -m {weights} --embeddings --pooling last -c {ctx} -np 4 -b {max_input_tokens} -ub {max_input_tokens} -ngl 99 --port 8770" }
```

A `weights` entry is `org/repo/file` (one file into `weights_dir`) or `org/repo`
(whole repo into the Hugging Face cache). `{weights}` in `launch` expands to the
first entry's local path. Embed launches also get `{max_input_tokens}`
(`models.embed.max_input_tokens`, default 2048) and `{ctx}` (that × `parallel`); a
llama-server embed launch with `-ub` below the cap is refused. One GPU rarely holds every model, so the daemon keeps
a residency profile: during bulk indexing annotate and embed each get the GPU
alone in turn, and while serving the classifier holds the GPU with embed and
annotate on CPU. Server output is discarded, but when a start fails the last
stderr lines appear in `cw status` and in the degraded lines of `cw index`.

Models are optional. Without them the structural and lexical paths still work;
indexing reports the skipped stages as degraded and `cw index` fills them in
once the servers are up. `cw models probe` times each slot on your hardware.

## Configuration

Settings are layered: built-in defaults, then
`~/.config/contextweave/config.toml` (honours `XDG_CONFIG_HOME`), then
`.contextweave.toml` in the project root. `cw config show` prints the effective
values.

```toml
[index]
include = ["src", "tests"]
ignore = ["vendor/**"]

[models.embed.gpu]
endpoint = "http://127.0.0.1:8770"
# Set launch to have the daemon start and supervise the server itself.
launch = ""
```

## CLI

| Command                         | Purpose                                    |
| ------------------------------- | ------------------------------------------ |
| `cw init` / `cw uninstall`      | Install or remove agent integration        |
| `cw index`                      | Build or refresh the index                 |
| `cw status`                     | Counts, freshness, daemon and model health |
| `cw search <query>`             | Run the retrieval pipeline and print the pack |
| `cw projects` / `cw forget`     | List or remove registered projects         |
| `cw daemon start\|stop\|status` | Per-user daemon (Linux / WSL2)             |
| `cw savings`                    | Token savings ledger                       |
| `cw config show`                | Effective configuration                    |
| `cw models fetch` / `probe`     | Download missing weights, time each slot   |
| `cw languages`                  | Supported languages and their profiles     |
| `cw cache`, `cw tune`, `cw eval`| Maintenance, score tuning, evaluation suites |

## Platforms

Linux and WSL2 are the primary targets and run the shared per-user daemon. On
Windows the server runs in-process; the daemon needs Unix sockets.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest            # add -m "not daemon" off Linux
```

## Licence

[Mozilla Public License 2.0](LICENSE). Model weights used at runtime are not
part of this project and are covered by their own licences.
