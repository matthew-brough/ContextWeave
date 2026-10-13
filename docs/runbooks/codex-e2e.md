# Codex end-to-end check (manual)

Verifies SPEC T23 / I.project: one global Codex registration, project resolved
per call. Automated simulation: `tests/agents/test_agents_codex_e2e.py`. This
runbook covers what only a real Codex install shows: which resolver steps the
client actually supplies.

Use a scratch `CODEX_HOME` so your real config is untouched.

## Setup

```sh
export CODEX_HOME=$(mktemp -d)
mkdir -p /tmp/cw-a /tmp/cw-b
printf 'def alpha_only():\n    return 1\n' > /tmp/cw-a/mod.py
printf 'def beta_only():\n    return 1\n' > /tmp/cw-b/mod.py
(cd /tmp/cw-a && git init -q && cw init --agent codex)
(cd /tmp/cw-b && git init -q && cw init --agent codex)
```

Expect: `$CODEX_HOME/config.toml` holds exactly one
`[mcp_servers.contextweave]` table (`command = "cw"`, `args = ["serve"]`);
`/tmp/cw-a/AGENTS.md` and `/tmp/cw-b/AGENTS.md` each hold one
`<!-- contextweave:begin v=1 -->` block; no `.mcp.json` in either project.

## Checks

Run each from a Codex session started in the stated directory. Every tool
payload echoes `project: {key, root, via}`.

| # | start dir | prompt | expected |
|---|---|---|---|
| 1 | `/tmp/cw-a` | "call get_symbol for alpha_only" | found; `root=/tmp/cw-a` |
| 2 | `/tmp/cw-b` | "call get_symbol for beta_only" | found; `root=/tmp/cw-b` |
| 3 | `/tmp/cw-a` | "call get_symbol for beta_only" | not found, `root=/tmp/cw-a` (no cross-project leak) |
| 4 | `/tmp/cw-a` | "call get_symbol for beta_only with project=/tmp/cw-b" | found; `via=arg` |
| 5 | `/tmp` | "call get_symbol for alpha_only" | `errors[not_indexed]` or `no_project`; never a guessed project |

Record the `via` value of checks 1 and 2:

- `cwd`: Codex starts the server in the session directory (resolver step 4).
- `roots`: Codex answers `roots/list` (step 3).
- A wrong project in 1 or 2 means Codex spawned the server once with a fixed
  cwd. Then the `project` arg is the only route: add it to the AGENTS.md block
  and note the finding in SPEC I.project.

## Hooks

Codex hook support is not assumed. `cw init --agent codex` installs none. If your Codex version
exposes lifecycle hooks, note the event names and payload shape here.

## Teardown

```sh
(cd /tmp/cw-a && cw uninstall --agent codex)
(cd /tmp/cw-b && cw uninstall --agent codex)
```

Expect both AGENTS.md blocks gone and the `[mcp_servers.contextweave]` table
removed from `$CODEX_HOME/config.toml`. The first uninstall already removes the
global table (it is shared), so run `cw init --agent codex` again in any other
project that still needs it.
