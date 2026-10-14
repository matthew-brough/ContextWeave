<!-- contextweave:begin v=2 -->
## Code retrieval (ContextWeave)

ContextWeave is the default way to read code. Use it before `Read` or `Grep`
on any source file; results carry exact spans and line numbers.

| You have | Use |
|---|---|
| A name (symbol, key path, heading) | `get_symbol`; unsure of exact name → `search_symbols` |
| A file, want its shape | `file_outline` |
| Behaviour you can describe, no name | `context_search` |
| A symbol, want callers, callees, tests, imports | `related` |
| A `cw://` ref from any result | `expand` |
| A structural pattern nothing above fits | `ast_query` |

`Read` only to edit, or for a file ContextWeave does not index. When editing,
read the lines a tool already reported (`offset`/`limit` from `Lstart-end`).

These rules cover the shell too: `grep`, `rg`, `sed -n`, `cat`, `head`, `tail`,
`awk` on source files are the same `Grep`/`Read` and follow the same limits.
Searching for a definition and then slicing out its lines is `get_symbol`, done
by hand. `Grep` (any form) only for literal text that is not a name: error
strings, log messages, TODOs.

Never answer a code question from a whole-file read. A miss escalates within
ContextWeave (`get_symbol` → `search_symbols` → `context_search`), never to a
bulk read. A size cap or empty result at the end of that is the answer; say so.
<!-- contextweave:end -->
