<!-- contextweave:begin v=1 -->
## Code retrieval (ContextWeave)

Route on what you can name.

- **Name it** (a symbol, key path or heading): `get_symbol` for one definition,
  `search_symbols` to find a name across the repo, `file_outline` for a file's
  shape, `ast_query` as the escape hatch.
- **Describe it** (behaviour you can characterise but not name): `context_search`.
  Its hits carry names and refs; hand them to `get_symbol` or `expand` for the
  exact definition.
- **Read it** (a path you are about to edit, or content you need verbatim): `Read`.

Do not read a whole file to answer a question about its shape. A tool that
reports a size cap or finds nothing is an answer; say so instead of falling
back to a bulk read.

Memory loop:

- `recall` with a topic phrase before answering a design question or a "why did we".
- `remember` after a choice you would not want re-litigated: a decision with its
  reason, or a code area with its path.
<!-- contextweave:end -->
