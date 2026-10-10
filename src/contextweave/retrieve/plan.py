import re
from dataclasses import replace

from contextweave.core.errors import ErrorEntry
from contextweave.index.store import Store
from contextweave.models.decision import CHOICE_OPTION_LIMIT
from contextweave.retrieve.ask import ask_stage
from contextweave.retrieve.candidates import Candidate, CandidateSet, Ranks
from contextweave.retrieve.context import Plan, QueryContext, RrfWeights
from contextweave.retrieve.freshness import freshen
from contextweave.retrieve.hydrate import load_candidates
from contextweave.retrieve.rows import as_int, as_str
from contextweave.retrieve.scope import normalize_scope, scope_clause
from contextweave.retrieve.score import mark_exact, with_final
from contextweave.retrieve.terms import words

STAGE_NAME = "plan"
QUESTION_SET = "plan"
RECENT_FILES = 10
SCOPE_MIN_CONFIDENCE = 0.5
GRAPH_MIN_PROBABILITY = 0.5
INTENTS: tuple[str, ...] = (
    "lookup",
    "implementation",
    "architecture",
    "debug",
    "config",
    "usage",
)
INTENT_RELS: dict[str, tuple[str, ...]] = {
    "lookup": (),
    "implementation": ("callees", "imports"),
    "architecture": ("imports", "importers", "inherits", "implements"),
    "debug": ("callers", "callees", "tests", "emitters", "listeners"),
    "config": ("config", "importers"),
    "usage": ("callers", "tests", "emitters"),
}
INTENT_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "debug",
        frozenset(
            [
                "error",
                "errors",
                "bug",
                "bugs",
                "fail",
                "fails",
                "failing",
                "failure",
                "crash",
                "crashes",
                "exception",
                "traceback",
                "broken",
                "raises",
                "raise",
                "wrong",
                "why",
            ]
        ),
    ),
    (
        "config",
        frozenset(
            [
                "config",
                "configuration",
                "configure",
                "setting",
                "settings",
                "env",
                "environment",
                "option",
                "options",
                "toml",
                "yaml",
                "json",
                "flag",
                "flags",
            ]
        ),
    ),
    (
        "architecture",
        frozenset(
            [
                "architecture",
                "overview",
                "structure",
                "design",
                "layers",
                "layer",
                "organized",
                "modules",
                "module",
                "components",
                "component",
                "diagram",
            ]
        ),
    ),
    (
        "usage",
        frozenset(
            [
                "usage",
                "used",
                "uses",
                "use",
                "callers",
                "caller",
                "calls",
                "call",
                "example",
                "examples",
            ]
        ),
    ),
    (
        "implementation",
        frozenset(
            ["implement", "implements", "implementation", "logic", "algorithm", "works", "work"]
        ),
    ),
)
DEFAULT_PLAN = Plan(intent="implementation")
NATURAL_LANGUAGE_WORDS = 4
SINGLE_WORD_WEIGHTS = RrfWeights(lexical=1.0, semantic=0.5, structural=2.0)
NATURAL_LANGUAGE_WEIGHTS = RrfWeights(lexical=1.0, semantic=1.5, structural=0.5)
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")


def rules_intent(query: str) -> str:
    """Keyword rules over the query words; a lone word is a lookup, free text an implementation."""
    tokens = {word.lower() for word in words(query)}
    for intent, keywords in INTENT_KEYWORDS:
        if tokens & keywords:
            return intent
    return "lookup" if len(words(query)) <= 1 else "implementation"


def rules_weights(query: str) -> RrfWeights:
    """Structural-heavy for a lone word, semantic-heavy for natural language."""
    count = len(words(query))
    if count == 1:
        return SINGLE_WORD_WEIGHTS
    if count >= NATURAL_LANGUAGE_WORDS:
        return NATURAL_LANGUAGE_WEIGHTS
    return RrfWeights()


def path_exists(store: Store, path: str) -> bool:
    clause, params = scope_clause("path", path)
    return bool(store.query(f"SELECT 1 FROM files WHERE {clause} LIMIT 1", params))


def rules_scope(store: Store, query: str) -> str | None:
    """The first query word that is a slash path naming an indexed file or directory."""
    for word in query.split():
        raw = word.strip("\"'`,;:()")
        text = normalize_scope(raw) if "/" in raw else None
        if text and path_exists(store, text):
            return text
    return None


def symbol_chunks(
    store: Store, query: str, scope: str | None
) -> tuple[tuple[int, ...], set[tuple[str, str]]]:
    """Chunk ids of symbols named exactly like the query, and their distinct (name, kind)."""
    clause, params = scope_clause("f.path", scope)
    rows = store.query(
        "SELECT s.name, s.kind, cs.chunk_id FROM symbols s"
        " JOIN chunk_symbols cs ON cs.symbol_id = s.id JOIN files f ON f.id = s.file_id"
        f" WHERE (s.qualified_name = ? OR s.name = ?) AND {clause}",
        [query, query, *params],
    )
    identities = {(as_str(row[0]), as_str(row[1])) for row in rows}
    return tuple(sorted({as_int(row[2]) for row in rows})), identities


def fast_path_candidates(ctx: QueryContext, scope: str | None) -> tuple[Candidate, ...]:
    """Symbols of one (name, kind) matching a single identifier query, else nothing."""
    query = ctx.query.strip()
    if ctx.mode != "auto" or not IDENTIFIER.fullmatch(query):
        return ()
    ids, identities = symbol_chunks(ctx.store, query, scope)
    if len(identities) != 1:
        return ()
    loaded = load_candidates(ctx.store, ids)
    found = sorted(loaded.values(), key=lambda cand: cand.order_key)
    ranked = CandidateSet.of(
        cand.with_ranks(Ranks(structural=rank)) for rank, cand in enumerate(found, start=1)
    )
    fresh = freshen(ctx.store, ranked, ctx.envelope, STAGE_NAME)
    return tuple(
        with_final(mark_exact(cand, "exact symbol match"), ctx.config.score) for cand in fresh
    )


def resolve_intent(ctx: QueryContext, default: str) -> str:
    if ctx.intent is None:
        return default
    if ctx.intent in INTENTS:
        return ctx.intent
    ctx.envelope.add_error(
        ErrorEntry(
            code="invalid_intent", message=f"unknown intent {ctx.intent!r}", stage=STAGE_NAME
        )
    )
    return default


def scope_hints(store: Store) -> tuple[list[str], list[str]]:
    """Top-level directories that fit beside the `any` scope option, and the 10 most recent files."""
    paths = [as_str(row[0]) for row in store.query("SELECT path FROM files ORDER BY path")]
    directories = sorted({path.split("/", 1)[0] for path in paths if "/" in path})
    recent = store.query(
        "SELECT path FROM files ORDER BY mtime_ns DESC, path LIMIT ?", (RECENT_FILES,)
    )
    return directories[: CHOICE_OPTION_LIMIT - 1], [as_str(row[0]) for row in recent]


def plan_state(query: str, directories: list[str], recent: list[str]) -> str:
    return (
        f"query: {query}\ndirectories: {', '.join(directories)}\nrecent files: {', '.join(recent)}"
    )


def refine_with_model(ctx: QueryContext, rules: Plan, explicit_scope: str | None) -> Plan:
    """Classifier intent, scope and graph need over a rules plan; the rules plan when skipped."""
    if ctx.models is None:
        return rules
    directories, recent = scope_hints(ctx.store)
    answers = ask_stage(
        ctx,
        QUESTION_SET,
        [plan_state(ctx.query, directories, recent)],
        stage=STAGE_NAME,
        dynamic={"scope_dirs": directories},
    )
    if answers is None:
        return rules
    row = answers[0]
    intent = rules.intent if ctx.intent in INTENTS else row.top("intent")
    chosen = row.top("scope")
    modelled = chosen in directories and row.confidence("scope") >= SCOPE_MIN_CONFIDENCE
    scope = explicit_scope or (normalize_scope(chosen) if modelled else None)
    needs_graph = row.value("needs_graph") >= GRAPH_MIN_PROBABILITY
    return replace(
        rules,
        intent=intent,
        scope=scope,
        rels=INTENT_RELS[intent] if needs_graph else (),
        planner="model",
    )


class PlanStage:
    """Rules first (fast path, intent, scope, rrf weights, rels); classify refines a non-fast-path plan."""

    name = STAGE_NAME

    def run(self, ctx: QueryContext, cands: CandidateSet) -> CandidateSet:
        scope = normalize_scope(ctx.scope) or rules_scope(ctx.store, ctx.query)
        hits = fast_path_candidates(ctx, scope)
        intent = resolve_intent(ctx, "lookup" if hits else rules_intent(ctx.query))
        plan = Plan(
            intent=intent,
            scope=scope,
            rrf_weights=rules_weights(ctx.query),
            rels=() if hits else INTENT_RELS[intent],
            fast_path=bool(hits),
        )
        if not hits:
            plan = refine_with_model(ctx, plan, scope)
        ctx.plan.value = plan
        return CandidateSet.of(hits) if hits else cands
