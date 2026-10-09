from dataclasses import replace

from pack_support import make_cand

from contextweave.core.config import Config, ScoreConfig
from contextweave.retrieve.candidates import Ranks, Scores
from contextweave.retrieve.score import (
    exact_name_hit,
    fallback_scores,
    final_score,
    rank_norm,
    role_of,
)


def test_rank_norm() -> None:
    assert rank_norm(None, 100) == 0.0
    assert rank_norm(1, 100) == 1.0
    assert rank_norm(51, 100) == 0.5


def test_role_rules() -> None:
    globs = Config().index.test_globs
    assert role_of(make_cand(1, "tests/test_a.py"), globs) == "test"
    assert role_of(make_cand(1, "pkg/a_test.go"), globs) == "test"
    assert role_of(make_cand(1, "pkg/a.py"), globs) == "implementation"
    assert role_of(replace(make_cand(1, "cfg.json"), profile="schema"), globs) == "config"


def test_exact_name_hit_matches_qualified_or_simple_name() -> None:
    cand = make_cand(1, name="Widget.render")
    assert exact_name_hit(cand, ("render",))
    assert exact_name_hit(cand, ("Widget.render",))
    assert not exact_name_hit(cand, ("renderer",))
    assert not exact_name_hit(make_cand(2), ("render",))


def test_final_score_formula() -> None:
    scores = Scores(gate=0.5, graph_proximity=1.0, redundancy=0.5)
    weights = ScoreConfig()
    assert final_score(scores, weights) == 0.9 * 0.5 + 0.1 * 1.0 - 0.2 * 0.5


def test_fallback_gate_is_fused_norm_unless_exact() -> None:
    config = Config()
    cand = make_cand(1, "pkg/a.py", name="alpha", ranks=Ranks(lexical=2), fused=0.7)
    plain = fallback_scores(cand, config, ("zzz",))
    assert plain.scores.gate == 0.7
    assert plain.scores.lexical_match == 0.99
    assert plain.scores.implementation == 1.0
    assert plain.scores.graph_proximity == 1.0
    assert not plain.critical
    assert plain.why == "lexical #2"
    exact = fallback_scores(cand, config, ("alpha",))
    assert exact.scores.gate == 1.0
    assert exact.critical
    assert exact.why == "exact name match"
