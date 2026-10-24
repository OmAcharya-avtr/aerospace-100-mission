"""The two published measurements, and the arithmetic behind their ratios."""

from __future__ import annotations

import numpy as np
import pytest

from cmdguard.benchmark import (
    FamilyMissRate,
    assert_disjoint,
    lint_miss_rate,
    ranker_comparison,
    timing_table,
    unused_report_guard,
)
from cmdguard.dataset import build_corpus, generate_clean_case
from cmdguard.lint import LintStrength
from cmdguard.trace import forward_execute


@pytest.fixture(scope="module")
def corpus() -> tuple:
    return build_corpus(range(6), include_clean=True)


def test_miss_rate_ratios_are_derived_from_the_counts(corpus: tuple) -> None:
    result = lint_miss_rate(corpus, LintStrength.PER_COMMAND)
    total = result.total
    assert total.n_truth > 0
    assert total.miss_rate == pytest.approx(total.n_missed / total.n_truth)
    assert total.n_missed <= total.n_truth
    assert total.n_lint_false_positive <= total.n_lint_findings
    assert result.seconds > 0.0


def test_family_totals_sum_to_the_aggregate(corpus: tuple) -> None:
    result = lint_miss_rate(corpus, LintStrength.WITH_TIMING)
    assert sum(f.n_truth for f in result.families.values()) == result.total.n_truth
    assert sum(f.n_missed for f in result.families.values()) == result.total.n_missed
    assert sum(f.n_cases for f in result.families.values()) == result.total.n_cases


def test_timing_strength_never_misses_more_than_per_command(corpus: tuple) -> None:
    """Adding a check cannot increase the miss count. A guard on the comparison."""
    weak = lint_miss_rate(corpus, LintStrength.PER_COMMAND).total
    strong = lint_miss_rate(corpus, LintStrength.WITH_TIMING).total
    assert strong.n_truth == weak.n_truth
    assert strong.n_missed <= weak.n_missed


def test_clean_cases_contribute_no_truth(corpus: tuple) -> None:
    result = lint_miss_rate(corpus, LintStrength.PER_COMMAND)
    clean = result.families["clean"]
    assert clean.n_truth == 0
    assert np.isnan(clean.miss_rate)
    assert clean.n_cases_without_violation == clean.n_cases


def test_undefined_ratios_are_nan_not_zero() -> None:
    empty = FamilyMissRate(family="x")
    assert np.isnan(empty.miss_rate)
    assert np.isnan(empty.ordering_miss_rate)
    assert np.isnan(empty.false_positive_rate)


def test_missed_by_kind_sums_to_the_missed_count(corpus: tuple) -> None:
    total = lint_miss_rate(corpus, LintStrength.PER_COMMAND).total
    assert sum(total.missed_by_kind.values()) == total.n_missed


def test_rows_render_every_family(corpus: tuple) -> None:
    result = lint_miss_rate(corpus, LintStrength.PER_COMMAND)
    rows = result.rows()
    assert len(rows) == len(result.families)
    assert all(len(row) == 7 for row in rows)


def test_assert_disjoint_accepts_and_rejects() -> None:
    assert assert_disjoint(range(5), range(100, 105)) == (5, 5)
    with pytest.raises(ValueError, match="overlap on 1 seed"):
        assert_disjoint(range(5), range(4, 9))


def test_ranker_comparison_scores_four_orderings() -> None:
    train = build_corpus(range(6), include_clean=False)
    test = build_corpus(range(500, 504), include_clean=False)
    result = ranker_comparison(
        train, test, ks=(5, 10), ranker=None, drop_features=(), random_seed=11
    )
    for label in ("learned", "irreversibility_flag", "resource_draw", "random_control"):
        for k in (5, 10):
            score = result.scores[(label, k)]
            assert score.n_scored + score.n_excluded == len(test)
            assert 0.0 <= score.mean_precision <= 1.0
            assert 0.0 <= score.mean_recall <= 1.0
    assert result.n_train_cases == len(train)
    assert result.n_test_cases == len(test)
    assert 0.0 < result.positive_fraction_test < 1.0
    assert result.mean_steps_per_case > 0.0


def test_ranker_comparison_table_and_best(corpus: tuple) -> None:
    train = build_corpus(range(6), include_clean=False)
    test = build_corpus(range(600, 604), include_clean=False)
    result = ranker_comparison(train, test, ks=(5,))
    lines = result.table()
    assert any("learned" in line for line in lines)
    assert any("95% CI" in line for line in lines)
    assert result.best_at(5) in {
        "learned",
        "irreversibility_flag",
        "resource_draw",
        "random_control",
    } or result.best_at(5).startswith("tie: ")
    assert result.best_at(99) == "none"


def test_paired_bootstrap_interval_contains_the_mean() -> None:
    train = build_corpus(range(6), include_clean=False)
    test = build_corpus(range(700, 704), include_clean=False)
    result = ranker_comparison(train, test, ks=(5,))
    mean, low, high = result.paired_bootstrap("learned", "irreversibility_flag", 5)
    assert low <= mean <= high
    with pytest.raises(KeyError, match="no score for"):
        result.paired_bootstrap("learned", "nonsense", 5)


def test_ranker_comparison_validates_inputs() -> None:
    train = build_corpus(range(2), include_clean=False)
    with pytest.raises(ValueError, match="non-empty train and test"):
        ranker_comparison(train, [])
    with pytest.raises(ValueError, match="k=0"):
        ranker_comparison(train, train, ks=(0,))


def test_timing_table_shape_and_validation() -> None:
    cases = [generate_clean_case(seed) for seed in range(2)]
    rows = timing_table(cases, repeats=2)
    assert len(rows) == 2
    for name, steps, forward_ms, lint_ms in rows:
        assert isinstance(name, str)
        assert steps == len(next(c for c in cases if c.name == name).sequence)
        assert forward_ms > 0.0
        assert lint_ms > 0.0
    with pytest.raises(ValueError, match="repeats=0"):
        timing_table(cases, repeats=0)


def test_unused_report_guard() -> None:
    case = generate_clean_case(0)
    assert unused_report_guard(forward_execute(case)) == 0
