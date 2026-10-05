"""Search strategies: budget discipline, coverage behaviour, validation."""

from __future__ import annotations

import numpy as np
import pytest

from faultinject.campaign import build_pool, evaluate_pool
from faultinject.search import (
    STRATEGIES,
    coverage_greedy,
    kind_mean,
    learned,
    uniform_random,
)
from faultinject.severity import SEVERE_THRESHOLD


@pytest.fixture(scope="module")
def pool_and_oracle():
    pool = build_pool(pool_seed=1, replicates=1)
    sev = evaluate_pool(pool)
    return pool, (lambda i: sev[i]), sev


def test_strategy_names():
    assert STRATEGIES == ("uniform_random", "coverage_greedy", "kind_mean", "learned")


def test_uniform_random_respects_budget(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    res = uniform_random(pool, oracle, 30, np.random.default_rng(1))
    assert len(res.order) == 30
    assert len(set(res.order)) == 30
    assert len(res.severities) == 30
    assert res.strategy == "uniform_random"


def test_found_curve_is_monotone(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    res = uniform_random(pool, oracle, 40, np.random.default_rng(2))
    curve = res.found_curve
    assert curve == sorted(curve)
    assert curve[-1] == res.n_severe
    assert res.n_severe == sum(1 for s in res.severities if s >= SEVERE_THRESHOLD)


def test_coverage_curve_is_monotone_and_bounded(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    res = coverage_greedy(pool, oracle, 40, np.random.default_rng(3))
    assert res.coverage_curve == sorted(res.coverage_curve)
    assert 0.0 < res.coverage_curve[-1] <= 1.0


def test_coverage_greedy_covers_a_new_cell_every_step_while_it_can(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    budget = 40
    res = coverage_greedy(pool, oracle, budget, np.random.default_rng(4))
    # The pool has one case per cell, so every pick must open a new cell.
    assert res.coverage_curve[-1] == pytest.approx(budget / 248, rel=1e-12)


def test_coverage_greedy_beats_random_on_coverage(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    cg = coverage_greedy(pool, oracle, 60, np.random.default_rng(5))
    ur = uniform_random(pool, oracle, 60, np.random.default_rng(5))
    assert cg.coverage_curve[-1] >= ur.coverage_curve[-1]


def test_kind_mean_respects_budget(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    res = kind_mean(pool, oracle, 40, np.random.default_rng(6), warmup=16)
    assert len(res.order) == 40
    assert len(set(res.order)) == 40


def test_kind_mean_shrinks_warmup_when_budget_is_small(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    res = kind_mean(pool, oracle, 6, np.random.default_rng(7), warmup=16)
    assert len(res.order) == 6


def test_learned_respects_budget_and_returns_a_model(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    res, model = learned(pool, oracle, 40, np.random.default_rng(8), warmup=16)
    assert len(res.order) == 40
    assert len(set(res.order)) == 40
    assert model.n_train >= 16
    assert res.strategy == "learned"


def test_learned_is_deterministic_for_a_fixed_seed(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    a, _ = learned(pool, oracle, 36, np.random.default_rng(9), warmup=16)
    b, _ = learned(pool, oracle, 36, np.random.default_rng(9), warmup=16)
    assert a.order == b.order
    assert a.severities == b.severities


def test_learned_exploration_kappa_changes_the_order(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    a, _ = learned(pool, oracle, 36, np.random.default_rng(10), warmup=16, kappa=0.0)
    b, _ = learned(pool, oracle, 36, np.random.default_rng(10), warmup=16, kappa=3.0)
    assert a.order != b.order


def test_search_validation(pool_and_oracle):
    pool, oracle, _ = pool_and_oracle
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="budget must be"):
        uniform_random(pool, oracle, 0, rng)
    with pytest.raises(ValueError, match="exceeds pool size"):
        uniform_random(pool, oracle, len(pool) + 1, rng)
    with pytest.raises(ValueError, match="warmup must be"):
        learned(pool, oracle, 20, rng, warmup=1)
    with pytest.raises(ValueError, match="must be smaller than budget"):
        learned(pool, oracle, 20, rng, warmup=20)
    with pytest.raises(ValueError, match="refit_every"):
        learned(pool, oracle, 20, rng, warmup=5, refit_every=0)


def test_mean_severity_of_empty_result():
    from faultinject.search import SearchResult

    assert SearchResult("x", [], []).mean_severity == 0.0
