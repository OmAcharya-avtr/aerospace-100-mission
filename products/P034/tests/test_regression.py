"""Regression and integration: recorded severities and a small benchmark.

The severities below were recorded from this code in the build session. They
are not reference values from any external source -- they are a tripwire: if a
change to the target, the handlers or the scoring moves them, the change is
deliberate or it is a defect, and either way it must be looked at.
"""

from __future__ import annotations

import numpy as np
import pytest

from faultinject.benchmark import bootstrap_ci, run_benchmark
from faultinject.campaign import FaultCase, build_pool, evaluate_pool, execute_case
from faultinject.faults import Injection
from faultinject.search import coverage_greedy, learned, uniform_random
from faultinject.severity import SEVERE_THRESHOLD
from faultinject.taxonomy import FaultKind

RECORDED = (
    (FaultKind.SENSOR_BIAS, "pos", {"offset": 3.0}, 40, 100, 7, 0.6657654143402159),
    (FaultKind.SENSOR_STUCK, "pos", {}, 50, 50, 11, 0.045107693121047746),
    (
        FaultKind.ACTUATOR_LOSS_EFFECTIVENESS,
        "u",
        {"retained": 0.0},
        30,
        60,
        5,
        0.5899983869240458,
    ),
    (FaultKind.BUS_LOSS, "bus", {"loss_prob": 1.0}, 20, 80, 2, 0.18686147549617582),
    (FaultKind.NUMERICAL_NAN, "vel", {}, 60, 10, 3, 1.0),
    (FaultKind.ACTUATOR_RUNAWAY, "u", {"slew": 5.0}, 40, 40, 9, 0.9186666666666667),
)


@pytest.mark.parametrize(
    ("kind", "channel", "params", "start", "dur", "seed", "expected"), RECORDED
)
def test_recorded_severity(kind, channel, params, start, dur, seed, expected):
    case = FaultCase(Injection.create(kind, channel, params, start, dur), seed, 150)
    got = execute_case(case).severity.severity
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    mean, lo, hi = bootstrap_ci(values, resamples=2000, seed=1)
    assert mean == pytest.approx(3.5, abs=1e-12)
    assert lo < mean < hi


def test_bootstrap_ci_is_deterministic():
    a = bootstrap_ci([1.0, 5.0, 9.0], resamples=500, seed=2)
    b = bootstrap_ci([1.0, 5.0, 9.0], resamples=500, seed=2)
    assert a == b


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_ci([])


def test_small_benchmark_structure():
    rep = run_benchmark(budget=30, n_pools=1, n_seeds=2, replicates=1, warmup=12)
    assert rep.budget == 30
    assert rep.pool_size == 248
    assert set(rep.stats) == {
        "uniform_random",
        "coverage_greedy",
        "kind_mean",
        "learned",
    }
    for st in rep.stats.values():
        assert len(st.scores) == 2
        assert st.lo <= st.mean <= st.hi
        assert all(0 <= s <= 30 for s in st.scores)
    assert rep.verdict() in {
        "no measurable advantage",
        "measurable advantage",
        "baseline uniform_random outperforms learned",
    }
    d = rep.to_dict()
    assert d["severe_threshold"] == SEVERE_THRESHOLD
    assert "verdict_vs_uniform_random" in d


def test_integration_campaign_end_to_end():
    pool = build_pool(pool_seed=1, replicates=1)
    sev = evaluate_pool(pool)

    def oracle(i: int) -> float:
        return sev[i]

    budget = 40
    ur = uniform_random(pool, oracle, budget, np.random.default_rng(5))
    cg = coverage_greedy(pool, oracle, budget, np.random.default_rng(5))
    lr, model = learned(pool, oracle, budget, np.random.default_rng(5), warmup=16)
    for res in (ur, cg, lr):
        assert len(res.order) == budget
        assert all(0.0 <= s <= 1.0 for s in res.severities)
    # The learned strategy must at least never execute a case outside the pool
    # and must produce a usable uncertainty output after the campaign.
    pred = model.predict(list(pool))
    assert np.all(np.isfinite(pred.mean))
    assert np.all(np.isfinite(pred.std))


def test_pool_severe_fraction_is_in_a_plausible_band():
    # Recorded 28.2 % severe for pool_seed 1, replicates 1. A wide band, so this
    # catches a scoring or target change rather than ordinary drift.
    pool = build_pool(pool_seed=1, replicates=1)
    sev = evaluate_pool(pool)
    frac = sum(1 for s in sev if s >= SEVERE_THRESHOLD) / len(sev)
    assert 0.15 <= frac <= 0.45, f"severe fraction {frac:.4f} outside the recorded band"
