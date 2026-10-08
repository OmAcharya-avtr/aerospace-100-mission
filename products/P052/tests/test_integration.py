"""End-to-end: the documented workflow, and one regression benchmark.

The regression test pins a single seeded number so that a change in any layer --
simulator, semantics, strategy, or the bookkeeping between them -- shows up as a
failing test rather than as a quietly different figure in the README.
"""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop import (
    Always,
    ForestSurrogate,
    Predicate,
    Signal,
    bootstrap_band,
    efficiency_curve,
    instance,
    robustness,
    run_benchmark,
    satisfies,
    simulate,
    suite,
    uniform_random,
)
from falsifyloop.systems import LoopInput


def test_the_documented_workflow_runs_end_to_end() -> None:
    # 1. Simulate one setting.
    trace = simulate(LoopInput(5.0, 1.8, 0.3, 2.0, 8.0, 0.9))
    # 2. State a requirement and evaluate it two independent ways.
    requirement = Always(Predicate(Signal("over"), "<=", 2.0, scale=2.0), 0.0, 2.0)
    rho = robustness(requirement, trace)
    assert (rho < 0.0) == (not satisfies(requirement, trace))
    # 3. Search an instance for a violation.
    inst = instance("overshoot-tight")
    result = uniform_random(inst, 200, 0)
    # 4. Build a curve from several seeded runs.
    runs = [uniform_random(inst, 60, s).first_violation for s in range(8)]
    curve = efficiency_curve(runs, 60)
    lower, upper = bootstrap_band(runs, 60, n_boot=200, seed=0)
    assert curve.shape == lower.shape == upper.shape == (60,)
    assert result.simulations <= 200


def test_counterexample_found_by_a_search_reproduces_when_resimulated() -> None:
    # The counterexample is only useful if it can be replayed, so the vector the
    # search returns is fed back through the simulator and the requirement.
    inst = instance("overshoot-tight")
    result = uniform_random(inst, 400, 1)
    assert result.found, "the design target 0.010 should be hit inside 400 draws at this seed"
    replayed = inst.evaluate(result.best_vector)
    assert replayed == pytest.approx(result.best_robustness, rel=0.0, abs=0.0)
    assert replayed < 0.0


def test_every_instance_is_falsifiable_by_the_baseline_given_enough_draws() -> None:
    # Not a correctness property of the package, but a property of the suite: an
    # instance nothing can falsify would make its column of the benchmark table
    # meaningless. The budgets differ because the instances differ by three
    # orders of difficulty.
    budgets = {
        "overshoot-loose": 100,
        "settling-band": 200,
        "multi-requirement": 300,
        "command-rate": 600,
        "overshoot-tight": 1200,
        "nested-capture": 4000,
        "attitude-envelope": 4000,
        "rate-envelope": 8000,
    }
    assert set(budgets) == {inst.identifier for inst in suite()}
    for identifier, budget in budgets.items():
        result = uniform_random(instance(identifier), budget, 0)
        assert result.found, f"{identifier} not falsified in {budget} uniform draws at seed 0"


def test_surrogate_fits_on_real_robustness_values_from_the_simulator() -> None:
    inst = instance("attitude-envelope")
    rng = np.random.default_rng(0)
    points = inst.sample(rng, 40)
    values = np.array([inst.evaluate(p) for p in points])
    assert np.all(np.isfinite(values))
    model = ForestSurrogate(n_estimators=10, random_state=0).fit(points, values)
    mean, spread = model.predict(points[:5])
    assert np.all(np.isfinite(mean))
    assert np.all(spread >= 0.0)


def test_benchmark_regression_pinned_seeded_numbers() -> None:
    """Pinned regression on a small grid. Update only with a stated reason."""
    report = run_benchmark(
        instances=[instance("overshoot-tight")],
        strategy_names=["uniform-random", "surrogate-guided"],
        budget=40,
        repeats=6,
        base_seed=2026,
    )
    baseline = report.cell("overshoot-tight", "uniform-random")
    learned = report.cell("overshoot-tight", "surrogate-guided")
    assert baseline.first_violations == (None, None, None, None, None, None)
    assert learned.first_violations == (34, 22, 36, None, 27, 24)
    assert baseline.mean_curve_probability == pytest.approx(0.0)
    assert learned.mean_curve_probability == pytest.approx(0.258333, abs=1e-6)
