"""The instance-by-strategy grid: integration, accounting, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.benchmark import run_benchmark, run_cell
from falsifyloop.instances import instance
from falsifyloop.search import BASELINE

EASY = instance("overshoot-loose")
HARD = instance("overshoot-tight")


def test_run_cell_summary_fields() -> None:
    cell, results = run_cell(EASY, BASELINE, budget=30, repeats=4, base_seed=100)
    assert cell.instance_id == EASY.identifier
    assert cell.strategy_name == BASELINE
    assert cell.budget == 30
    assert cell.repeats == 4
    assert len(cell.first_violations) == 4
    assert len(results) == 4
    assert 0.0 <= cell.success_rate <= 1.0
    assert 0.0 <= cell.mean_curve_probability <= 1.0
    assert cell.best_robustness == pytest.approx(min(r.best_robustness for r in results))


def test_run_cell_uses_consecutive_seeds_from_the_base() -> None:
    _, results = run_cell(EASY, BASELINE, budget=10, repeats=3, base_seed=55)
    assert [r.seed for r in results] == [55, 56, 57]


def test_run_cell_curve_and_band_agree_with_the_summary() -> None:
    cell, _ = run_cell(HARD, BASELINE, budget=40, repeats=6, base_seed=7)
    curve = cell.curve()
    assert curve.shape == (40,)
    assert curve[-1] == pytest.approx(cell.success_rate)
    lower, upper = cell.band(n_boot=200, seed=1)
    assert np.all(lower <= curve + 1e-12)
    assert np.all(upper >= curve - 1e-12)


def test_run_cell_rejects_a_single_repeat() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        run_cell(EASY, BASELINE, budget=10, repeats=1, base_seed=0)


def test_run_cell_rejects_a_zero_budget() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run_cell(EASY, BASELINE, budget=0, repeats=2, base_seed=0)


def test_run_cell_passes_options_through_to_the_strategy() -> None:
    plain, _ = run_cell(HARD, "surrogate-guided", 12, 2, 3)
    tuned, _ = run_cell(HARD, "surrogate-guided", 12, 2, 3, options={"n_initial": 12})
    lhs, _ = run_cell(HARD, "latin-hypercube", 12, 2, 3)
    # With n_initial equal to the budget the surrogate strategy degenerates to
    # its Latin hypercube warm start, which is a check that options arrive.
    assert tuned.first_violations == lhs.first_violations
    assert isinstance(plain.mean_curve_probability, float)


def test_run_benchmark_covers_the_requested_grid() -> None:
    report = run_benchmark(
        instances=[EASY, HARD],
        strategy_names=[BASELINE, "latin-hypercube"],
        budget=20,
        repeats=3,
        base_seed=200,
    )
    assert report.instance_ids == (EASY.identifier, HARD.identifier)
    assert report.strategy_names == (BASELINE, "latin-hypercube")
    assert len(report.cells) == 4
    assert report.budget == 20
    assert report.repeats == 3
    assert report.base_seed == 200
    assert report.wall_clock_seconds > 0.0


def test_run_benchmark_is_deterministic() -> None:
    kwargs = {
        "instances": [HARD],
        "strategy_names": [BASELINE, "cross-entropy"],
        "budget": 20,
        "repeats": 3,
        "base_seed": 11,
    }
    first = run_benchmark(**kwargs)
    second = run_benchmark(**kwargs)
    for key in first.cells:
        assert first.cells[key].first_violations == second.cells[key].first_violations


def test_report_lookup_and_aggregate_views() -> None:
    report = run_benchmark(
        instances=[EASY, HARD],
        strategy_names=[BASELINE, "simulated-annealing"],
        budget=25,
        repeats=4,
        base_seed=300,
    )
    cell = report.cell(EASY.identifier, BASELINE)
    assert cell.strategy_name == BASELINE
    per_instance = report.per_instance(BASELINE)
    assert set(per_instance) == set(report.instance_ids)
    curve = report.aggregate_curve(BASELINE)
    assert curve.shape == (25,)
    lower, upper = report.aggregate_band(BASELINE, n_boot=200, seed=0)
    assert np.all(lower <= curve + 1e-12)
    assert np.all(upper >= curve - 1e-12)
    assert report.aggregate_mean_probability(BASELINE) == pytest.approx(float(curve.mean()))


def test_baseline_wins_names_instances_and_the_baseline_never_beats_itself() -> None:
    report = run_benchmark(
        instances=[EASY, HARD],
        strategy_names=[BASELINE, "simulated-annealing"],
        budget=25,
        repeats=4,
        base_seed=300,
    )
    assert report.baseline_wins(BASELINE) == ()
    losses = report.baseline_wins("simulated-annealing")
    assert set(losses) <= set(report.instance_ids)


def test_hardest_instance_is_the_one_the_baseline_does_worst_on() -> None:
    report = run_benchmark(
        instances=[EASY, HARD],
        strategy_names=[BASELINE],
        budget=30,
        repeats=5,
        base_seed=400,
    )
    # The design targets are 0.31 and 0.010, so the baseline must do worse on
    # the tight instance. This is the one place the suite's difficulty ordering
    # is checked end to end through an actual search rather than by declaration.
    assert report.hardest_instance() == HARD.identifier


def test_report_lookup_of_a_missing_cell_raises() -> None:
    report = run_benchmark(
        instances=[EASY], strategy_names=[BASELINE], budget=10, repeats=2, base_seed=1
    )
    with pytest.raises(KeyError, match="no cell for instance"):
        report.cell(EASY.identifier, "cross-entropy")


def test_run_benchmark_validation() -> None:
    with pytest.raises(ValueError, match="at least one instance"):
        run_benchmark(instances=[], budget=5, repeats=2)
    with pytest.raises(ValueError, match="at least one strategy"):
        run_benchmark(instances=[EASY], strategy_names=[], budget=5, repeats=2)
    with pytest.raises(KeyError, match="unknown strategies"):
        run_benchmark(instances=[EASY], strategy_names=["nope"], budget=5, repeats=2)


def test_default_benchmark_runs_the_whole_suite() -> None:
    report = run_benchmark(budget=4, repeats=2, base_seed=9, strategy_names=[BASELINE])
    assert len(report.instance_ids) >= 6
