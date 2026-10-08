"""Rendering: content and shape of the strings the CLI prints."""

from __future__ import annotations

import numpy as np
import pytest

from falsifyloop.benchmark import run_benchmark, run_cell
from falsifyloop.curves import bootstrap_band, efficiency_curve
from falsifyloop.instances import instance, suite
from falsifyloop.report import (
    ONE_SIDED_NOTE,
    render_aggregate_table,
    render_cell_detail,
    render_cell_table,
    render_curve_points,
    render_difficulty_table,
    render_instances,
    render_search_result,
)
from falsifyloop.search import BASELINE, uniform_random

EASY = instance("overshoot-loose")
HARD = instance("rate-envelope")


def test_one_sided_note_is_the_exact_required_sentence() -> None:
    assert ONE_SIDED_NOTE == (
        "Falsification is one-sided: finding no violation is not evidence of correctness."
    )


def test_render_found_violation_reports_the_index_and_counterexample() -> None:
    result = uniform_random(EASY, 60, 0)
    assert result.found
    text = render_search_result(result, EASY)
    assert f"VIOLATION FOUND at simulation {result.first_violation}" in text
    assert "counterexample" in text
    assert "step_amplitude" in text


def test_render_no_violation_states_the_one_sidedness() -> None:
    result = uniform_random(HARD, 5, 0)
    assert not result.found
    text = render_search_result(result, HARD)
    assert "NO VIOLATION FOUND within 5 simulations." in text
    assert ONE_SIDED_NOTE in text
    assert "statement about the search" in text


def test_render_instances_covers_every_instance() -> None:
    text = render_instances(suite())
    for inst in suite():
        assert inst.identifier in text


def test_render_cell_table_has_a_row_per_cell() -> None:
    report = run_benchmark(
        instances=[EASY, HARD],
        strategy_names=[BASELINE, "latin-hypercube"],
        budget=15,
        repeats=3,
        base_seed=1,
    )
    text = render_cell_table(report)
    for iid in report.instance_ids:
        for name in report.strategy_names:
            assert any(iid in line and name in line for line in text.splitlines())
    assert "budget 15 simulations" in text


def test_render_aggregate_table_names_the_baseline_and_the_losses() -> None:
    report = run_benchmark(
        instances=[EASY, HARD],
        strategy_names=[BASELINE, "cross-entropy"],
        budget=15,
        repeats=3,
        base_seed=2,
    )
    text = render_aggregate_table(report)
    assert "(is the baseline)" in text
    assert "Hardest instance for the baseline:" in text
    assert ONE_SIDED_NOTE in text
    assert "Read the per-instance table first" in text


def test_render_difficulty_table_known_answer_row() -> None:
    text = render_difficulty_table((("inst", "hard", 5, 1000, 0.004),))
    line = [row for row in text.splitlines() if row.startswith("inst ")][0]
    assert "0.005000" in line  # 5/1000 point estimate
    assert "1000" in line
    assert "0.0040" in line  # the design target column


def test_render_curve_points_known_answer() -> None:
    runs = (2, 5, None, None)
    curve = efficiency_curve(runs, 10)
    lower, upper = bootstrap_band(runs, 10, n_boot=200, seed=0)
    text = render_curve_points(curve, lower, upper, marks=(1, 5, 10))
    assert "0.0000" in text  # P(found by 1) = 0
    assert text.count("\n") == 4  # header, rule, three rows


def test_render_curve_points_validation() -> None:
    curve = np.zeros(5)
    with pytest.raises(ValueError, match="same shape"):
        render_curve_points(curve, np.zeros(4), np.zeros(5), marks=(1,))
    with pytest.raises(ValueError, match="outside"):
        render_curve_points(curve, curve, curve, marks=(9,))


def test_render_cell_detail_shows_censored_runs_as_none() -> None:
    cell, _ = run_cell(HARD, BASELINE, budget=5, repeats=3, base_seed=0)
    text = render_cell_detail(cell)
    assert "none" in text
    assert ONE_SIDED_NOTE in text
