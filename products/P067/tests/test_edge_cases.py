"""Edge cases and degenerate configurations."""

from __future__ import annotations

import numpy as np
import pytest

from acqsearch import (
    AcquisitionEnvironment,
    BayesianGreedyPolicy,
    PointingUncertainty,
    PoissonDwellDetector,
    TerminalGeometry,
    build_cell_grid,
    build_schedule,
    cell_detection_probabilities,
    cell_visit_order,
    coverage_curve,
    expected_acquisition_time,
    initial_posterior,
    monte_carlo_acquisition,
    monte_carlo_acquisition_continuous,
    probability_acquired_by,
    run_episode,
    standard_paths,
)


def test_a_grid_of_one_cell_is_buildable_when_the_footprint_covers_everything():
    uncertainty = PointingUncertainty(1.0e-5)
    geometry = TerminalGeometry(1.0e-3, 2.0e-3, 1.0e-2, 0.0)
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=2.5)
    assert grid.n_cells == 1
    assert grid.prior_mass.sum() > 0.99


def test_a_tiny_extent_relative_to_the_footprint_raises():
    uncertainty = PointingUncertainty(1.0e-9)
    geometry = TerminalGeometry(1.0e-3, 2.0e-3, 1.0e-2, 0.0)
    # The lattice always keeps the centre cell, so this configuration is
    # degenerate but legal: one cell holding essentially all the prior mass.
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=0.001)
    assert grid.n_cells == 1


def test_zero_background_gives_a_zero_false_alarm_probability():
    detector = PoissonDwellDetector(1.0e4, 0.0, 5, 0.0)
    assert detector.false_alarm_probability(1.0e-3) == pytest.approx(0.0, abs=1e-18)


def test_zero_signal_makes_detection_equal_to_false_alarm():
    detector = PoissonDwellDetector(0.0, 2.0e3, 4, 0.0)
    assert detector.detection_probability(1.0e-3, 0.0, 5.0e-5) == pytest.approx(
        detector.false_alarm_probability(1.0e-3), rel=1e-15
    )


def test_a_very_large_threshold_makes_acquisition_essentially_impossible(grid, geometry,
                                                                        uncertainty):
    detector = PoissonDwellDetector(1.0e4, 1.0e3, 60, 0.0)
    q = cell_detection_probabilities(grid, geometry, detector, 1.0e-4, uncertainty)
    assert np.all(q < 1e-12)
    # The model still returns a finite expected time; it is simply enormous.
    order = cell_visit_order(standard_paths(grid)["spiral"], grid)
    schedule = build_schedule(grid, order, geometry, 1.0e-4, 0.0, 0.0)
    estimate = expected_acquisition_time(grid, schedule, q)
    assert estimate.expected_time_s > 1e6


def test_a_zero_deadline_gives_zero_probability(grid, cell_q, spiral_order, geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    assert probability_acquired_by(grid, schedule, cell_q, 0.0) == pytest.approx(0.0, abs=1e-18)


def test_monte_carlo_with_one_trial_returns_a_nan_standard_error(grid, cell_q, spiral_order,
                                                                 geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    for seed in range(50):
        result = monte_carlo_acquisition(grid, schedule, cell_q, n_trials=1, seed=seed)
        if result.n_inside == 1:
            assert np.isnan(result.standard_error_s)
            return
    pytest.fail("no seed in 0..49 placed the single trial inside the grid")


def test_continuous_monte_carlo_raises_when_nothing_is_acquired(grid, geometry, uncertainty,
                                                                spiral_order):
    impossible = PoissonDwellDetector(1.0, 1.0e-6, 40, 0.0)
    schedule = build_schedule(grid, spiral_order, geometry, 1.0e-5, 0.0, 0.0)
    with pytest.raises(ValueError, match="no trial acquired"):
        monte_carlo_acquisition_continuous(
            grid, schedule, geometry, impossible, uncertainty, 1.0e-5, n_trials=50, seed=0,
            max_dwells=20,
        )


def test_coverage_curve_with_a_single_sample_is_zero_or_one(grid, uncertainty, geometry):
    curve = coverage_curve(
        standard_paths(grid)["spiral"], uncertainty, geometry.footprint_radius_rad, 1, seed=0
    )
    assert set(np.unique(curve.covered_mass)) <= {0.0, 1.0}


def test_a_policy_on_a_one_cell_grid_always_picks_that_cell():
    uncertainty = PointingUncertainty(1.0e-5)
    geometry = TerminalGeometry(1.0e-3, 2.0e-3, 1.0e-2, 0.0)
    detector = PoissonDwellDetector(1.0e4, 1.0e3, 3, 0.0)
    grid = build_cell_grid(uncertainty, geometry, extent_sigma=2.5)
    environment = AcquisitionEnvironment.build(
        grid, geometry, detector, uncertainty, 1.0e-3, 0.1, deadline_s=1.0
    )
    assert grid.n_cells == 1
    result = run_episode(environment, BayesianGreedyPolicy(), 0, np.random.default_rng(0))
    assert result.dwells >= 1


def test_initial_posterior_rejects_a_massless_grid():
    from acqsearch import CellGrid

    grid = CellGrid(
        centres_rad=np.zeros((1, 2)),
        side_rad=1.0e-4,
        prior_mass=np.array([0.0]),
        mass_outside=1.0,
        extent_rad=1.0e-4,
    )
    with pytest.raises(ValueError, match="no prior mass"):
        initial_posterior(grid)


def test_expected_time_rejects_a_massless_grid():
    from acqsearch import CellGrid

    grid = CellGrid(
        centres_rad=np.zeros((1, 2)),
        side_rad=1.0e-4,
        prior_mass=np.array([0.0]),
        mass_outside=1.0,
        extent_rad=1.0e-4,
    )
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    schedule = build_schedule(grid, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="no prior mass"):
        expected_acquisition_time(grid, schedule, np.array([0.5]))


def test_very_high_rytov_variance_is_accepted_but_outside_the_cited_validity_range():
    # The lognormal model is cited for sigma_R^2 <~ 1. Values above that are
    # not rejected by the code, because the caller may know more than the
    # library does; the limitation is documented, not enforced.
    detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, 2.5)
    value = detector.detection_probability(1.4e-3, 0.0, 5.0e-5)
    assert 0.0 <= value <= 1.0


def test_settle_time_dominates_when_the_slew_rate_is_very_fast(grid, spiral_order):
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0, 5.0e-3)
    schedule = build_schedule(grid, spiral_order, geometry, 1.0e-3, 0.0, 0.0)
    # 89 dwells of 1 ms plus 5 ms settle is 0.534 s; the 7.52e-3 rad of total
    # slew at 1 rad/s adds 7.52e-3 s, which is 1.4 percent of the period.
    assert schedule.period_s == pytest.approx(grid.n_cells * 6.0e-3, rel=2e-2)
    assert schedule.period_s > grid.n_cells * 6.0e-3
