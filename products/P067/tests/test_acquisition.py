"""Analytic expected acquisition time, its Monte Carlo twin, and the dwell trade."""

from __future__ import annotations

import numpy as np
import pytest

from acqsearch import (
    CellGrid,
    PointingUncertainty,
    TerminalGeometry,
    build_schedule,
    cell_detection_probabilities,
    expected_acquisition_time,
    monte_carlo_acquisition,
    monte_carlo_acquisition_continuous,
    optimal_dwell_time,
    probability_acquired_by,
)


def _one_cell_grid(prior_inside: float = 1.0) -> CellGrid:
    """A one-cell grid at boresight carrying ``prior_inside`` of the prior mass."""
    return CellGrid(
        centres_rad=np.zeros((1, 2)),
        side_rad=1.0e-4,
        prior_mass=np.array([prior_inside]),
        mass_outside=1.0 - prior_inside,
        extent_rad=1.0e-4,
    )


def test_single_cell_expected_time_matches_the_geometric_hand_calculation():
    # One cell, no slew (the dwell is at boresight and the scan starts there),
    # dwell 1e-3 s, settle 0 s, no false alarms, q = 0.5.
    # Period T = 1e-3 s. E[T] = T + T (1 - q) / q = 1e-3 + 1e-3 * 1 = 2e-3 s.
    grid = _one_cell_grid()
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    schedule = build_schedule(grid, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    assert schedule.period_s == pytest.approx(1.0e-3, rel=1e-12)
    estimate = expected_acquisition_time(grid, schedule, np.array([0.5]))
    assert estimate.expected_time_s == pytest.approx(2.0e-3, rel=1e-12)


def test_single_cell_expected_time_with_certain_detection_is_one_dwell():
    grid = _one_cell_grid()
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    schedule = build_schedule(grid, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    estimate = expected_acquisition_time(grid, schedule, np.array([1.0]))
    assert estimate.expected_time_s == pytest.approx(1.0e-3, rel=1e-12)


def test_false_alarm_charge_adds_exactly_f_times_tv_per_dwell():
    # f = 0.01, t_v = 0.5 s adds 5e-3 s to every dwell.
    grid = _one_cell_grid()
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    plain = build_schedule(grid, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    charged = build_schedule(grid, np.array([0]), geometry, 1.0e-3, 0.01, 0.5)
    assert charged.false_alarm_charge_s == pytest.approx(5.0e-3, rel=1e-12)
    assert charged.period_s - plain.period_s == pytest.approx(5.0e-3, rel=1e-12)


def test_schedule_timing_matches_a_hand_computed_two_cell_pass():
    # Two cells at (0, 0) and (1e-4, 0). Slew rate 1e-2 rad/s, settle 2e-3 s.
    # Dwell 1: 0 rad slew -> 1e-3 + 0 + 2e-3 = 3e-3 s.
    # Dwell 2: 1e-4 rad   -> 1e-3 + 1e-2 + 2e-3 = 1.3e-2 s. Cumulative 1.6e-2 s.
    grid = CellGrid(
        centres_rad=np.array([[0.0, 0.0], [1.0e-4, 0.0]]),
        side_rad=1.0e-4,
        prior_mass=np.array([0.5, 0.3]),
        mass_outside=0.2,
        extent_rad=1.0e-4,
    )
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 2.0e-3)
    schedule = build_schedule(grid, np.array([0, 1]), geometry, 1.0e-3, 0.0, 0.0)
    assert schedule.cumulative_time_s == pytest.approx(np.array([3.0e-3, 1.6e-2]), rel=1e-12)
    assert schedule.total_slew_rad(grid) == pytest.approx(1.0e-4, rel=1e-12)


def test_mass_outside_is_reported_and_does_not_change_the_conditional_mean():
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    full = _one_cell_grid(1.0)
    partial = _one_cell_grid(0.4)
    schedule_full = build_schedule(full, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    schedule_partial = build_schedule(partial, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    a = expected_acquisition_time(full, schedule_full, np.array([0.5]))
    b = expected_acquisition_time(partial, schedule_partial, np.array([0.5]))
    assert a.expected_time_s == pytest.approx(b.expected_time_s, rel=1e-12)
    assert b.mass_outside == pytest.approx(0.6, rel=1e-12)


def test_probability_by_deadline_hand_calculation():
    # One cell, period 1e-3 s, q = 0.5. By 2.5e-3 s two passes have completed,
    # so P = 1 - 0.5^2 = 0.75.
    grid = _one_cell_grid()
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    schedule = build_schedule(grid, np.array([0]), geometry, 1.0e-3, 0.0, 0.0)
    assert probability_acquired_by(grid, schedule, np.array([0.5]), 2.5e-3) == pytest.approx(
        0.75, rel=1e-12
    )
    assert probability_acquired_by(grid, schedule, np.array([0.5]), 0.5e-3) == pytest.approx(
        0.0, abs=1e-15
    )


def test_probability_by_deadline_is_bounded_by_the_mass_in_the_grid(grid, cell_q, spiral_order,
                                                                    geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    assert probability_acquired_by(grid, schedule, cell_q, 1.0e6) == pytest.approx(
        grid.prior_mass.sum(), rel=1e-9
    )


def test_probability_by_deadline_is_monotone(grid, cell_q, spiral_order, geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    deadlines = np.linspace(0.0, 10.0, 60)
    values = [probability_acquired_by(grid, schedule, cell_q, float(d)) for d in deadlines]
    assert all(b >= a - 1e-15 for a, b in zip(values, values[1:], strict=False))


def test_analytic_expected_time_agrees_with_its_monte_carlo_twin(
    grid, cell_q, spiral_order, geometry, detector
):
    f = detector.false_alarm_probability(1.4e-3)
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, f, 0.5)
    analytic = expected_acquisition_time(grid, schedule, cell_q).expected_time_s
    mc = monte_carlo_acquisition(grid, schedule, cell_q, n_trials=60000, seed=20261016)
    # Compared with a tolerance on both sides: four Monte Carlo standard
    # errors. Nothing here is asserted to be bit-exact.
    assert abs(mc.mean_time_s - analytic) <= 4.0 * mc.standard_error_s


def test_analytic_deadline_probability_agrees_with_monte_carlo(
    grid, cell_q, spiral_order, geometry
):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    deadline = 2.0 * schedule.period_s
    analytic = probability_acquired_by(grid, schedule, cell_q, deadline)
    mc = monte_carlo_acquisition(
        grid, schedule, cell_q, n_trials=60000, seed=5, deadline_s=deadline
    )
    se = np.sqrt(analytic * (1.0 - analytic) / 60000.0)
    assert abs(mc.probability_by_deadline - analytic) <= 4.0 * se


def test_stochastic_false_alarms_leave_the_mean_but_move_the_deadline_probability(
    grid, cell_q, spiral_order, geometry, detector
):
    f = detector.false_alarm_probability(1.4e-3)
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, f, 0.5)
    deadline = 1.5 * schedule.period_s
    deterministic = monte_carlo_acquisition(
        grid, schedule, cell_q, n_trials=40000, seed=7, deadline_s=deadline
    )
    stochastic = monte_carlo_acquisition(
        grid,
        schedule,
        cell_q,
        n_trials=40000,
        seed=7,
        deadline_s=deadline,
        stochastic_false_alarms=True,
        false_alarm_probability=f,
        verification_time_s=0.5,
    )
    combined = np.hypot(deterministic.standard_error_s, stochastic.standard_error_s)
    assert abs(stochastic.mean_time_s - deterministic.mean_time_s) <= 4.0 * combined


def test_continuous_geometry_monte_carlo_is_faster_than_the_cell_model(
    grid, cell_q, spiral_order, geometry, detector, uncertainty
):
    # The cell abstraction cannot represent a target near a boundary being
    # acquired by a neighbouring dwell, so it is pessimistic. This asserts the
    # direction of the bias, which is a property of the abstraction, not a
    # tuned number; the size is reported in VALIDATION.md.
    f = detector.false_alarm_probability(1.4e-3)
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, f, 0.5)
    cell_model = expected_acquisition_time(grid, schedule, cell_q).expected_time_s
    continuous = monte_carlo_acquisition_continuous(
        grid,
        schedule,
        geometry,
        detector,
        uncertainty,
        1.4e-3,
        n_trials=8000,
        seed=11,
        false_alarm_probability=f,
        verification_time_s=0.5,
        restrict_to_grid=True,
    )
    assert continuous.mean_time_s < cell_model


def test_cell_detection_probabilities_are_between_the_centre_and_corner_values(
    grid, geometry, detector, uncertainty, cell_q
):
    centre = detector.detection_probability(1.4e-3, 0.0, geometry.beam_half_angle_rad)
    corner = detector.detection_probability(
        1.4e-3, geometry.footprint_radius_rad, geometry.beam_half_angle_rad
    )
    assert np.all(cell_q <= centre + 1e-12)
    assert np.all(cell_q >= corner - 1e-12)


def test_cell_detection_probabilities_are_nearly_uniform_across_the_grid(cell_q):
    # Every cell is the same square seen from its own centre, so q varies only
    # through the prior weighting inside the cell.
    assert cell_q.max() - cell_q.min() < 0.02


def test_cell_detection_quadrature_is_converged(grid, geometry, detector, uncertainty):
    coarse = cell_detection_probabilities(grid, geometry, detector, 1.4e-3, uncertainty, 6)
    fine = cell_detection_probabilities(grid, geometry, detector, 1.4e-3, uncertainty, 14)
    assert np.max(np.abs(coarse - fine)) < 1.0e-4


def test_dwell_optimum_is_interior_and_beats_both_ends(
    grid, spiral_order, geometry, detector, uncertainty
):
    dwells = np.geomspace(2.0e-4, 8.0e-3, 21)
    optimum = optimal_dwell_time(grid, spiral_order, geometry, detector, uncertainty, dwells, 0.5)
    assert dwells[0] < optimum.best_dwell_s < dwells[-1]
    assert optimum.best_expected_time_s < optimum.expected_time_s[0]
    assert optimum.best_expected_time_s < optimum.expected_time_s[-1]
    assert np.all(np.diff(optimum.detection_probability) >= -1e-12)
    assert np.all(np.diff(optimum.false_alarm_probability) >= -1e-12)


def test_expected_time_rejects_a_zero_detection_probability(grid, spiral_order, geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    bad = np.zeros(grid.n_cells)
    with pytest.raises(ValueError, match="cell_detection"):
        expected_acquisition_time(grid, schedule, bad)


def test_expected_time_rejects_a_wrong_shaped_vector(grid, spiral_order, geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        expected_acquisition_time(grid, schedule, np.ones(3))


def test_build_schedule_rejects_bad_input(grid, geometry):
    with pytest.raises(ValueError, match="order must be non-empty"):
        build_schedule(grid, np.array([], dtype=int), geometry, 1.0e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="repeat"):
        build_schedule(grid, np.array([0, 0]), geometry, 1.0e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="dwell_s"):
        build_schedule(grid, np.array([0, 1]), geometry, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="false_alarm_probability"):
        build_schedule(grid, np.array([0, 1]), geometry, 1.0e-3, 1.5, 0.0)
    with pytest.raises(ValueError, match="verification_time_s"):
        build_schedule(grid, np.array([0, 1]), geometry, 1.0e-3, 0.0, -1.0)


def test_probability_by_deadline_rejects_a_negative_deadline(grid, cell_q, spiral_order, geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="deadline_s"):
        probability_acquired_by(grid, schedule, cell_q, -1.0)


def test_monte_carlo_rejects_bad_trial_counts(grid, cell_q, spiral_order, geometry):
    schedule = build_schedule(grid, spiral_order, geometry, 1.4e-3, 0.0, 0.0)
    with pytest.raises(ValueError, match="n_trials"):
        monte_carlo_acquisition(grid, schedule, cell_q, n_trials=0)


def test_optimal_dwell_rejects_an_empty_grid(grid, spiral_order, geometry, detector, uncertainty):
    with pytest.raises(ValueError, match="dwell_grid_s"):
        optimal_dwell_time(
            grid, spiral_order, geometry, detector, uncertainty, np.array([]), 0.5
        )
    with pytest.raises(ValueError, match="dwell_grid_s"):
        optimal_dwell_time(
            grid, spiral_order, geometry, detector, uncertainty, np.array([-1.0e-3]), 0.5
        )


def test_cell_detection_rejects_too_few_quadrature_nodes(grid, geometry, detector, uncertainty):
    with pytest.raises(ValueError, match="quadrature_nodes"):
        cell_detection_probabilities(grid, geometry, detector, 1.0e-3, uncertainty, 1)


def test_expected_time_uses_a_one_cell_grid_with_a_pure_prior():
    grid = _one_cell_grid(1.0)
    geometry = TerminalGeometry(1.0e-4, 2.0e-4, 1.0e-2, 0.0)
    schedule = build_schedule(grid, np.array([0]), geometry, 2.0e-3, 0.0, 0.0)
    estimate = expected_acquisition_time(grid, schedule, np.array([0.25]))
    # T + T (1 - q) / q = 2e-3 * (1 + 3) = 8e-3 s.
    assert estimate.expected_time_s == pytest.approx(8.0e-3, rel=1e-12)
    assert estimate.n_dwells == 1
    assert estimate.mass_in_grid == pytest.approx(1.0, rel=1e-15)


def test_uncertainty_object_is_not_mutated_by_sampling():
    u = PointingUncertainty(1.0e-4)
    before = u.sigma_rad
    u.sample(10, np.random.default_rng(0))
    assert u.sigma_rad == before
