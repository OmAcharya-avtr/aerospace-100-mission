"""Pinned regression tests.

Two kinds of thing live here.

1. Every falsifying example Hypothesis has found in this repository, written
   out with literal inputs. ``.hypothesis/`` is gitignored, so an example that
   is only in the local database is lost on a fresh clone and the limit it
   exposed is rediscovered the hard way.
2. The places where a strict-threshold boolean could be asserted and must not
   be. A boolean produced by comparing a floating-point statistic with a
   threshold is a step function of a rounded quantity, so the statistic is
   compared with a tolerance and the boolean is allowed to differ only inside
   that tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from acqsearch import (
    PointingUncertainty,
    PoissonDwellDetector,
    build_cell_grid,
    build_schedule,
    expected_acquisition_time,
    monte_carlo_acquisition,
    spiral_closed_form_coverage,
    spiral_path,
)

# ----------------------------------------------------------------------------
# 1. Hypothesis falsifiers, pinned with their literal inputs.
# ----------------------------------------------------------------------------


def test_regression_spiral_closed_form_cancellation_at_a_vanishing_pitch():
    """Hypothesis, 2026-10-16: pitch = 1e-12, turns = 1, a = sigma = 0.0078125 rad.

    The closed form is ``1 - (sigma / p) sqrt(pi / 2) [erf(u1 / (sigma sqrt 2))
    - erf(u0 / (sigma sqrt 2))]``. As ``p`` goes to zero the bracket goes to
    zero and the division by ``p`` amplifies the rounding of the difference.
    At ``p / sigma = 1.28e-10`` the result lands 5.40e-7 BELOW the exact
    limit, which is the one-disc Rayleigh mass, and therefore 5.40e-7 below
    the geometric lower bound the value can never actually be below.

    This is a numerical limit of the expression, not an error in the geometry,
    and it is outside the range any scan uses: the pitch of a real spiral is
    within an order of magnitude of the footprint. The guarantee that holds is
    an absolute error of 1e-6, and that is what is asserted.
    """
    u = PointingUncertainty(0.0078125)
    value = spiral_closed_form_coverage(1e-12, 1, 0.0078125, u)
    exact = u.radial_cdf(0.0078125)
    assert value == pytest.approx(0.3934688006264304, abs=1e-15)
    assert exact == pytest.approx(0.3934693402873666, abs=1e-15)
    assert abs(value - exact) <= 1e-6
    # The strict geometric bound is violated, by this much and no more.
    assert exact - value == pytest.approx(5.396609361e-07, abs=1e-14)


def test_regression_closed_form_is_accurate_once_the_pitch_is_realistic():
    """The same expression at ``pitch / sigma >= 1e-6`` is accurate.

    Measured at the same sigma as the falsifier above: the error against the
    exact limit is 3.882e-5 at ``p / sigma = 1.28e-4`` and falls linearly with
    the pitch, which is the true mathematical difference and not rounding.
    """
    u = PointingUncertainty(0.0078125)
    value = spiral_closed_form_coverage(1e-6, 1, 0.0078125, u)
    assert value >= u.radial_cdf(0.0078125)
    assert value == pytest.approx(0.3935081582502524, rel=1e-12)


def test_regression_first_spiral_chord_is_much_shorter_than_the_arc_spacing():
    """Found while writing ``tests/test_patterns.py``: uniform ARC sampling of an
    Archimedean spiral gives a first chord of 0.6624 of the spacing, because
    the curvature near the origin is extreme. Any test that assumes uniform
    chord length fails on the first few points; the guarantee is that the
    chord never exceeds the arc spacing and rises monotonically towards it.
    """
    path = spiral_path(7.0e-5, 5.0, 3.5e-5)
    steps = np.diff(path.points_rad, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    assert lengths[0] / 3.5e-5 == pytest.approx(0.662353793307427, rel=1e-9)
    assert lengths[1] / 3.5e-5 == pytest.approx(0.936417834, rel=1e-6)
    assert np.all(np.diff(lengths) >= -1e-12)


def test_regression_detection_table_interpolation_worst_case():
    """Found while writing ``tests/test_detection.py``: a 1025-point linear table
    out to five beam half-widths is NOT accurate to 1e-5, as first claimed.
    The measured worst case is 1.65e-5 absolute, on the steep shoulder of the
    beam. The docstring and the tolerance were corrected to the measured
    value rather than the value that would have been convenient.
    """
    detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, 0.2)
    offsets, values = detector.detection_probability_table(1.4e-3, 5.0e-5, 2.5e-4, 1025)
    probe = np.linspace(0.0, 2.5e-4, 97)
    error = np.max(
        np.abs(np.interp(probe, offsets, values) - detector.detection_probability(
            1.4e-3, probe, 5.0e-5
        ))
    )
    assert 1.0e-5 < error < 2.0e-5


# ----------------------------------------------------------------------------
# 2. Strict-threshold booleans, compared the only way they can honestly be.
# ----------------------------------------------------------------------------


def test_detection_boolean_is_only_asserted_away_from_the_threshold():
    """``P_d > threshold`` is a step function of a rounded quantity.

    This compares the detection probability computed with 24 and with 48
    Gauss-Hermite nodes. The two statistics agree to within 2e-9 (measured
    worst case 1.418e-9, at zero offset where the irradiance average is
    hardest). The derived boolean ``P_d > 0.5`` is asserted to agree ONLY
    where the statistic sits further than that tolerance from 0.5; where it
    sits inside the tolerance, the boolean is allowed to differ, because
    floating point cannot promise otherwise.
    """
    tolerance = 2e-9
    coarse = PoissonDwellDetector(2.0e4, 2.0e3, 8, 0.2, quadrature_nodes=24)
    fine = PoissonDwellDetector(2.0e4, 2.0e3, 8, 0.2, quadrature_nodes=48)
    offsets = np.linspace(0.0, 1.2e-4, 400)
    a = np.asarray(coarse.detection_probability(1.4e-3, offsets, 5.0e-5))
    b = np.asarray(fine.detection_probability(1.4e-3, offsets, 5.0e-5))
    assert np.max(np.abs(a - b)) < tolerance
    decided = np.abs(a - 0.5) > tolerance
    assert np.array_equal((a > 0.5)[decided], (b > 0.5)[decided])


def test_acquisition_within_deadline_is_compared_as_a_rate_not_as_a_boolean():
    """A single episode's "acquired before the deadline" flag flips on a
    comparison of two floats, so it is never asserted. What is asserted is
    that two Monte Carlo runs of the same model with different seeds agree on
    the RATE to within their combined standard error.
    """
    uncertainty = PointingUncertainty(150e-6)
    from acqsearch import (
        TerminalGeometry,
        cell_detection_probabilities,
        cell_visit_order,
        standard_paths,
    )

    geometry = TerminalGeometry(100e-6, 200e-6, 10e-3, 2e-3)
    detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, 0.2)
    grid = build_cell_grid(uncertainty, geometry, 2.5)
    q = cell_detection_probabilities(grid, geometry, detector, 1.4e-3, uncertainty)
    order = cell_visit_order(standard_paths(grid)["spiral"], grid)
    schedule = build_schedule(grid, order, geometry, 1.4e-3, 0.0, 0.0)
    deadline = 2.0 * schedule.period_s
    first = monte_carlo_acquisition(grid, schedule, q, 30000, seed=1, deadline_s=deadline)
    second = monte_carlo_acquisition(grid, schedule, q, 30000, seed=2, deadline_s=deadline)
    rate_se = np.sqrt(2.0 * 0.25 / 30000.0)
    assert abs(first.probability_by_deadline - second.probability_by_deadline) < 4.0 * rate_se


def test_reference_configuration_numbers_are_pinned():
    """The configuration every README number is quoted at.

    These are structural quantities, not statistics, so they are pinned
    tightly: the cell count is an integer and the prior mass is a sum of exact
    error-function rectangles.
    """
    uncertainty = PointingUncertainty(150e-6)
    from acqsearch import TerminalGeometry

    geometry = TerminalGeometry(100e-6, 200e-6, 10e-3, 2e-3)
    grid = build_cell_grid(uncertainty, geometry, 2.5)
    assert grid.n_cells == 89
    assert grid.prior_mass.sum() == pytest.approx(0.9544979085580522, rel=1e-12)
    assert grid.side_rad == pytest.approx(7.0710678118654755e-05, rel=1e-15)


def test_expected_time_at_the_reference_configuration_is_pinned():
    """The analytic expected acquisition time quoted in the README.

    Pinned with a relative tolerance of 1e-9, not bit-exactly: it is the end
    of a floating-point pipeline through Gauss-Legendre quadrature,
    Gauss-Hermite quadrature and a Poisson tail sum.
    """
    from acqsearch import (
        TerminalGeometry,
        cell_detection_probabilities,
        cell_visit_order,
        standard_paths,
    )

    uncertainty = PointingUncertainty(150e-6)
    geometry = TerminalGeometry(100e-6, 200e-6, 10e-3, 2e-3)
    detector = PoissonDwellDetector(2.0e4, 2.0e3, 8, 0.2)
    grid = build_cell_grid(uncertainty, geometry, 2.5)
    q = cell_detection_probabilities(grid, geometry, detector, 1.4e-3, uncertainty)
    f = detector.false_alarm_probability(1.4e-3)
    order = cell_visit_order(standard_paths(grid)["spiral"], grid)
    schedule = build_schedule(grid, order, geometry, 1.4e-3, f, 0.5)
    estimate = expected_acquisition_time(grid, schedule, q)
    assert estimate.expected_time_s == pytest.approx(0.5481830843, rel=1e-6)
    assert f == pytest.approx(0.008130741975, rel=1e-6)
