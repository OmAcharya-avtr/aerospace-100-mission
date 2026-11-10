"""Counterexamples found during the build, pinned with their literal inputs.

``.hypothesis/`` is gitignored and the Hypothesis database is disabled in
``tests/conftest.py``, so a falsifying example found by one run is LOST on a
fresh clone. Every counterexample this build found is therefore written out
here as an explicit test with its literal inputs, and every defect fixed during
the build is pinned so that a regression cannot pass quietly.

Each test below names what was wrong, what the measured value was, and what was
changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sitediverse.bounds import (
    MACHINE_EPSILON,
    SLACK_FACTOR,
    bound_slack,
    frechet_hoeffding,
)
from sitediverse.climatology import NHOUR, NSEASON, cell_index
from sitediverse.copula import bivariate_density, correlation_excess, epoch_nodes
from sitediverse.field import EPOCHS_PER_YEAR
from sitediverse.geometry import (
    EARTH_RADIUS_KM,
    along_cross_km,
    along_cross_matrices,
    bearing_matrix,
    great_circle_km,
    initial_bearing_deg,
    midpoint_bearing_deg,
    separation_matrix,
)


def test_regression_bearing_at_a_tiny_negative_argument_is_not_three_sixty():
    """Found by Hypothesis at ``(0.0, 0.0) -> (-1.0, -1e-17)``.

    ``x % 360.0`` does not guarantee the half-open range ``[0, 360)``: for a
    tiny negative ``x`` the IEEE remainder rounds UP to exactly 360.0.
    Measured: ``(-1e-20) % 360.0 == 360.0`` exactly. The implementation now
    folds the result a second time.
    """
    assert (-1e-20) % 360.0 == 360.0
    got = float(initial_bearing_deg(0.0, 0.0, -1.0, -1e-17))
    assert 0.0 <= got < 360.0


def test_regression_the_longitude_negative_zero_case():
    """A second instance of the same class, at a longitude of ``-0.0``."""
    got = float(initial_bearing_deg(1.0, 0.0, 0.0, -0.0))
    assert 0.0 <= got < 360.0


def test_regression_the_along_cross_decomposition_was_asymmetric():
    """Found by a property test on the declared network. A real defect.

    Built on the INITIAL bearing, the along and cross matrices were asymmetric
    by up to 1478.5 km, so a pair's decomposition depended on which station
    had the lower index -- which is not a property of the pair. The fix was to
    build equation (3) on the azimuth at the arc MIDPOINT, which is symmetric
    to within 180 degrees and therefore invariant under the absolute values of
    equation (3).

    The test pins BOTH facts: that the old route is still asymmetric by
    hundreds of kilometres, and that the current one is symmetric to rounding.
    """
    separation = separation_matrix()
    old_along, old_cross = along_cross_km(separation, bearing_matrix())
    assert float(np.max(np.abs(old_along - old_along.T))) > 1000.0
    new_along, new_cross = along_cross_matrices()
    assert float(np.max(np.abs(new_along - new_along.T))) < 1e-8
    assert float(np.max(np.abs(new_cross - new_cross.T))) < 1e-8


def test_regression_the_haversine_antipodal_relative_error():
    """A tolerance that was wrong, not a defect. Pinned as a measurement.

    An initial test asserted the antipodal separation to a relative 1e-12. The
    haversine form cannot deliver that: ``arcsin(sqrt(a))`` has a diverging
    derivative as ``a -> 1``, so the relative error is of order ``sqrt(eps)``.
    Measured at ``(10, 20)`` and its antipode: a relative error of 9.486e-09,
    which is 0.6366 times ``sqrt(eps)`` -- the same order, as the analysis says,
    and a factor that depends on how close to exactly antipodal the pair is.
    """
    expected = np.pi * EARTH_RADIUS_KM
    got = float(great_circle_km(10.0, 20.0, -10.0, -160.0))
    relative = abs(got - expected) / expected
    assert relative == pytest.approx(9.486e-09, rel=0.0, abs=1e-11)
    assert relative / np.sqrt(MACHINE_EPSILON) == pytest.approx(
        0.6366, rel=0.0, abs=0.01
    )


def test_regression_an_antipodal_midpoint_azimuth_is_nan_and_not_a_number():
    """The great circle through antipodes is not unique."""
    assert np.isnan(float(midpoint_bearing_deg(10.0, 20.0, -10.0, -160.0)))


def test_regression_the_bound_slack_was_an_absolute_constant_in_disguise():
    """A real defect in the check that exists to catch this defect class.

    The first version of equation (19) returned
    ``SLACK_FACTOR * eps * kappa * |bound|`` with ``kappa = sum|q| / |bound|``,
    which cancels to ``SLACK_FACTOR * eps * sum|q|``: it looked conditioned and
    was independent of the bound value altogether. Worse, it would have made
    the slack VANISH with the bound in any implementation that did the
    multiplication in the other order.

    Measured at the two cases that caught it: a well-conditioned bound of 0.9
    on marginals ``(0.95, 0.95)`` got a slack of 2.700e-14, and an
    ill-conditioned bound of 1e-7 on marginals ``(0.5, 0.5000001)`` got
    1.421e-14 -- the ill-conditioned case got the SMALLER slack. The current
    form returns the absolute rounding scale of the summands and exposes the
    conditioning separately, and the relative slack diverges as it should.
    """
    well = bound_slack(np.array([0.95, 0.95]), 0.9)
    ill = bound_slack(np.array([0.5, 0.5000001]), 1e-7)
    assert well == pytest.approx(
        SLACK_FACTOR * MACHINE_EPSILON * (1.9 + 0.9), rel=1e-12, abs=0.0
    )
    assert ill == pytest.approx(
        SLACK_FACTOR * MACHINE_EPSILON * (1.0000001 + 1e-7), rel=1e-12, abs=0.0
    )
    # The relative slack, which is the conditioned quantity, is far larger for
    # the ill-conditioned bound -- which is the property the first form lacked.
    assert (ill / 1e-7) > 1e6 * (well / 0.9)


def test_regression_an_ill_conditioned_lower_bound_is_flagged_uninformative():
    """The consequence of the fix above, pinned.

    At a ``delta`` of 1e-17 the lower bound of equation (17) falls below its
    own rounding scale and the check on that side admits anything.
    """
    bounds = frechet_hoeffding(np.array([0.5, 0.5 + 1e-17]))
    assert not bounds.lower_informative


def test_regression_the_epoch_nodes_used_cell_centres_and_lost_three_decades():
    """A real accuracy defect, found by a tolerance that would not hold.

    ``EpochNodes`` originally used the CELL CENTRE of each histogram cell as
    its node. Measured on the declared records at 48 bins, the worst error
    against the exact epoch-by-epoch mean was 4.4e-04, and it was NOT monotone
    in the bin count. Using the cell MEAN makes the first moment of each cell
    exact and the worst error falls to 6.0e-07.

    The test rebuilds the old form and pins both numbers, so a change back
    cannot pass.
    """
    from sitediverse.climatology import estimate_climatology
    from sitediverse.copula import bivariate_orthant, pair_joint_exact
    from sitediverse.field import RECORD_SPECS, generate_record

    record = generate_record(RECORD_SPECS["short"])
    threshold = estimate_climatology(record.cloud).threshold_series()
    left, right = threshold[:, 0], threshold[:, 1]
    rho = 0.7

    counts, edges_left, edges_right = np.histogram2d(left, right, bins=48)
    centre_left = 0.5 * (edges_left[:-1] + edges_left[1:])
    centre_right = 0.5 * (edges_right[:-1] + edges_right[1:])
    occupied = counts > 0
    grid_left, grid_right = np.meshgrid(centre_left, centre_right, indexing="ij")
    weight = counts[occupied] / float(left.size)
    old = float(
        np.sum(
            weight
            * bivariate_orthant(
                grid_left[occupied],
                grid_right[occupied],
                np.full(int(occupied.sum()), rho),
            )
        )
    )
    exact = pair_joint_exact(left, right, rho)
    new = epoch_nodes(left, right).joint(rho)
    assert abs(old - exact) > 1e-5
    assert abs(new - exact) < 1e-6


def test_regression_the_first_order_copula_expansion_remainder_is_second_order():
    """Found by Hypothesis at ``h = k = 1.0``. A wrong tolerance, not a defect.

    An initial test asserted ``excess = rho phi_2(h, k; 0)`` to a relative
    1e-8 at ``rho = 1e-5``. The remainder of that expansion is SECOND order in
    ``rho``, not third: the second derivative of ``Phi_2`` in ``rho`` does not
    vanish for ``h k != 0``. Measured at ``h = k = 1.0``, ``rho = 1e-5``: a
    relative discrepancy of 5.0e-06, which is 0.5 ``rho`` as it should be.
    """
    h = k = 1.0
    rho = 1e-5
    first_order = rho * float(bivariate_density(h, k, 0.0))
    got = float(correlation_excess(h, k, rho))
    relative = abs(got - first_order) / first_order
    assert relative == pytest.approx(5.0e-06, rel=0.0, abs=5e-7)
    # And at 1e-9 the same remainder is four decades smaller.
    smaller = 1e-9
    first_order = smaller * float(bivariate_density(h, k, 0.0))
    got = float(correlation_excess(h, k, smaller))
    assert abs(got - first_order) / first_order < 1e-8


def test_regression_the_season_by_hour_cells_are_not_equal_in_size():
    """A wrong claim in a test, corrected and pinned.

    An initial test asserted that all 96 season-by-hour cells hold the same
    number of epochs in one declared year. They do not: 2920 does not divide by
    96, so cells hold 30 or 31 epochs. The estimator divides by each cell's own
    occupancy, so the imbalance is carried correctly, but the claim was false
    and is now pinned as a measurement.
    """
    counts = np.bincount(cell_index(EPOCHS_PER_YEAR), minlength=NSEASON * NHOUR)
    assert int(counts.min()) == 30
    assert int(counts.max()) == 31
    assert int(counts.sum()) == EPOCHS_PER_YEAR


def test_regression_the_share_of_a_pure_sampling_discrepancy_is_not_one_half():
    """A wrong claim in a test, corrected and pinned.

    An initial test asserted that equation (28)'s uncorrected share averages
    0.5 when the discrepancy is pure sampling noise, reasoning from
    ``E[g^2] = s^2``. The expectation of a ratio is not the ratio of
    expectations: the correct value is ``E[1 / (1 + z^2)]`` with ``z``
    standard normal, which is 0.655680 by quadrature. The difference is 0.156
    and would have been read as a bias that is not there.
    """
    from scipy.integrate import quad
    from scipy.stats import norm

    value, _ = quad(lambda z: norm.pdf(z) / (1.0 + z * z), -40.0, 40.0)
    assert value == pytest.approx(0.6556795424187994, rel=1e-9, abs=0.0)
    assert abs(value - 0.5) > 0.15


def test_regression_the_margin_bootstrap_resamples_within_the_rows_window():
    """A real memory defect. 7.8 GiB was not enough for the first form.

    The first version of the equation (41) bootstrap indexed the WHOLE record
    with a ``(replicates, epochs)`` draw array for every row at once: 400
    replicates of 23360 epochs against 420 rows is a boolean array of 3.9
    gigabytes, and the build process was killed by the kernel. The current form
    resamples blocks WITHIN each row's own regime and time half, shares the
    draws across the rows of a window, and processes the replicates in chunks.

    The test pins the memory arithmetic that made the old form impossible, so
    that the comment in ``study._margins`` has a number behind it.
    """
    old_bytes = 400 * 23360 * 420
    assert old_bytes > 3.9e9
    new_bytes = 400 * (23360 // 8) * 85
    assert new_bytes < 1.1e8


def test_regression_a_substring_feature_check_flagged_along_km_as_a_longitude():
    """A wrong claim in a test, corrected and pinned.

    An initial test asserted that no feature name contains the substring
    ``lon``. ``along_km`` does: ``a-l-o-n-g``. The check is now on whole
    underscore-separated tokens, which is what it meant.
    """
    assert "lon" in "along_km"
    assert "lon" not in set("along_km".split("_"))
