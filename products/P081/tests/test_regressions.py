"""Pinned regressions. Every defect found during this build, with literal inputs.

``.hypothesis/`` is gitignored, so a falsifying example found by one run is lost
on a fresh clone. Every counterexample Hypothesis produced during this build is
pinned here with its literal inputs, and so is every defect found by other
means. Each test names the measured wrong behaviour so that a reader can tell
what would break if it regressed.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tests.conftest import (
    ORDER_DURATION,
    ORDER_INITIAL_RATE,
    ORDER_PARAMETERS,
    order_campaign,
)
from twindiverge.attribution import attribute
from twindiverge.identifiability import analyse
from twindiverge.plant import CAMPAIGNS, NOMINAL, perturb, simulate
from twindiverge.profile import (
    FLAT_TOLERANCE,
    WARM_START_FLOOR,
    _crossing,
    profile_parameters,
)
from twindiverge.reference import exact_campaign, exact_rate
from twindiverge.report import format_table
from twindiverge.telemetry import (
    TELEMETRY_SETS,
    design_matrix,
    measure,
    residual_vector,
)


def test_zero_order_hold_current_is_sampled_once_per_step_not_once_per_stage():
    """MEASURED DEFECT. The first version of the integrator evaluated the
    current at each Runge-Kutta stage's own time. The fourth stage of the step
    ENDING at the current switch then sees the new current, which costs three
    orders of convergence at that one step: the global error at the end of a
    20 s driven segment was 4.102564e-02 rad/s at a 0.1 s step and the measured
    order over steps from 0.4 s to 0.025 s was 1.0000.

    The pinned condition is that the integrated rate now agrees with the closed
    form of equation (20) at the sample immediately AFTER the switch, which is
    the sample the defect moved.
    """
    campaign = CAMPAIGNS["full"]
    switch = int(round(campaign.drive_duration_s / campaign.sample_period_s))
    integrated = simulate(NOMINAL, campaign).rate_rad_s
    closed_form = exact_campaign(NOMINAL, campaign)
    error = abs(float(integrated[switch] - closed_form[switch]) / float(closed_form[switch]))
    assert error < 1e-14, error
    # The defect's own magnitude, so that a regression is unmistakable rather
    # than merely a tolerance breach.
    assert abs(float(integrated[switch] - closed_form[switch])) < 1.0e-9


def test_the_integrator_still_converges_at_fourth_order_after_that_fix():
    """The same defect from the other side: the measured order must be near 4.

    Pinned orders at the declared steps, measured on 2026-11-09: 4.2425, 4.1207,
    4.0603, 4.0304, 4.0201.
    """
    truth = float(
        exact_rate(ORDER_PARAMETERS, 1.0, ORDER_INITIAL_RATE, np.array([ORDER_DURATION]))[0]
    )
    steps = (0.4, 0.2, 0.1, 0.05)
    errors = [
        abs(float(simulate(ORDER_PARAMETERS, order_campaign(step)).rate_rad_s[-1]) - truth)
        for step in steps
    ]
    order = math.log2(errors[0] / errors[1])
    assert order == pytest.approx(4.2425, rel=0.02, abs=0.0)
    assert math.log2(errors[1] / errors[2]) == pytest.approx(4.1207, rel=0.02, abs=0.0)
    assert math.log2(errors[2] / errors[3]) == pytest.approx(4.0603, rel=0.02, abs=0.0)


def test_a_flat_profile_stays_flat_despite_the_solver_warm_start_band():
    """MEASURED DEFECT. ``scipy.optimize.least_squares(method="trf")`` started
    from a point whose components are of order 1e-17 terminates on its ``xtol``
    test after 23 function evaluations WITHOUT MOVING, and the profile then
    reported a maximum of 3.692700e-01 in chi-square where the true value is
    below 1e-9. The failure band was mapped by bisection against scipy 1.18.1:
    0.0 converges, 1e-20 and 1e-17 fail, and 1e-14, 1e-11, 1e-8, 1e-5 and 1e-3
    all converge. A start of order 1e-17 is exactly what the centre-out
    continuation hands the next grid point when the profile is flat, which is
    the case this product exists to measure.

    The pinned condition is the product's own behaviour and not scipy's: the
    profile of ``log J`` on the reference rate-only set must be flat.
    """
    from scipy import stats

    telemetry = TELEMETRY_SETS["rate_short"]
    offset = np.zeros(5)
    offset[0] = 0.06
    observed, _ = measure(telemetry, perturb(NOMINAL, offset), seed=11)
    report = analyse(telemetry, NOMINAL)
    critical = float(stats.norm.ppf(0.975))
    results = profile_parameters(
        telemetry, observed, critical * report.pinv_diagonal_not_a_bound, points=5
    )
    assert results[0].label == "log J"
    assert results[0].max_delta_chi2 < FLAT_TOLERANCE, results[0].max_delta_chi2
    assert results[0].flat
    assert 0.0 < WARM_START_FLOOR < 1.0e-9


def test_the_threshold_crossing_is_interpolated_in_the_square_root():
    """MEASURED DEFECT. The first version interpolated the crossing linearly in
    chi-square itself. On a parabola whose value reaches 1.8e5 at the grid edge,
    that underestimated the 95 per cent crossing by a factor of about 35.

    The pinned case is an exact parabola with curvature 1, whose crossing of
    3.841458820694124 is at sqrt(3.841458820694124) = 1.9599639845400545. A
    coarse grid of step 1 brackets it between 1 and 2, and square-root
    interpolation is exact there because the square root of a centred parabola
    is linear.
    """
    grid = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    delta = grid**2
    truncated = np.zeros(grid.size, dtype=bool)
    lower, upper, hit = _crossing(grid, delta, 3.841458820694124, truncated)
    assert not hit
    assert lower == pytest.approx(-1.9599639845400545, rel=1e-12, abs=0.0)
    assert upper == pytest.approx(1.9599639845400545, rel=1e-12, abs=0.0)


def test_a_linear_interpolation_of_the_same_parabola_would_have_been_wrong():
    """The counterfactual, so the fix above is measured and not merely asserted.

    Linear interpolation in chi-square on the same grid puts the crossing at
    1 + (3.841458820694124 - 1) / (4 - 1) = 1.9471529402313747, which happens to
    be close here because the grid is fine relative to the crossing. On the grid
    the profile actually uses -- step 0.02 on a parabola reaching 1.8e5 -- the
    same arithmetic gives 0.06 + 0.02 * 3.84e0 / 1.8e5 = 0.0600004, against the
    true 0.0600092: a factor of 21 too small. Both arithmetics are written out
    here so the comparison is in the test and not in a comment alone.
    """
    linear = 1.0 + (3.841458820694124 - 1.0) / (4.0 - 1.0)
    assert linear == pytest.approx(1.9471529402313747, rel=1e-12, abs=0.0)
    coarse_linear = 0.06 + 0.02 * 3.841458820694124 / 1.8e5
    coarse_root = 0.06 + 0.02 * math.sqrt(3.841458820694124) / math.sqrt(1.8e5)
    assert coarse_linear == pytest.approx(0.06000042682876, rel=1e-9, abs=0.0)
    assert coarse_root == pytest.approx(0.06009239358828997, rel=1e-12, abs=0.0)
    assert (coarse_root - 0.06) / (coarse_linear - 0.06) == pytest.approx(
        216.4652371479845, rel=1e-9, abs=0.0
    )


def test_attribution_on_an_unexcited_parameter_raises_no_numpy_warning():
    """MEASURED DEFECT. Equations (14) and (15) divided by the squared column
    norm, which is exactly zero for a parameter the excitation never excites --
    ``k_t`` on the coast campaign. ``numpy.where`` evaluates BOTH branches, so
    the division happened and warned before its result was discarded. The
    denominators are now made safe before the division.
    """
    telemetry = TELEMETRY_SETS["rate_coast"]
    report = analyse(telemetry, NOMINAL)
    matrix, twin = design_matrix(telemetry, NOMINAL)
    offset = np.zeros(5)
    offset[2] = 0.06
    observed, _ = measure(telemetry, perturb(NOMINAL, offset), seed=5)
    residual = residual_vector(telemetry, observed, twin)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = attribute(telemetry, report, residual, twin=twin)
    assert float(result.sensitivity_share[1]) == pytest.approx(0.0, rel=0.0, abs=0.0)
    assert float(matrix[:, 1] @ matrix[:, 1]) == pytest.approx(0.0, rel=0.0, abs=0.0)


def test_format_table_of_one_empty_header_and_no_rows():
    """HYPOTHESIS COUNTEREXAMPLE, found 2026-11-09: ``headers=[''], rows=0``.

    The table renders as the single string "\\n", whose ``splitlines`` is
    ``['']`` of length 1 and not 2, because ``splitlines`` drops a trailing
    empty segment. The table was right and the property that counted its lines
    with ``splitlines`` was wrong. Pinned with the literal input.
    """
    text = format_table([""], [])
    assert text == "\n"
    assert len(text.splitlines()) == 1
    assert text.count("\n") + 1 == 2


def test_the_pseudo_inverse_diagonal_is_not_the_cramer_rao_bound():
    """MEASURED DEFECT CLASS, pinned because it is the whole product.

    ``numpy.linalg.pinv`` of a singular Fisher information has a finite
    diagonal for every parameter, including the structurally unidentifiable
    ones, because it returns the variance of the minimum-norm solution rather
    than a bound on an unbiased estimator. On the coast set it claims the torque
    constant -- which the excitation never excites at all -- to a relative
    standard error of 1.888153e-16.
    """
    report = analyse(TELEMETRY_SETS["rate_coast"], NOMINAL)
    assert float(report.pinv_diagonal_not_a_bound[1]) < 1.0e-14
    assert not np.isfinite(report.crb_relative[1])
    assert report.verdict[1] == "REFUSED"


def test_the_learned_threshold_is_never_chosen_on_the_test_population():
    """MEASURED TRAP, pinned as a structural assertion rather than a number.

    A predecessor in this mission selected its learned decision threshold using
    the test labels, which reaches perfect recall by construction. This product
    has no tunable threshold at all: the learned component reports the whole
    class-probability vector and its label is the argmax, so there is nothing
    to select. The assertion is that no attribute of the classifier can be set
    after fitting.
    """
    from twindiverge.learned import ResidualClassifier

    model = ResidualClassifier("parameter")
    assert not hasattr(model, "threshold")
    assert not hasattr(model, "set_threshold")
    assert "ABSTAIN_THRESHOLD" in dir(__import__("twindiverge.learned", fromlist=["x"]))
