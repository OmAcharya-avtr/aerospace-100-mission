"""CRPS, PIT calibration and coverage, against closed forms where they exist."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passyield.scoring import (
    crps_ensemble,
    crps_mean,
    interval_coverage,
    mean_absolute_error,
    pit_calibration_error,
    pit_value,
    pit_values,
    root_mean_square_error,
)


def test_crps_of_a_point_forecast_is_the_absolute_error():
    """A one-member ensemble: the pairwise term vanishes and CRPS is ``|x-y|``.

    Hand-calculated: ``x = 3``, ``y = 1`` gives ``CRPS = 2``. Gneiting &
    Raftery (2007) section 4.2 make this reduction explicit, and it is why
    baseline 1 can be scored on the same axis as a distribution.
    """
    assert crps_ensemble(np.asarray([3.0]), 1.0) == pytest.approx(2.0, rel=0.0, abs=0.0)


def test_crps_of_two_members_hand_calculated():
    """Hand-calculated: members ``[0, 10]``, observation 4.

    First term: ``(|0-4| + |10-4|)/2 = (4 + 6)/2 = 5``.
    Pairwise term: ``sum_i sum_j |x_i - x_j| = 0 + 10 + 10 + 0 = 20``, divided
    by ``2 n**2 = 8``, which is 2.5.
    So ``CRPS = 5 - 2.5 = 2.5``.
    """
    assert crps_ensemble(np.asarray([0.0, 10.0]), 4.0) == pytest.approx(
        2.5, rel=0.0, abs=1.0e-12
    )


def test_crps_of_a_standard_normal_at_its_mean_matches_the_closed_form():
    """Known answer: the closed-form CRPS of ``N(0,1)`` evaluated at ``y = 0``.

    ``CRPS(N(mu, sigma), y) = sigma [ z(2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]``
    with ``z = (y - mu)/sigma`` (Gneiting & Raftery 2007, equation 21 and the
    Gaussian case). At ``z = 0`` the first term is 0 and
    ``2 phi(0) = 2/sqrt(2 pi) = 0.7978845608``, so
    ``CRPS = 0.7978845608 - 0.5641895835 = 0.2336949773``.

    The ensemble estimator is compared to that at four standard errors of a
    200000-member draw, not bit-exactly.
    """
    closed = 2.0 / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert closed == pytest.approx(0.2336949773, rel=0.0, abs=1.0e-9)
    rng = np.random.default_rng(99099)
    members = rng.normal(0.0, 1.0, 200000)
    estimate = crps_ensemble(members, 0.0)
    # Monte Carlo standard error of the first term is sd(|x|)/sqrt(n); the
    # pairwise term is a U-statistic with a smaller error. Four of those is a
    # declared band, not a tuned one.
    band = 4.0 * float(np.std(np.abs(members)) / math.sqrt(members.size))
    assert estimate == pytest.approx(closed, rel=0.0, abs=band)


def test_crps_pairwise_term_matches_the_naive_double_sum():
    """The O(n log n) sorted form against the O(n^2) double sum.

    Two implementations of the same quantity, compared to tolerance. The
    sorted identity is ``sum_i sum_j |x_i - x_j| = 2 sum_k (2k - n + 1) x_(k)``.
    """
    rng = np.random.default_rng(7)
    members = rng.normal(5.0, 2.0, 60)
    naive = float(np.abs(members[:, None] - members[None, :]).sum())
    ordered = np.sort(members)
    weights = 2.0 * np.arange(members.size) - members.size + 1.0
    fast = 2.0 * float(np.dot(weights, ordered))
    assert fast == pytest.approx(naive, rel=1.0e-12, abs=0.0)


@given(shift=st.floats(min_value=-50.0, max_value=50.0))
@settings(max_examples=120, deadline=None)
def test_crps_is_translation_equivariant(shift):
    """Property: shifting the forecast and the observation together is free.

    CRPS is a distance in the observation's own units, so it is invariant
    under a common translation. Checked to tolerance, because the shift
    perturbs the floating-point partial sums.
    """
    members = np.asarray([1.0, 2.0, 5.0, 11.0])
    base = crps_ensemble(members, 3.0)
    moved = crps_ensemble(members + shift, 3.0 + shift)
    assert moved == pytest.approx(base, rel=1.0e-9, abs=1.0e-9)


def test_crps_is_zero_for_a_perfect_deterministic_forecast():
    """A point forecast that hits the observation scores zero."""
    assert crps_ensemble(np.asarray([7.5]), 7.5) == 0.0


def test_crps_rejects_an_empty_ensemble():
    """Input validation."""
    with pytest.raises(ValueError, match="at least one member"):
        crps_ensemble(np.asarray([]), 1.0)


def test_crps_rejects_non_finite_inputs():
    """Input validation."""
    with pytest.raises(ValueError, match="finite"):
        crps_ensemble(np.asarray([1.0, np.inf]), 1.0)
    with pytest.raises(ValueError, match="observation_mbit"):
        crps_ensemble(np.asarray([1.0]), float("nan"))


def test_crps_mean_rejects_misaligned_inputs():
    """Input validation."""
    with pytest.raises(ValueError, match="observations"):
        crps_mean(np.zeros((3, 4)), np.zeros(2))


def test_pit_value_uses_the_mid_rank_convention():
    """Hand-calculated: members ``[1, 2, 2, 3]``, observation 2.

    One member is strictly below and two are equal, so the PIT is
    ``(1 + 0.5*2)/4 = 0.5``. Without the mid-rank convention a forecast with
    an atom -- and baseline 2 has one, at zero loss -- produces a PIT pinned
    at 0 or 1 and the calibration error becomes an artefact of the atom.
    """
    assert pit_value(np.asarray([1.0, 2.0, 2.0, 3.0]), 2.0) == pytest.approx(
        0.5, rel=0.0, abs=1.0e-12
    )


def test_pit_value_at_the_extremes():
    """An observation below every member is 0; above every member is 1."""
    members = np.asarray([1.0, 2.0, 3.0])
    assert pit_value(members, 0.0) == 0.0
    assert pit_value(members, 4.0) == 1.0


def test_pit_calibration_error_hand_calculated():
    """Hand-calculated: PIT values ``[0.25, 0.5, 0.75]``.

    With ``n = 3`` the empirical CDF steps at each value. The supremum of
    ``|F_hat(u) - u|`` is attained at an order statistic and equals
    ``max_k max(k/n - u_(k), u_(k) - (k-1)/n)``:
      k=1: max(1/3 - 0.25, 0.25 - 0) = max(0.08333, 0.25) = 0.25
      k=2: max(2/3 - 0.5, 0.5 - 1/3) = max(0.16667, 0.16667) = 0.16667
      k=3: max(1 - 0.75, 0.75 - 2/3) = max(0.25, 0.08333) = 0.25
    So the statistic is 0.25.
    """
    assert pit_calibration_error(np.asarray([0.25, 0.5, 0.75])) == pytest.approx(
        0.25, rel=0.0, abs=1.0e-12
    )


def test_pit_calibration_error_of_a_calibrated_forecast_is_small():
    """A forecast drawn from the same law as the observation is calibrated.

    The statistic is compared against the asymptotic Kolmogorov 1 per cent
    critical value ``1.63/sqrt(n)``, which is a declared band from the
    distribution of the statistic and not a tuned constant.
    """
    rng = np.random.default_rng(99099)
    count = 2000
    observations = rng.normal(0.0, 1.0, count)
    ensembles = rng.normal(0.0, 1.0, (count, 200))
    statistic = pit_calibration_error(pit_values(ensembles, observations))
    assert statistic < 1.63 / math.sqrt(count)


def test_pit_calibration_error_detects_an_over_confident_forecast():
    """An ensemble five times too narrow is caught, and the control FIRES.

    Anchored on the TRUE law, not on an output: the observations come from
    ``N(0,1)`` and the forecast from ``N(0,0.2)``, so the forecast is wrong
    by construction and the PIT law is known in closed form. The PIT is
    ``Phi(y/0.2)`` with ``y ~ N(0,1)``, whose distribution function is
    ``Phi(0.2 * Phi^{-1}(u))``, and the supremum of its distance from the
    uniform is 0.3236 to four places -- computed from that expression, not
    measured here.

    The criterion is therefore an asymptotic constant, 0.25, chosen as a
    margin below 0.3236 and above the Kolmogorov 1 per cent critical value of
    ``1.63/sqrt(2000) = 0.0364``: the over-confident statistic converges to a
    non-zero constant while a calibrated one shrinks like ``1/sqrt(n)``, so
    the two are separated by the sample size and not by the seed. The
    criterion was NOT adjusted until the control passed -- an earlier draft
    asked for ten times the critical value, which is 0.3645 and sits ABOVE
    the asymptote, so no sample size could ever have satisfied it. The
    criterion was wrong; the seed was left alone.
    """
    rng = np.random.default_rng(99099)
    count = 2000
    observations = rng.normal(0.0, 1.0, count)
    ensembles = rng.normal(0.0, 0.2, (count, 200))
    statistic = pit_calibration_error(pit_values(ensembles, observations))
    assert statistic > 0.25, "the negative control did not fire"


def test_pit_calibration_error_rejects_values_outside_the_unit_interval():
    """Input validation."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pit_calibration_error(np.asarray([0.5, 1.5]))


def test_pit_calibration_error_rejects_an_empty_sample():
    """Input validation."""
    with pytest.raises(ValueError, match="at least one value"):
        pit_calibration_error(np.asarray([]))


def test_pit_value_rejects_an_empty_ensemble():
    """Input validation."""
    with pytest.raises(ValueError, match="at least one member"):
        pit_value(np.asarray([]), 1.0)


def test_pit_values_rejects_misaligned_inputs():
    """Input validation."""
    with pytest.raises(ValueError, match="observations"):
        pit_values(np.zeros((3, 4)), np.zeros(5))


def test_interval_coverage_hand_calculated():
    """Hand-calculated: three of four observations inside their intervals.

    Intervals ``[0,1]``, ``[0,1]``, ``[0,1]``, ``[0,1]`` and observations
    ``0.5, 0.2, 0.9, 1.5``: the first three are inside and the fourth is
    not, so the coverage is 0.75.
    """
    coverage = interval_coverage(
        np.zeros(4), np.ones(4), np.asarray([0.5, 0.2, 0.9, 1.5])
    )
    assert coverage == pytest.approx(0.75, rel=0.0, abs=1.0e-12)


def test_interval_coverage_is_inclusive_at_the_endpoints():
    """An observation exactly on a bound counts as covered."""
    assert interval_coverage(np.zeros(1), np.ones(1), np.ones(1)) == 1.0
    assert interval_coverage(np.zeros(1), np.ones(1), np.zeros(1)) == 1.0


def test_interval_coverage_rejects_a_reversed_interval():
    """Input validation."""
    with pytest.raises(ValueError, match="must not fall below"):
        interval_coverage(np.ones(2), np.zeros(2), np.zeros(2))


def test_interval_coverage_rejects_misaligned_inputs():
    """Input validation."""
    with pytest.raises(ValueError, match="must align"):
        interval_coverage(np.zeros(2), np.ones(3), np.zeros(2))


def test_error_measures_hand_calculated():
    """Hand-calculated: predictions ``[1,2,3]``, observations ``[1,4,7]``.

    Errors are 0, -2, -4. MAE is ``(0+2+4)/3 = 2``. RMSE is
    ``sqrt((0+4+16)/3) = sqrt(6.6667) = 2.581988897...``.
    """
    prediction = np.asarray([1.0, 2.0, 3.0])
    observation = np.asarray([1.0, 4.0, 7.0])
    assert mean_absolute_error(prediction, observation) == pytest.approx(
        2.0, rel=0.0, abs=1.0e-12
    )
    assert root_mean_square_error(prediction, observation) == pytest.approx(
        math.sqrt(20.0 / 3.0), rel=1.0e-12, abs=0.0
    )


@pytest.mark.parametrize("function", [mean_absolute_error, root_mean_square_error])
def test_error_measures_reject_misaligned_inputs(function):
    """Input validation."""
    with pytest.raises(ValueError, match="must align"):
        function(np.zeros(2), np.zeros(3))
