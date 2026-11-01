"""Known-answer, unit, input-validation and edge-case tests for baseline 1.

Every known-answer test carries its hand calculation in a comment. Four of
them are against PUBLISHED values rather than against this implementation:
the classical one-sided normal tolerance factors, which are tabulated in
Krishnamoorthy, K. and Mathew, T. (2009), *Statistical Tolerance Regions*,
Wiley, chapter 1, and in every standard quality-engineering handbook.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from transfergap.bounds import (
    empirical_bernstein_mean_upper,
    empirical_bernstein_slack,
    hoeffding_mean_upper,
    normal_tolerance_required_n,
    normal_tolerance_upper,
    required_n_empirical_bernstein,
    required_n_hoeffding,
    required_n_wilks,
    tolerance_factor_normal,
    wilks_tolerance_upper,
)


def test_hoeffding_known_answer(simple_gaps):
    """Hand calculation of equation (3) on [1, 2, 3, 4] with R = 10, 1-delta = 0.95.

    delta           = 0.05
    ln(1/delta)     = ln(20)            = 2.995732273553991
    ln(20)/(2 n)    = 2.995732273553991 / 8 = 0.37446653419424886
    sqrt of that                        = 0.6119367076702041
    slack = R * that = 10 * 0.6119367076702041 = 6.119367076702041
    dbar  = (1+2+3+4)/4                 = 2.5
    upper = 2.5 + 6.119367076702041     = 8.619367076702041
    """
    bound = hoeffding_mean_upper(simple_gaps, 10.0, 0.95, "kN.m")
    assert bound.estimate == pytest.approx(2.5, rel=0.0, abs=1e-15)
    assert bound.slack == pytest.approx(6.119367076702041, rel=1e-13, abs=0.0)
    assert bound.upper == pytest.approx(8.619367076702041, rel=1e-13, abs=0.0)
    assert bound.target == "mean"
    assert bound.distribution_free is True
    assert bound.n == 4
    assert bound.unit == "kN.m"


def test_hoeffding_slack_does_not_look_at_the_data():
    """The slack of equation (3) is a function of R, delta and n only."""
    first = hoeffding_mean_upper(np.zeros(20), 10.0, 0.95)
    second = hoeffding_mean_upper(np.full(20, 10.0), 10.0, 0.95)
    assert first.slack == pytest.approx(second.slack, rel=1e-15, abs=0.0)


def test_hoeffding_works_at_n_equals_one():
    """Edge case: equation (3) has no n-1 anywhere and is defined at n = 1.

    slack = 10 * sqrt(ln(20) / 2) = 10 * sqrt(1.4978661367769954)
          = 10 * 1.2238734153404082 = 12.238734153404082
    """
    bound = hoeffding_mean_upper(np.array([3.0]), 10.0, 0.95)
    assert bound.slack == pytest.approx(12.238734153404082, rel=1e-13, abs=0.0)


def test_empirical_bernstein_known_answer():
    """Hand calculation of equation (5) on [0, 0, 10, 10], R = 10, 1-delta = 0.95.

    ln(2/delta) = ln(40)              = 3.6888794541139363
    dbar                               = 5.0
    V_n (ddof 1) = (4*25 + ... )       = 100/3 = 33.333333333333336
    2 V_n ln(40) / n = 2*33.3333...*3.6888794541139363/4 = 61.481324235232274
    sqrt of that                       = 7.840998930920745
    7 R ln(40) / (3 (n-1)) = 7*10*3.6888794541139363/9 = 28.6913069208
    slack = 7.840998930920745 + 28.69128464310839 = 36.53228740010524
    upper = 5.0 + 36.53228740010524 = 41.53228740010524
    """
    bound = empirical_bernstein_mean_upper(np.array([0.0, 0.0, 10.0, 10.0]), 10.0, 0.95)
    assert bound.slack == pytest.approx(36.53228740010524, rel=1e-13, abs=0.0)
    assert bound.upper == pytest.approx(41.53228740010524, rel=1e-13, abs=0.0)


def test_empirical_bernstein_may_exceed_the_declared_range():
    """A valid bound can sit above R. It is conservative, not inconsistent.

    The sample mean of a quantity in [0, R] plus a slack of order R exceeds R
    whenever n is small. The bound still HOLDS -- the true mean is below R and
    therefore below the bound -- it is simply vacuous, and a vacuous bound is
    the honest answer at that sample size. Nothing in this package clips it,
    because clipping would silently convert a vacuous bound into a bound that
    looks informative.
    """
    bound = empirical_bernstein_mean_upper(np.array([0.0, 0.0, 10.0, 10.0]), 10.0, 0.95)
    assert bound.upper > 10.0


def test_empirical_bernstein_rejects_n_equals_one():
    with pytest.raises(ValueError, match="at least 2"):
        empirical_bernstein_mean_upper(np.array([1.0]), 10.0, 0.95)


def test_empirical_bernstein_on_a_constant_sample_keeps_only_the_range_term():
    """Edge case: V_n = 0 kills the square-root term exactly.

    slack = 0 + 7*10*ln(40)/(3*9) = 258.22156179 / 27 ... no: 7*10*3.6888794541139363
          = 258.22156178797554, divided by (3*(10-1)) = 27 -> 9.56376...
    """
    bound = empirical_bernstein_mean_upper(np.full(10, 4.0), 10.0, 0.95)
    expected = 7.0 * 10.0 * math.log(40.0) / 27.0
    assert bound.slack == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_empirical_bernstein_slack_matches_the_bound(field_sample):
    bound = empirical_bernstein_mean_upper(field_sample.gaps, field_sample.bound_range, 0.95)
    direct = empirical_bernstein_slack(
        field_sample.size,
        float(np.var(field_sample.gaps, ddof=1)),
        field_sample.bound_range,
        0.95,
    )
    assert bound.slack == pytest.approx(direct, rel=1e-13, abs=0.0)


def test_hoeffding_beats_empirical_bernstein_at_a_field_sample_size(field_sample):
    """A measured, reported fact and not an accident of this sample.

    At n = 60 with R = 40 kN.m the 7 R ln(2/delta) / (3(n-1)) term of equation
    (5) is 5.78 kN.m on its own, while the whole Hoeffding slack of equation
    (3) is 6.32. The empirical Bernstein bound only becomes the tighter of the
    two at a sample size this product's own subject matter -- a handful of
    flights -- does not reach. The crossover is measured in
    validation/validate_bounds_output.txt.
    """
    hoeffding = hoeffding_mean_upper(field_sample.gaps, field_sample.bound_range, 0.95)
    bernstein = empirical_bernstein_mean_upper(
        field_sample.gaps, field_sample.bound_range, 0.95
    )
    assert hoeffding.slack < bernstein.slack


def test_wilks_required_n_known_answer():
    """Hand calculation of equation (7) at p = 0.95, 1-delta = 0.95.

    ln(delta) / ln(p) = ln(0.05) / ln(0.95)
                      = -2.995732273553991 / -0.05129329438755058
                      = 58.40397481431972
    ceil                = 59
    """
    assert required_n_wilks(0.95, 0.95) == 59


@pytest.mark.parametrize(
    ("proportion", "confidence", "expected"),
    [
        (0.90, 0.90, 22),  # ln(0.10)/ln(0.90) = 21.854 -> 22
        (0.99, 0.95, 299),  # ln(0.05)/ln(0.99) = 298.07 -> 299
        (0.50, 0.50, 1),  # ln(0.5)/ln(0.5) = 1 exactly
    ],
)
def test_wilks_required_n_table(proportion, confidence, expected):
    assert required_n_wilks(proportion, confidence) == expected


def test_wilks_limit_is_the_largest_order_statistic():
    gaps = np.linspace(0.0, 9.0, 100)
    bound = wilks_tolerance_upper(gaps, 10.0, 0.95, 0.95)
    assert bound.upper == pytest.approx(9.0, rel=0.0, abs=1e-14)
    assert bound.target == "quantile"
    assert bound.distribution_free is True


def test_wilks_refuses_below_the_required_n(small_field):
    """At n = 15 the statement does not exist, and the message says so."""
    with pytest.raises(ValueError, match="needs n >= 59"):
        wilks_tolerance_upper(small_field.gaps, small_field.bound_range, 0.95, 0.95)


@pytest.mark.parametrize(
    ("n", "proportion", "confidence", "published"),
    [
        (3, 0.90, 0.95, 6.155),
        (5, 0.95, 0.95, 4.202),
        (10, 0.90, 0.95, 2.355),
        (20, 0.99, 0.95, 3.295),
    ],
)
def test_normal_tolerance_factor_against_published_table(n, proportion, confidence, published):
    """Equation (9) against the classical one-sided normal tolerance factors.

    These four values are the classical tabulated one-sided normal tolerance
    factors, as given in Krishnamoorthy and Mathew (2009), chapter 1, and
    reproduced in every standard quality-engineering handbook, quoted to three
    decimal places as the tables print them.

    The tolerance is 1e-3 and not 5e-4, and the reason is stated rather than
    tuned. The MEASURED deviations of this implementation from the four table
    values are 2.81e-4 at n = 3, 6.81e-4 at n = 5, 3.60e-4 at n = 10 and
    1.57e-4 at n = 20. A table printed to three decimals pins the factor only
    to within half a unit in the last place, 5e-4, and the n = 5 row exceeds
    that. Published tables disagree in the third decimal depending on whether
    the factor was computed from the exact non-central t distribution, which is
    what equation (9) and this implementation use, or from the Natrella
    normal-approximation formula that many printings use instead. The 1e-3
    tolerance covers that disagreement; a tolerance tight enough to exclude it
    would be testing which printing the table came from.
    """
    assert tolerance_factor_normal(n, proportion, confidence) == pytest.approx(
        published, rel=0.0, abs=1e-3
    )


def test_normal_tolerance_factor_decreases_with_n():
    factors = [tolerance_factor_normal(n, 0.95, 0.95) for n in (5, 10, 20, 50, 200)]
    pairs = list(zip(factors[:-1], factors[1:], strict=True))
    assert all(later < earlier for earlier, later in pairs)


def test_normal_tolerance_factor_tends_to_the_normal_quantile():
    """As n grows, k -> z_p. At p = 0.95, z_p = 1.6448536269514722."""
    from scipy import stats

    assert tolerance_factor_normal(200000, 0.95, 0.95) == pytest.approx(
        float(stats.norm.ppf(0.95)), rel=0.0, abs=0.01
    )


def test_normal_tolerance_upper_is_mean_plus_k_times_s(field_sample):
    bound = normal_tolerance_upper(field_sample.gaps, field_sample.bound_range, 0.95, 0.95)
    factor = tolerance_factor_normal(field_sample.size, 0.95, 0.95)
    expected = float(np.mean(field_sample.gaps)) + factor * float(
        np.std(field_sample.gaps, ddof=1)
    )
    assert bound.upper == pytest.approx(expected, rel=1e-13, abs=0.0)
    assert bound.distribution_free is False


def test_required_n_hoeffding_known_answer():
    """Hand calculation of equation (10) at R = 10, eps = 1, 1-delta = 0.95.

    R^2 ln(1/delta) / (2 eps^2) = 100 * 2.995732273553991 / 2 = 149.78661367769955
    ceil                         = 150
    """
    assert required_n_hoeffding(10.0, 1.0, 0.95) == 150


def test_required_n_hoeffding_round_trips_through_the_forward_slack():
    """The slack at the returned n is at or below the target, and above it at n-1."""
    for target in (0.5, 1.0, 2.5, 7.0):
        n = required_n_hoeffding(40.0, target, 0.95)
        slack = 40.0 * math.sqrt(math.log(20.0) / (2.0 * n))
        assert slack <= target
        if n > 1:
            previous = 40.0 * math.sqrt(math.log(20.0) / (2.0 * (n - 1)))
            assert previous > target


def test_required_n_empirical_bernstein_satisfies_the_forward_slack():
    for variance in (1.0, 16.0, 100.0):
        for target in (1.0, 2.0, 5.0):
            n = required_n_empirical_bernstein(variance, 40.0, target, 0.95)
            assert empirical_bernstein_slack(n, variance, 40.0, 0.95) <= target
            assert empirical_bernstein_slack(n - 1, variance, 40.0, 0.95) > target


def test_required_n_empirical_bernstein_closed_form_is_close_to_the_walk():
    """The n-1 neglected in equation (11) costs at most a few samples.

    The closed form solves b u^2 + a u - eps = 0 with n-1 replaced by n; the
    function then walks upward against the true expression. This test measures
    how far the walk goes, which is what the docstring claims is small.
    """
    log_term = math.log(2.0 / 0.05)
    for variance in (1.0, 16.0, 100.0):
        a = math.sqrt(2.0 * variance * log_term)
        b = 7.0 * 40.0 * log_term / 3.0
        for target in (1.0, 2.0, 5.0):
            u = (-a + math.sqrt(a * a + 4.0 * b * target)) / (2.0 * b)
            closed = max(2, math.ceil(1.0 / (u * u)))
            walked = required_n_empirical_bernstein(variance, 40.0, target, 0.95)
            assert 0 <= walked - closed <= 5


def test_normal_tolerance_required_n_is_a_monotone_search():
    n = normal_tolerance_required_n(0.95, 0.95, 2.0)
    assert tolerance_factor_normal(n, 0.95, 0.95) <= 2.0
    assert tolerance_factor_normal(n - 1, 0.95, 0.95) > 2.0


def test_normal_tolerance_required_n_rejects_an_unreachable_target():
    """k is bounded below by z_p, so no n reaches a target under it."""
    with pytest.raises(ValueError, match="bounded below"):
        normal_tolerance_required_n(0.95, 0.95, 1.0, max_n=20000)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_required_n_hoeffding_rejects_a_bad_target(bad):
    with pytest.raises(ValueError, match="target_slack"):
        required_n_hoeffding(10.0, bad, 0.95)


def test_required_n_empirical_bernstein_rejects_a_negative_variance():
    with pytest.raises(ValueError, match="variance"):
        required_n_empirical_bernstein(-1.0, 10.0, 1.0, 0.95)


def test_tolerance_factor_rejects_n_below_two():
    with pytest.raises(ValueError, match="n >= 2"):
        tolerance_factor_normal(1, 0.95, 0.95)
