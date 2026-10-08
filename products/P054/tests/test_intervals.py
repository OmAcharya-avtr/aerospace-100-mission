"""Known-answer, edge-case and validation tests for the interval module."""

from __future__ import annotations

import math

import pytest
from scipy import stats

from rareverify.intervals import (
    clopper_pearson,
    exact_coverage,
    proportion_interval,
    rule_of_three_upper,
    wilson,
    zero_failure_upper,
)


def test_clopper_pearson_zero_failure_one_sided_hand_calculation():
    """k=0, n=100, 95 % one-sided upper: 1 - 0.05 ** (1/100).

    Hand calculation:
        ln(0.05)      = -2.99573227355399
        / 100         = -0.0299573227355399
        exp(...)      =  0.970486960...
        1 - that      =  0.029513039...
    """
    result = clopper_pearson(0, 100, confidence=0.95, side="upper")
    assert result.lower == 0.0
    assert result.upper == pytest.approx(0.0295130496, abs=1e-9)
    assert result.upper == pytest.approx(1.0 - 0.05 ** (1.0 / 100), rel=1e-12)


def test_clopper_pearson_zero_failure_two_sided_hand_calculation():
    """k=0, n=100, 95 % two-sided upper: 1 - 0.025 ** (1/100).

    Hand calculation:
        ln(0.025) = -3.68887945411394
        / 100     = -0.0368887945411394
        exp(...)  =  0.963783307...
        1 - that  =  0.036216693...
    """
    result = clopper_pearson(0, 100, confidence=0.95, side="two-sided")
    assert result.upper == pytest.approx(0.0362166926, abs=1e-9)


def test_wilson_zero_failure_hand_calculation():
    """k=0, n=100, 95 % two-sided Wilson upper: z^2 / (n + z^2).

    Hand calculation with z = 1.959963984540054:
        z^2        = 3.8414588206941254
        n + z^2    = 103.84145882069413
        ratio      = 0.036993498...
    and the lower limit is exactly 0 because centre = halfwidth at k = 0.
    """
    result = wilson(0, 100, confidence=0.95, side="two-sided")
    assert result.lower == 0.0
    assert result.upper == pytest.approx(0.0369934982, abs=1e-9)


def test_wilson_is_wider_than_clopper_pearson_at_zero_failures():
    """A measured counter-example to 'Clopper-Pearson is always the wider one'.

    At k = 0, n = 100, 95 % two-sided: Wilson upper 0.0369935 exceeds
    Clopper-Pearson upper 0.0362167. The ordering that holds in the interior
    does not hold at the boundary, which is why no test in this suite asserts
    containment of one interval in the other.
    """
    cp = clopper_pearson(0, 100).upper
    wi = wilson(0, 100).upper
    assert wi > cp
    assert wi - cp == pytest.approx(7.76805563e-4, rel=1e-6)


def test_rule_of_three_relative_error_tends_to_a_non_zero_limit():
    """3/n does NOT converge to the exact 95 % zero-failure bound.

    The exact bound expands as

        1 - 0.05 ** (1/n) = -ln(0.05)/n - (ln 0.05)^2 / (2 n^2) - ...
                          = 2.995732/n - 4.487/n^2 - ...

    so 3/n has a relative error tending to 3 / 2.995732 - 1 = 1.424602e-3 from
    below, not to zero. Measured here at three decades. A test asserting
    convergence to zero would be wrong, and this is the correct property.
    """
    limit = 3.0 / 2.9957322735539909 - 1.0
    assert limit == pytest.approx(1.4246021e-3, rel=1e-6)
    errors = []
    for n in (100, 1000, 10000, 100000):
        exact = zero_failure_upper(n, confidence=0.95, side="upper")
        errors.append(rule_of_three_upper(n) / exact - 1.0)
    assert errors[0] == pytest.approx(1.6499494e-2, rel=1e-6)
    assert errors[1] == pytest.approx(2.9253510e-3, rel=1e-6)
    assert errors[2] == pytest.approx(1.5746096e-3, rel=1e-6)
    assert errors[3] == pytest.approx(1.4396022e-3, rel=1e-6)
    # Monotone decreasing towards the limit, approached from above.
    assert errors[0] > errors[1] > errors[2] > errors[3] > limit


def test_zero_failure_closed_form_matches_general_code_path():
    """The k=0 closed form must equal the Beta-quantile path for both methods."""
    for n in (1, 2, 10, 137, 5000):
        for method in ("clopper-pearson", "wilson"):
            closed = zero_failure_upper(n, 0.95, method=method, side="upper")
            general = proportion_interval(
                0, n, confidence=0.95, method=method, side="upper"
            ).upper
            assert closed == pytest.approx(general, rel=1e-12, abs=1e-15)


def test_full_failure_and_zero_failure_boundaries_are_exact():
    for n in (1, 5, 1000):
        assert clopper_pearson(0, n).lower == 0.0
        assert clopper_pearson(n, n).upper == 1.0
        assert wilson(0, n).lower == 0.0
        assert wilson(n, n).upper == pytest.approx(1.0, abs=1e-12)


def test_one_sided_intervals_force_the_other_limit():
    upper = clopper_pearson(3, 50, side="upper")
    assert upper.lower == 0.0
    assert upper.upper < 1.0
    lower = clopper_pearson(3, 50, side="lower")
    assert lower.upper == 1.0
    assert lower.lower > 0.0


def test_clopper_pearson_known_answer_interior():
    """k=5, n=1000: Beta quantiles, cross-checked against scipy directly.

    lower = BetaInv(0.025; 5, 996), upper = BetaInv(0.975; 6, 995).
    """
    result = clopper_pearson(5, 1000, confidence=0.95)
    assert result.lower == pytest.approx(float(stats.beta.ppf(0.025, 5, 996)), rel=1e-12)
    assert result.upper == pytest.approx(float(stats.beta.ppf(0.975, 6, 995)), rel=1e-12)
    assert result.lower == pytest.approx(1.6254204e-3, rel=1e-6)
    assert result.upper == pytest.approx(1.1629468e-2, rel=1e-6)


def test_exact_coverage_of_clopper_pearson_is_at_least_nominal():
    """Exact, not simulated: sum the binomial mass over covering k."""
    for p in (0.01, 0.05, 0.1, 0.3, 0.5):
        coverage = exact_coverage(40, p, confidence=0.95, method="clopper-pearson")
        assert coverage >= 0.95


def test_exact_coverage_of_wilson_dips_below_nominal_somewhere():
    """A measured failure of the Wilson interval, reported rather than hidden."""
    dips = [
        p
        for p in (0.01 * i for i in range(1, 50))
        if exact_coverage(40, p, confidence=0.95, method="wilson") < 0.95
    ]
    assert dips, "expected at least one p where the Wilson interval under-covers"


def test_coverage_at_p_zero_and_one():
    assert exact_coverage(10, 0.0, method="clopper-pearson") == pytest.approx(1.0)
    assert exact_coverage(10, 1.0, method="clopper-pearson") == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("k", "n"),
    [(-1, 10), (11, 10), (0, 0), (0, -5)],
)
def test_invalid_counts_raise_value_error(k, n):
    with pytest.raises(ValueError):
        clopper_pearson(k, n)


def test_non_integer_counts_raise_type_error():
    with pytest.raises(TypeError):
        clopper_pearson(1.5, 10)
    with pytest.raises(TypeError):
        clopper_pearson(True, 10)


@pytest.mark.parametrize("confidence", [0.0, 1.0, -0.1, 1.5, math.nan])
def test_invalid_confidence_raises(confidence):
    with pytest.raises(ValueError):
        clopper_pearson(1, 10, confidence=confidence)


def test_invalid_side_and_method_raise():
    with pytest.raises(ValueError):
        clopper_pearson(1, 10, side="both")
    with pytest.raises(ValueError):
        proportion_interval(1, 10, method="wald")
    with pytest.raises(ValueError):
        zero_failure_upper(10, method="jeffreys")
    with pytest.raises(ValueError):
        zero_failure_upper(10, side="lower")
    with pytest.raises(ValueError):
        exact_coverage(10, 1.5)


def test_describe_and_properties():
    result = clopper_pearson(0, 1000, side="upper")
    assert result.zero_failure is True
    assert result.width == pytest.approx(result.upper)
    assert "clopper-pearson" in result.describe()
    assert result.contains(1e-5)
    assert not result.contains(0.9)
