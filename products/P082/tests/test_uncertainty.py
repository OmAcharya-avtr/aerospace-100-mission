"""Climatology sampling uncertainty: propagation and the share of equation (28).

The share has three stated properties -- it is 1 at zero discrepancy, it is not
a p-value, and the uncorrected form is biased toward zero -- and each of the
three is tested, including the bias, which is measured by simulation rather than
asserted.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.uncertainty import (
    FigureWithUncertainty,
    climatology_share,
    climatology_share_corrected,
    naive_product_relative_sd,
)


def test_the_relative_sd_of_a_two_factor_product_is_hand_checkable():
    """Equation (27): two rates of 0.4 each with a 4 per cent relative error.

    ``sqrt(0.04^2 + 0.04^2) = 0.04 sqrt(2) = 0.056569``, by hand.
    """
    got = naive_product_relative_sd(np.array([0.4, 0.4]), np.array([0.016, 0.016]))
    assert got == pytest.approx(0.04 * np.sqrt(2.0), rel=1e-14, abs=0.0)


def test_the_relative_sd_of_a_single_factor_is_its_own_relative_error():
    """Equation (27) with one term reduces to the term."""
    got = naive_product_relative_sd(np.array([0.5]), np.array([0.02]))
    assert got == pytest.approx(0.04, rel=1e-14, abs=0.0)


def test_the_relative_sd_is_zero_when_every_rate_is_known_exactly():
    """No sampling error in, no sampling error out. Exactly zero by the branch."""
    assert naive_product_relative_sd(np.full(4, 0.3), np.zeros(4)) == 0.0


@given(st.lists(st.floats(0.05, 0.95), min_size=1, max_size=8))
def test_the_relative_sd_grows_with_the_station_count(rates):
    """Each factor adds variance in quadrature, so more stations means more."""
    values = np.array(rates, dtype=float)
    errors = 0.02 * np.ones_like(values)
    whole = naive_product_relative_sd(values, errors)
    part = naive_product_relative_sd(values[:-1], errors[:-1]) if values.size > 1 else 0.0
    assert whole >= part - 1e-15


def test_the_relative_sd_rejects_a_rate_of_zero():
    """Logarithmic propagation divides by the rate."""
    with pytest.raises(ValueError, match="strictly positive"):
        naive_product_relative_sd(np.array([0.0, 0.4]), np.array([0.01, 0.01]))


def test_the_relative_sd_rejects_a_negative_standard_error():
    """Input validation."""
    with pytest.raises(ValueError, match="non-negative"):
        naive_product_relative_sd(np.array([0.4]), np.array([-0.01]))


def test_the_relative_sd_rejects_mismatched_inputs():
    """Input validation."""
    with pytest.raises(ValueError, match="same shape"):
        naive_product_relative_sd(np.zeros(3) + 0.3, np.zeros(4))


def test_the_relative_sd_rejects_an_empty_input():
    """Input validation."""
    with pytest.raises(ValueError, match="non-empty"):
        naive_product_relative_sd(np.array([]), np.array([]))


def test_the_analytic_propagation_is_a_lower_bound_on_the_bootstrap(
    reference_record, reference_climatology
):
    """Equation (27)'s declared independence approximation, measured.

    The propagation treats the stations' estimation errors as independent, and
    they are not: the series are correlated, which is the whole subject. The
    correlated case has the larger variance, so the analytic figure must come
    out BELOW a block bootstrap of the same product, which makes no
    independence assumption. This is the direction the module docstring claims
    and it is measured here on a near network rather than argued.
    """
    from sitediverse.empirical import block_indices
    from sitediverse.independence import product_of_marginals

    sites = [0, 1, 2, 3]
    rates = reference_climatology.long_term_rate[sites]
    errors = reference_climatology.rate_standard_error()[sites]
    analytic = naive_product_relative_sd(rates, errors) * product_of_marginals(rates)
    draws = block_indices(reference_record.epochs, 64, 300, seed=28)
    cloud = reference_record.cloud[:, sites]
    replicate = np.array(
        [float(np.exp(np.sum(np.log(cloud[draw].mean(axis=0))))) for draw in draws]
    )
    bootstrap = float(np.std(replicate, ddof=1))
    assert analytic < bootstrap
    assert bootstrap / analytic > 1.0


def test_the_share_is_one_at_a_discrepancy_of_zero():
    """Equation (28)'s first declared property, exact by the branch."""
    assert climatology_share(0.0, 0.01) == 1.0


def test_the_share_is_zero_at_a_zero_bootstrap_error():
    """A reference known exactly leaves no discrepancy to sampling."""
    assert climatology_share(0.1, 0.0) == 0.0


def test_the_share_is_nan_when_both_inputs_vanish():
    """No information at all, and ``nan`` says so rather than inventing a number."""
    assert np.isnan(climatology_share(0.0, 0.0))


def test_the_share_is_one_half_at_a_discrepancy_of_one_standard_error():
    """Equation (28)'s second declared property: this is NOT significance.

    A share of 0.5 is a one-sigma discrepancy, which for a two-sided 95 per
    cent test is nowhere near significant. The test pins the correspondence so
    that the warning in the docstring has a number behind it.
    """
    assert climatology_share(0.01, 0.01) == pytest.approx(0.5, rel=1e-14, abs=0.0)


def test_the_share_falls_monotonically_as_the_discrepancy_grows():
    """Which is what makes it readable as a share."""
    values = [climatology_share(gap, 0.01) for gap in (0.0, 0.005, 0.01, 0.05, 0.5)]
    for earlier, later in zip(values[:-1], values[1:], strict=True):
        assert later <= earlier


@given(
    st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    st.floats(1e-6, 1.0, allow_nan=False, allow_infinity=False),
)
def test_the_share_lies_in_the_unit_interval(discrepancy, noise):
    """A property over the whole input range."""
    got = climatology_share(discrepancy, noise)
    assert 0.0 <= got <= 1.0


def test_the_share_is_insensitive_to_the_sign_of_the_discrepancy():
    """It is a variance share, so only the magnitude matters."""
    assert climatology_share(0.02, 0.01) == climatology_share(-0.02, 0.01)


def test_the_corrected_share_exceeds_one_when_the_discrepancy_is_small():
    """Equation (28)'s third declared property, and why it is not clipped.

    The bias-corrected numerator subtracts ``s^2`` from the squared
    discrepancy, so where the observed discrepancy is smaller than the sampling
    noise -- the expected case for a model that is right -- the share exceeds
    one. Clipping it to 1.0 would make it indistinguishable from an exactly
    zero discrepancy, and the two mean different things.
    """
    assert climatology_share_corrected(0.005, 0.01) > 1.0
    assert climatology_share(0.005, 0.01) < 1.0


def test_the_corrected_share_agrees_with_the_uncorrected_one_at_a_large_discrepancy():
    """The correction is second order in ``s / gap`` and vanishes as that falls."""
    plain = climatology_share(1.0, 0.001)
    corrected = climatology_share_corrected(1.0, 0.001)
    assert corrected == pytest.approx(plain, rel=1e-5, abs=0.0)


def test_the_uncorrected_share_is_biased_toward_zero_by_simulation():
    """The bias of equation (28), measured rather than asserted.

    A model with zero structural error is compared against a reference that
    differs from the truth only by sampling noise, so the honest answer is that
    the whole discrepancy is sampling: a share of 1. The uncorrected form does
    not give 1. It gives ``E[1 / (1 + z^2)]`` with ``z`` standard normal, which
    is 0.655680 by quadrature -- and NOT 0.5, which is what a first guess from
    ``E[g^2] = s^2`` would say and which an earlier version of this test
    asserted. The expectation of a ratio is not the ratio of expectations, and
    the difference is 0.156.

    The quantity is scale-free: it does not depend on ``s``, which the test
    confirms at two values three decades apart. The tolerance is four standard
    errors of a mean of 20000 draws of a quantity bounded in [0, 1].
    """
    reference = 0.6556795424187994
    for noise in (0.01, 1e-5):
        rng = np.random.default_rng(282)
        plain = climatology_share_array(rng, noise, 20_000)
        assert float(np.mean(plain)) == pytest.approx(
            reference, rel=0.0, abs=0.01
        )
    # And the corrected form has no such floor: its mean is not 0.6557.
    rng = np.random.default_rng(283)
    corrected = np.array(
        [
            climatology_share_corrected(float(rng.normal(0.0, 0.01)), 0.01)
            for _ in range(2000)
        ]
    )
    assert abs(float(np.median(corrected)) - reference) > 0.1


def climatology_share_array(rng, noise: float, count: int) -> np.ndarray:
    """``count`` draws of equation (28) with a purely sampling discrepancy."""
    return np.array(
        [
            climatology_share(float(rng.normal(0.0, noise)), noise)
            for _ in range(count)
        ]
    )


def test_the_corrected_share_rejects_a_negative_bootstrap_error():
    """Input validation."""
    with pytest.raises(ValueError, match="non-negative"):
        climatology_share_corrected(0.1, -0.01)


def test_the_share_rejects_a_negative_bootstrap_error():
    """Input validation."""
    with pytest.raises(ValueError, match="non-negative"):
        climatology_share(0.1, -0.01)


def test_a_figure_derives_its_availability_from_its_unavailability():
    """Primary and derived, in the direction equation (14) requires."""
    figure = FigureWithUncertainty("x", 0.02, 0.001, "block-bootstrap")
    assert figure.availability == pytest.approx(0.98, rel=1e-14, abs=0.0)
    assert figure.relative_sd == pytest.approx(0.05, rel=1e-14, abs=0.0)


def test_a_figure_with_no_computed_uncertainty_renders_as_such():
    """A ``nan`` standard deviation must not render as a zero.

    A zero reads as certainty. The rendering says the uncertainty was not
    computed, which is a different and honest statement.
    """
    figure = FigureWithUncertainty("x", 0.02, float("nan"), "none")
    assert "not computed" in figure.render()
    assert "+/-" not in figure.render()


def test_a_figure_with_a_computed_uncertainty_renders_the_plus_minus_form():
    """Every availability figure this product reports takes this form."""
    figure = FigureWithUncertainty("x", 0.02, 0.0015, "block-bootstrap")
    rendered = figure.render()
    assert "0.020000 +/- 0.001500" == rendered


def test_a_figure_at_zero_unavailability_has_infinite_relative_error():
    """An unavailability of zero has no relative error."""
    figure = FigureWithUncertainty("x", 0.0, 0.001, "block-bootstrap")
    assert np.isinf(figure.relative_sd)
