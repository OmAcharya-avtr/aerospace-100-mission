"""The headline output object: normalisation, quantiles, truncation, units."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from passyield.distribution import (
    PASS_RESAMPLING_UNIT,
    ShortfallDistribution,
    distribution_from_samples,
    uniform_grid,
)


def _uniform(bins: int = 100, low: float = 0.0, high: float = 100.0) -> ShortfallDistribution:
    """A uniform distribution on ``[low, high]``, for hand calculations."""
    grid = uniform_grid(low, high, bins)
    return ShortfallDistribution("test", grid, np.full(bins, 1.0 / bins))


def test_uniform_grid_hand_calculation():
    """Hand-calculated: 4 bins on ``[0, 100]`` have centres 12.5, 37.5, 62.5, 87.5.

    The step is ``100/4 = 25`` and the centres are ``0 + (k + 0.5) * 25``.
    """
    grid = uniform_grid(0.0, 100.0, 4)
    assert list(grid) == pytest.approx([12.5, 37.5, 62.5, 87.5], rel=0.0, abs=1.0e-12)


def test_uniform_distribution_mean_and_quantiles_hand_calculated():
    """A uniform law on ``[0, 100]``: mean 50, median 50, 0.9 quantile 90.

    With 100 bins of width 1 the bin centres are 0.5, 1.5, ..., 99.5 and each
    carries mass 0.01. The mean is the average of the centres, which is 50.0
    exactly. The cumulative mass reaches 0.90 at the end of bin 90, so the
    linearly interpolated 0.90 quantile is 90.0.
    """
    law = _uniform()
    assert law.mean_mbit() == pytest.approx(50.0, rel=0.0, abs=1.0e-12)
    assert law.quantile(0.5) == pytest.approx(50.0, rel=0.0, abs=1.0e-12)
    assert law.quantile(0.9) == pytest.approx(90.0, rel=0.0, abs=1.0e-12)
    assert law.grid_step_mbit == pytest.approx(1.0, rel=0.0, abs=1.0e-12)
    assert law.quantile_discretisation_mbit() == pytest.approx(0.5, rel=0.0, abs=1.0e-12)


def test_uniform_distribution_standard_deviation_hand_calculated():
    """A discrete uniform on 100 bins of width 1 centred at 0.5 ... 99.5.

    Its variance is ``(n**2 - 1) * w**2 / 12`` with ``n = 100`` and
    ``w = 1``, which is ``9999/12 = 833.25``, so the standard deviation is
    ``sqrt(833.25) = 28.8660...``.
    """
    law = _uniform()
    assert law.standard_deviation_mbit() == pytest.approx(
        np.sqrt(833.25), rel=1.0e-12, abs=0.0
    )


def test_interval_is_equal_tailed():
    """A 0.8 interval of the uniform law on ``[0, 100]`` is ``[10, 90]``."""
    low, high = _uniform().interval(0.8)
    assert low == pytest.approx(10.0, rel=0.0, abs=1.0e-12)
    assert high == pytest.approx(90.0, rel=0.0, abs=1.0e-12)


def test_cdf_is_monotone_and_reaches_the_retained_mass():
    """``P(S <= x)`` rises from 0 to the retained mass."""
    law = _uniform()
    values = [law.cdf(x) for x in np.linspace(-10.0, 110.0, 25)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0, rel=0.0, abs=1.0e-12)
    assert values[-1] == pytest.approx(1.0, rel=0.0, abs=1.0e-12)


@given(level=st.floats(min_value=1.0e-4, max_value=1.0 - 1.0e-4))
@settings(max_examples=150, deadline=None)
def test_quantile_inverts_the_cumulative_distribution(level):
    """Property: the quantile lands inside the bin the cumulative mass reaches.

    The check is a bound of one grid step, not an equality, because the
    quantile of a binned law is only defined to within a bin.
    """
    law = _uniform()
    value = law.quantile(level)
    assert law.cdf(value) >= level - law.grid_step_mbit / 100.0 - 1.0e-12
    assert law.cdf(value - law.grid_step_mbit) <= level + 1.0e-12


def test_quantile_refuses_a_level_inside_the_truncated_tail():
    """A truncated tail holds no information, so no quantile is returned there."""
    grid = uniform_grid(0.0, 10.0, 10)
    law = ShortfallDistribution("test", grid, np.full(10, 0.08), truncated_mass=0.2)
    assert law.quantile(0.5) > 0.0
    with pytest.raises(ValueError, match="truncated upper tail"):
        law.quantile(0.9)


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.1, 2.0])
def test_quantile_rejects_levels_outside_the_open_unit_interval(level):
    """Input validation."""
    with pytest.raises(ValueError, match="level"):
        _uniform().quantile(level)


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.5, 1.5])
def test_interval_rejects_bad_coverage(coverage):
    """Input validation."""
    with pytest.raises(ValueError, match="coverage"):
        _uniform().interval(coverage)


def test_distribution_rejects_unnormalised_mass():
    """A law that does not normalise cannot have quantiles, so it is refused."""
    grid = uniform_grid(0.0, 10.0, 10)
    with pytest.raises(ValueError, match="must total 1"):
        ShortfallDistribution("test", grid, np.full(10, 0.05))


def test_distribution_rejects_negative_mass():
    """Input validation."""
    grid = uniform_grid(0.0, 10.0, 10)
    mass = np.full(10, 0.1)
    mass[3] = -0.1
    mass[4] = 0.3
    with pytest.raises(ValueError, match="non-negative"):
        ShortfallDistribution("test", grid, mass)


def test_distribution_rejects_a_non_uniform_grid():
    """Baseline 2 is a convolution, and a convolution needs a uniform grid."""
    grid = np.asarray([0.0, 1.0, 3.0, 6.0])
    with pytest.raises(ValueError, match="uniformly spaced"):
        ShortfallDistribution("test", grid, np.full(4, 0.25))


def test_distribution_rejects_a_decreasing_grid():
    """Input validation."""
    with pytest.raises(ValueError, match="strictly increasing"):
        ShortfallDistribution("test", np.asarray([3.0, 2.0, 1.0]), np.full(3, 1.0 / 3.0))


def test_distribution_rejects_a_single_bin():
    """A one-bin law has no resolution at all."""
    with pytest.raises(ValueError, match="at least 2 bins"):
        ShortfallDistribution("test", np.asarray([1.0]), np.asarray([1.0]))


def test_distribution_rejects_a_mass_of_the_wrong_shape():
    """Input validation."""
    with pytest.raises(ValueError, match="same shape"):
        ShortfallDistribution("test", uniform_grid(0.0, 10.0, 10), np.full(5, 0.2))


@pytest.mark.parametrize("truncated", [-0.1, 1.0, 1.5])
def test_distribution_rejects_bad_truncated_mass(truncated):
    """Input validation."""
    with pytest.raises(ValueError, match="truncated_mass"):
        ShortfallDistribution("test", uniform_grid(0.0, 10.0, 10), np.full(10, 0.1), truncated)


def test_samples_outside_the_grid_are_counted_not_clipped():
    """Mass outside the grid is reported, never pushed into the end bins.

    Clipping would turn an unrepresented tail into a spike at the boundary,
    and the spike would then read as a quantile.
    """
    grid = uniform_grid(0.0, 10.0, 10)
    samples = np.asarray([-5.0, 1.0, 2.0, 3.0, 50.0])
    law = distribution_from_samples(samples, grid, "monte-carlo")
    assert law.truncated_mass == pytest.approx(0.4, rel=0.0, abs=1.0e-12)
    assert float(law.mass.sum()) == pytest.approx(0.6, rel=0.0, abs=1.0e-12)


def test_distribution_from_samples_rejects_an_empty_sample():
    """Input validation."""
    with pytest.raises(ValueError, match="non-empty"):
        distribution_from_samples(np.asarray([]), uniform_grid(0.0, 10.0, 10), "monte-carlo")


def test_distribution_from_samples_rejects_non_finite_samples():
    """Input validation."""
    with pytest.raises(ValueError, match="finite"):
        distribution_from_samples(
            np.asarray([1.0, np.nan]), uniform_grid(0.0, 10.0, 10), "monte-carlo"
        )


def test_scoring_ensemble_is_deterministic_and_the_right_length():
    """Two calls agree exactly: the ensemble is a quantile grid, not a draw."""
    law = _uniform()
    first = law.samples_for_scoring(64)
    second = law.samples_for_scoring(64)
    assert first.size == 64
    assert np.array_equal(first, second)


def test_scoring_ensemble_rejects_a_tiny_count():
    """Input validation."""
    with pytest.raises(ValueError, match="count"):
        _uniform().samples_for_scoring(1)


def test_resampling_unit_is_carried_on_the_object():
    """The specification fixes the bootstrap unit as the pass; it is an attribute."""
    grid = uniform_grid(0.0, 10.0, 10)
    law = ShortfallDistribution(
        "bootstrap", grid, np.full(10, 0.1), resampling_unit=PASS_RESAMPLING_UNIT
    )
    assert law.resampling_unit == "pass"
    assert PASS_RESAMPLING_UNIT == "pass"


def test_uniform_grid_rejects_bad_arguments():
    """Input validation."""
    with pytest.raises(ValueError, match="high_mbit"):
        uniform_grid(10.0, 0.0, 10)
    with pytest.raises(ValueError, match="bins"):
        uniform_grid(0.0, 10.0, 1)


def test_mean_of_a_law_with_no_retained_mass_raises():
    """A law whose grid retains nothing has no mean, and says so."""
    grid = uniform_grid(0.0, 10.0, 10)
    law = ShortfallDistribution("test", grid, np.zeros(10), truncated_mass=1.0 - 1.0e-16)
    with pytest.raises(ValueError, match="retains no mass"):
        law.mean_mbit()
