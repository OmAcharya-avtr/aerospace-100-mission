"""The marginal climatology estimator and the effective sample size.

The estimator reads only the record. The ground truth is available in the test
fixtures and is used only to SCORE it, which is the one thing a synthetic record
buys and is checked here against four standard errors rather than against a
chosen constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from sitediverse.climatology import (
    NHOUR,
    NSEASON,
    SHRINKAGE,
    cell_index,
    estimate_climatology,
    lag_one_autocorrelation,
)
from sitediverse.field import EPOCHS_PER_DAY, EPOCHS_PER_YEAR


def test_the_declared_cell_grid_is_the_one_documented():
    """12 season cells by 8 hour cells, as equation (12) says."""
    assert NSEASON == 12
    assert NHOUR == EPOCHS_PER_DAY == 8
    assert SHRINKAGE == 4.0


def test_cell_index_partitions_a_year_into_nearly_equal_cells():
    """96 cells of 30 or 31 epochs each, not of equal size, and why.

    2920 epochs do not divide by 96: a declared year is 365 days and a season
    cell is a twelfth of it, which is 30.42 days, so the integer season
    boundary falls at a different hour each time and the cells carry 30 or 31
    epochs. The imbalance is one part in thirty and it is carried by the
    occupancy weighting of equation (12), which divides each cell's total by
    its OWN count rather than by a nominal one. The test pins the imbalance so
    that a change in the cell grid cannot silently make it large.
    """
    cells = cell_index(EPOCHS_PER_YEAR)
    counts = np.bincount(cells, minlength=NSEASON * NHOUR)
    assert counts.size == NSEASON * NHOUR
    assert np.all(counts > 0)
    assert int(counts.max() - counts.min()) == 1
    assert int(counts.sum()) == EPOCHS_PER_YEAR


def test_cell_index_is_periodic_in_the_year():
    """The calendar repeats, so the cell labels do."""
    cells = cell_index(2 * EPOCHS_PER_YEAR)
    assert np.array_equal(cells[:EPOCHS_PER_YEAR], cells[EPOCHS_PER_YEAR:])


def test_cell_index_separates_the_hours_of_the_day():
    """Consecutive epochs land in consecutive hour cells within one season cell."""
    cells = cell_index(16)
    assert cells[0] != cells[1]
    assert cells[0] == cells[8]


def test_cell_index_rejects_an_empty_record():
    """Input validation."""
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        cell_index(0)


def test_cell_index_rejects_a_zero_cell_count():
    """Input validation."""
    with pytest.raises(ValueError, match="at least 1"):
        cell_index(100, nseason=0)


def test_the_estimated_rate_stays_strictly_inside_the_unit_interval(
    short_climatology,
):
    """Equation (12)'s shrinkage exists exactly to guarantee this.

    A cell rate of 0 or 1 gives an infinite probit threshold and propagates an
    infinity into equation (8).
    """
    assert float(short_climatology.cell_rate.min()) > 0.0
    assert float(short_climatology.cell_rate.max()) < 1.0
    assert np.all(np.isfinite(short_climatology.threshold_series()))


def test_the_estimated_long_term_rate_is_the_observed_cloud_rate(
    short_record, short_climatology
):
    """It is a plain column mean, so it is exact up to summation rounding."""
    observed = short_record.cloud.mean(axis=0)
    assert float(
        np.max(np.abs(short_climatology.long_term_rate - observed))
    ) < 1e-15


def test_the_estimated_cell_rate_recovers_the_declared_climatology(
    reference_record, reference_climatology
):
    """Equation (12) scored against the generator's own ``q_it``.

    The tolerance is DERIVED. Each cell holds ``epochs / 96`` observations of
    which about ``(1 - phi) / (1 + phi)`` are independent, so the standard
    error of a cell rate near 0.4 is the figure below, and the criterion is
    four of them. It is a per-cell criterion applied to the worst cell over 96
    cells and 20 stations, so a four-sigma criterion on 1920 cells is expected
    to be met: the Bonferroni-corrected two-sided level at 1920 comparisons is
    3.7 sigma.
    """
    from scipy.stats import norm

    truth = norm.cdf(reference_record.threshold_true)
    estimated = reference_climatology.rate_series()
    per_cell = reference_record.epochs / (NSEASON * NHOUR)
    effective = per_cell * (1.0 - reference_record.spec.persistence) / (
        1.0 + reference_record.spec.persistence
    )
    tolerance = 4.0 * float(np.sqrt(0.25 / effective))
    assert float(np.max(np.abs(estimated - truth))) < tolerance


def test_the_estimated_cell_rate_is_unbiased_on_average(
    reference_record, reference_climatology
):
    """The mean error over all cells and stations is far below one cell's error.

    The shrinkage of equation (12) biases each cell toward the station mean,
    which is a bias toward the station's own long-term rate and therefore
    nearly unbiased in the aggregate. The size of that bias is what this test
    bounds.
    """
    from scipy.stats import norm

    truth = norm.cdf(reference_record.threshold_true)
    estimated = reference_climatology.rate_series()
    assert abs(float(np.mean(estimated - truth))) < 0.005


def test_the_shrinkage_pulls_an_extreme_cell_toward_the_station_mean():
    """Equation (12)'s mechanism, on a hand-built record.

    A station that is clear in one cell and cloudy elsewhere. Without
    shrinkage the cell rate is exactly 0; with ``nu = 4`` it is
    ``4 m / (n + 4)`` where ``m`` is the station's long-term rate.
    """
    epochs = 4 * EPOCHS_PER_YEAR
    cloud = np.ones((epochs, 2), dtype=bool)
    cells = cell_index(epochs)
    cloud[cells == 0, :] = False
    estimated = estimate_climatology(cloud, shrinkage=4.0)
    occupancy = float(np.count_nonzero(cells == 0))
    long_term = float(cloud[:, 0].mean())
    expected = 4.0 * long_term / (occupancy + 4.0)
    assert estimated.cell_rate[0, 0] == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_zero_shrinkage_is_permitted_and_raises_on_a_degenerate_cell():
    """It is allowed so that validation can measure what the shrinkage buys.

    With ``nu = 0`` an all-clear cell gives a rate of exactly 0, an infinite
    probit threshold, and the estimator raises with an actionable message
    rather than returning an infinity downstream.
    """
    epochs = 4 * EPOCHS_PER_YEAR
    cloud = np.ones((epochs, 2), dtype=bool)
    cloud[cell_index(epochs) == 0, :] = False
    with pytest.raises(ValueError, match="reached 0 or 1"):
        estimate_climatology(cloud, shrinkage=0.0)


def test_the_lag_one_autocorrelation_of_an_exact_ar_one_series_recovers_its_coefficient():
    """A known-answer case on a series whose coefficient is declared."""
    rng = np.random.default_rng(606)
    phi = 0.65
    epochs = 200_000
    series = np.empty((epochs, 1))
    series[0, 0] = rng.standard_normal()
    noise = rng.standard_normal(epochs) * np.sqrt(1.0 - phi**2)
    for epoch in range(1, epochs):
        series[epoch, 0] = phi * series[epoch - 1, 0] + noise[epoch]
    got = float(lag_one_autocorrelation(series)[0])
    assert got == pytest.approx(phi, rel=0.0, abs=0.01)


def test_the_lag_one_autocorrelation_of_white_noise_is_near_zero():
    """A control: an independent series has no lag-one correlation."""
    rng = np.random.default_rng(607)
    series = rng.standard_normal((100_000, 1))
    assert abs(float(lag_one_autocorrelation(series)[0])) < 0.02


def test_the_lag_one_autocorrelation_of_a_constant_column_is_zero():
    """A degenerate series has no correlation defined, and 0.0 declines to inflate."""
    assert float(lag_one_autocorrelation(np.ones((100, 1)))[0]) == 0.0


def test_the_indicator_lag_one_is_below_the_latent_lag_one(reference_record):
    """Thresholding destroys correlation, which is why the name says INDICATOR.

    Equation (13) uses the indicator's value and the module docstring says the
    choice is conservative. This test is the measurement behind that sentence.
    """
    latent = lag_one_autocorrelation(reference_record.latent)
    indicator = lag_one_autocorrelation(reference_record.cloud)
    assert np.all(indicator < latent)
    assert float(np.max(indicator)) < reference_record.spec.persistence


def test_the_lag_one_autocorrelation_rejects_a_one_dimensional_series():
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        lag_one_autocorrelation(np.zeros(10))


def test_the_lag_one_autocorrelation_rejects_a_record_of_two_epochs():
    """Input validation."""
    with pytest.raises(ValueError, match="at least 3 epochs"):
        lag_one_autocorrelation(np.zeros((2, 3)))


def test_the_effective_sample_size_is_below_the_epoch_count(short_climatology):
    """Equation (13) deflates, never inflates, for a positively correlated series."""
    effective = short_climatology.effective_sample_size()
    assert np.all(effective < short_climatology.epochs)
    assert np.all(effective >= 1.0)


def test_the_effective_sample_size_matches_the_declared_formula(short_climatology):
    """``n (1 - phi) / (1 + phi)``, checked against its own definition."""
    phi = np.clip(short_climatology.lag_one_indicator, 0.0, 0.999)
    expected = short_climatology.epochs * (1.0 - phi) / (1.0 + phi)
    assert float(
        np.max(np.abs(short_climatology.effective_sample_size() - expected))
    ) < 1e-9


def test_the_rate_standard_error_exceeds_the_naive_binomial_one(short_climatology):
    """What equation (13) is for: the naive binomial error understates.

    The ratio is ``sqrt((1 + phi) / (1 - phi))``, which at the measured
    indicator lag-one of this record is above 1.5.
    """
    rate = short_climatology.long_term_rate
    naive = np.sqrt(rate * (1.0 - rate) / short_climatology.epochs)
    honest = short_climatology.rate_standard_error()
    assert np.all(honest > naive)
    assert float(np.min(honest / naive)) > 1.5


def test_the_estimated_long_term_rate_is_within_four_standard_errors_of_the_truth(
    reference_record, reference_climatology
):
    """The headline accuracy claim for the climatology, on its own ground truth."""
    truth = reference_record.marginal_rate_true()
    errors = reference_climatology.rate_standard_error()
    standardised = np.abs(reference_climatology.long_term_rate - truth) / errors
    assert float(np.max(standardised)) < 4.0


def test_the_climatology_reports_its_own_shape(short_record, short_climatology):
    """Accessors agree with the arrays they describe."""
    assert short_climatology.site_count == short_record.site_count
    assert short_climatology.epochs == short_record.epochs
    assert short_climatology.rate_series().shape == short_record.cloud.shape
    assert short_climatology.threshold_series().shape == short_record.cloud.shape


@given(st.integers(1, 20))
def test_a_larger_effective_window_gives_a_smaller_standard_error(divisor):
    """The standard error falls as the square root of the window length."""
    epochs = 4 * EPOCHS_PER_YEAR
    rng = np.random.default_rng(99)
    cloud = rng.random((epochs, 3)) < 0.4
    climatology = estimate_climatology(cloud)
    whole = climatology.rate_standard_error()
    part = climatology.rate_standard_error(epochs // divisor)
    assert np.all(part >= whole - 1e-15)


def test_estimate_climatology_rejects_a_one_dimensional_record():
    """Input validation."""
    with pytest.raises(ValueError, match="two-dimensional"):
        estimate_climatology(np.zeros(10, dtype=bool))


def test_estimate_climatology_rejects_a_record_of_two_epochs():
    """Input validation."""
    with pytest.raises(ValueError, match="at least 3 epochs"):
        estimate_climatology(np.zeros((2, 3), dtype=bool))


def test_estimate_climatology_rejects_a_negative_shrinkage():
    """Input validation."""
    with pytest.raises(ValueError, match="non-negative"):
        estimate_climatology(np.zeros((100, 2), dtype=bool), shrinkage=-1.0)


def test_estimate_climatology_rejects_a_record_too_short_for_its_cells():
    """A cell with no epochs in it has no rate, and the message says why."""
    with pytest.raises(ValueError, match="cells are"):
        estimate_climatology(np.zeros((40, 2), dtype=bool))


def test_the_effective_sample_size_rejects_a_zero_window(short_climatology):
    """Input validation."""
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        short_climatology.effective_sample_size(0)
