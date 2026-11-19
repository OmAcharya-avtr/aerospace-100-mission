"""The three baselines: the booked figure, the convolution, the accounted truth."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from passyield.baselines import (
    DEFAULT_BINS,
    POISSON_TAIL_TOLERANCE,
    ConvolutionOptions,
    EventModel,
    accounted_yield,
    analytic_shortfall_distribution,
    backlog_dry_time_s,
    booked_shortfall_mbit,
    cumulative_volume,
    fit_event_model,
    monte_carlo_shortfall_distribution,
)
from passyield.bounds import inside, shortfall_bracket
from passyield.decompose import decompose_pass
from passyield.link import DEFAULT_BUDGET

#: A declared event model for the unit tests, independent of the record.
TEST_MODEL = EventModel(
    acquisition_shape=4.0,
    acquisition_scale=10.0,
    outage_intensity_per_s=0.004,
    outage_mean_s=14.0,
    idle_shape=3.0,
    idle_scale=9.0,
    sample_passes=500,
)


# ---------------------------------------------------------------- baseline 1


def test_baseline_one_predicts_no_shortfall():
    """A schedule that books ``T * R_nom`` predicts ``T * R_nom`` will arrive.

    Zero is the whole content of baseline 1 as a predictor, and it is in the
    code rather than only in a document so that the comparison tables have a
    call to point at.
    """
    assert booked_shortfall_mbit() == 0.0


def test_baseline_one_booked_volume_is_duration_times_a_real_rate(
    satrec, station, long_window, events
):
    """The booked volume is the geometric duration times a link-budget rate.

    Full strength, not a straw man: the duration is the true mask-to-mask
    window from SGP4 and the rate comes from the budget at the declared
    reference elevation, which is what a scheduler with an ephemeris and a
    station catalogue can compute before the pass.
    """
    item = decompose_pass(satrec, station, long_window, events)
    implied_rate = item.nominal_bits / item.window_duration_s
    assert 0.0 < implied_rate <= DEFAULT_BUDGET.max_rate_mbit_s
    assert item.window_duration_s == pytest.approx(
        long_window.duration_s, rel=1.0e-14, abs=0.0
    )


# ---------------------------------------------------------------- baseline 2


def test_cumulative_volume_starts_at_zero_and_is_non_decreasing(satrec, station, long_window):
    """``Lambda(0) = 0`` and ``Lambda`` rises, because the rate is non-negative."""
    offsets, cumulative = cumulative_volume(satrec, station, long_window)
    assert float(cumulative[0]) == 0.0
    assert np.all(np.diff(cumulative) >= -1.0e-12)
    assert float(offsets[-1]) == pytest.approx(long_window.duration_s, rel=1.0e-14, abs=0.0)


def test_cumulative_volume_endpoint_matches_the_simpson_capacity(
    satrec, station, long_window, events
):
    """Two quadratures of the same integral agree to the stated tolerance.

    ``scipy.integrate.cumulative_simpson`` at the window end against the
    composite Simpson sum in ``passyield.decompose``. Compared with a
    tolerance, never bit-exactly: the two accumulate different partial sums.
    """
    item = decompose_pass(satrec, station, long_window, events)
    _offsets, cumulative = cumulative_volume(satrec, station, long_window)
    assert float(cumulative[-1]) == pytest.approx(item.capacity_bits, rel=1.0e-6, abs=0.0)


def test_cumulative_volume_rejects_an_even_node_count(satrec, station, long_window):
    """Input validation on the quadrature."""
    with pytest.raises(ValueError, match="profile_nodes"):
        cumulative_volume(satrec, station, long_window, profile_nodes=512)


def test_analytic_distribution_normalises_and_reports_its_truncation(
    satrec, station, long_window, events
):
    """The convolution's mass plus its reported truncation totals 1.

    The truncation is the probability the independence approximation placed
    above the window capacity plus the Poisson truncation; it is reported, not
    discarded, because an unreported truncation turns a quantile into a lower
    bound.
    """
    item = decompose_pass(satrec, station, long_window, events)
    law = analytic_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits
    )
    assert law.method == "numerical-convolution"
    assert float(law.mass.sum()) + law.truncated_mass == pytest.approx(
        1.0, rel=1.0e-9, abs=1.0e-12
    )
    assert law.truncated_mass >= 0.0


def test_analytic_and_monte_carlo_quantiles_agree_within_the_discretisation(
    satrec, station, long_window, events
):
    """Two methods on the same laws: the convolution against a direct draw.

    The tolerance is DERIVED, not tuned, from two terms.

    A discrete convolution adds BIN INDICES, not values, so each convolution
    stage can displace the result by up to one grid step. The stages on a
    pass are one for the acquisition law, one for the idle law, and one per
    outage in the compound Poisson sum truncated at its 1-in-1e9 quantile, so
    the discretisation term is ``(1 + stages)`` grid steps.

    The Monte Carlo quantile's standard error is
    ``4*sigma*sqrt(p(1-p)/n)``, using ``1/(4 sigma)`` as a crude lower bound
    on the density of a unimodal law; three of those.

    ``validation/validate_convolution.py`` measures the worst disagreement
    over 24 passes of the committed campaign and reports it in grid steps.
    """
    item = decompose_pass(satrec, station, long_window, events)
    analytic = analytic_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits
    )
    draws = 60000
    direct = monte_carlo_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
        seed=99099, draws=draws,
    )
    spread = analytic.standard_deviation_mbit()
    intensity = TEST_MODEL.outage_intensity_per_s * long_window.duration_s
    stages = 2 + int(stats.poisson.isf(POISSON_TAIL_TOLERANCE, mu=intensity)) + 1
    for level in (0.1, 0.25, 0.5, 0.75, 0.9):
        sampling = 4.0 * spread * float(np.sqrt(level * (1.0 - level) / draws))
        tolerance = (1 + stages) * analytic.grid_step_mbit + 3.0 * sampling
        assert analytic.quantile(level) == pytest.approx(
            direct.quantile(level), rel=0.0, abs=tolerance
        )


def test_analytic_quantiles_lie_inside_the_independent_bracket(
    satrec, station, long_window, events
):
    """Every quantile of baseline 2 is inside a bracket neither method uses.

    The bracket comes from the pass duration and the rate ceiling alone: no
    quadrature of the rate profile, no pushforward, no convolution, no random
    numbers. It brackets the convolution and the Monte Carlo together, which
    an agreement test between those two cannot do.
    """
    item = decompose_pass(satrec, station, long_window, events)
    bracket = shortfall_bracket(
        item.nominal_bits, item.window_duration_s, DEFAULT_BUDGET.max_rate_mbit_s
    )
    analytic = analytic_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits
    )
    direct = monte_carlo_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
        seed=99099, draws=20000,
    )
    for level in (0.01, 0.1, 0.5, 0.9, 0.99):
        assert inside(analytic.quantile(level), bracket)
        assert inside(direct.quantile(level), bracket)


def test_finer_grids_move_the_analytic_quantile_less(satrec, station, long_window, events):
    """Halving the grid step twice: the quantile changes converge.

    Asserted as a direction -- the second refinement moves the quantile less
    than the first -- rather than as an order, because the pushforward's
    binning is first order and the source grids contribute their own error.
    ``validation/validate_convolution.py`` measures and publishes the
    observed order.
    """
    item = decompose_pass(satrec, station, long_window, events)
    values = []
    for bins in (DEFAULT_BINS // 4, DEFAULT_BINS // 2, DEFAULT_BINS):
        law = analytic_shortfall_distribution(
            satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
            ConvolutionOptions(bins=bins),
        )
        values.append(law.quantile(0.5))
    first = abs(values[1] - values[0])
    second = abs(values[2] - values[1])
    assert second <= first + 1.0e-9


def test_strengthened_variant_shifts_the_law_by_the_dry_volume(
    satrec, station, long_window, events
):
    """A deterministic dry interval shifts the grid by its exact volume.

    The variant replaces the fitted idle law by the backlog-limited dry time a
    scheduler can compute before the pass. Shifting the grid is the same
    operation as convolving a delta and avoids a binning rounding, so the
    shift must be exact.
    """
    item = decompose_pass(satrec, station, long_window, events)
    offsets, cumulative = cumulative_volume(satrec, station, long_window)
    dry = 30.0
    expected = float(cumulative[-1]) - float(
        np.interp(long_window.duration_s - dry, offsets, cumulative)
    )
    base = analytic_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
        deterministic_idle_s=0.0,
    )
    shifted = analytic_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
        deterministic_idle_s=dry,
    )
    assert float(shifted.grid_mbit[0] - base.grid_mbit[0]) == pytest.approx(
        expected, rel=1.0e-9, abs=0.0
    )


def test_strengthened_variant_rejects_a_negative_dry_time(satrec, station, long_window, events):
    """Input validation."""
    item = decompose_pass(satrec, station, long_window, events)
    with pytest.raises(ValueError, match="deterministic_idle_s"):
        analytic_shortfall_distribution(
            satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
            deterministic_idle_s=-1.0,
        )


def test_analytic_distribution_rejects_a_non_positive_capacity(satrec, station, long_window):
    """Input validation: the volume grid spans ``[0, capacity]``."""
    with pytest.raises(ValueError, match="capacity_bits"):
        analytic_shortfall_distribution(
            satrec, station, long_window, TEST_MODEL, 1000.0, 0.0
        )


def test_monte_carlo_rejects_too_few_draws(satrec, station, long_window, events):
    """Input validation: a quantile from 10 draws is not a quantile."""
    item = decompose_pass(satrec, station, long_window, events)
    with pytest.raises(ValueError, match="draws"):
        monte_carlo_shortfall_distribution(
            satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
            seed=1, draws=10,
        )


def test_monte_carlo_is_reproducible_at_a_declared_seed(satrec, station, long_window, events):
    """Two runs at the same seed give the same law, bin for bin.

    This one IS bit-exact, and legitimately: it is the same arithmetic on the
    same inputs in the same order, which is what a seed is for. It is not a
    claim about a floating-point pipeline agreeing with a different one.
    """
    item = decompose_pass(satrec, station, long_window, events)
    first = monte_carlo_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
        seed=99099, draws=5000,
    )
    second = monte_carlo_shortfall_distribution(
        satrec, station, long_window, TEST_MODEL, item.nominal_bits, item.capacity_bits,
        seed=99099, draws=5000,
    )
    assert np.array_equal(first.mass, second.mass)


def test_backlog_dry_time_is_zero_when_the_backlog_exceeds_the_capacity(
    satrec, station, long_window
):
    """A full queue cannot run dry, so the dry time is exactly zero."""
    _offsets, cumulative = cumulative_volume(satrec, station, long_window)
    assert backlog_dry_time_s(
        satrec, station, long_window, float(cumulative[-1]) * 2.0
    ) == 0.0


def test_backlog_dry_time_is_the_whole_window_for_an_empty_queue(satrec, station, long_window):
    """An empty queue runs dry at once, so the dry time is the whole window."""
    dry = backlog_dry_time_s(satrec, station, long_window, 0.0)
    assert dry == pytest.approx(long_window.duration_s, rel=1.0e-9, abs=0.0)


def test_backlog_dry_time_rejects_a_negative_backlog(satrec, station, long_window):
    """Input validation."""
    with pytest.raises(ValueError, match="backlog_mbit"):
        backlog_dry_time_s(satrec, station, long_window, -1.0)


def test_poisson_tail_tolerance_is_declared():
    """The compound sum's truncation point is a declared constant."""
    assert 0.0 < POISSON_TAIL_TOLERANCE < 1.0e-6


@pytest.mark.parametrize(
    ("bins", "profile", "time", "duration", "placement"),
    [(8, 513, 257, 49, 33), (1024, 512, 257, 49, 33), (1024, 513, 2, 49, 33),
     (1024, 513, 257, 2, 33), (1024, 513, 257, 49, 2)],
)
def test_convolution_options_reject_out_of_range_settings(
    bins, profile, time, duration, placement
):
    """Input validation on the discretisation."""
    with pytest.raises(ValueError):
        ConvolutionOptions(bins, profile, time, duration, placement)


# ---------------------------------------------------------------- the event model


def test_fit_event_model_recovers_declared_moments():
    """Known answer: method of moments on a sample with known mean and variance.

    A sample of ``[10, 20, 30, 40]`` has mean 25 and unbiased variance
    ``((15**2 + 5**2 + 5**2 + 15**2) / 3) = 500/3 = 166.667``. The Gamma
    method of moments gives ``scale = var/mean = 6.6667`` and
    ``shape = mean/scale = 3.75``.
    """
    model = fit_event_model(
        acquisition_s=[10.0, 20.0, 30.0, 40.0],
        outage_counts=[1, 2],
        window_durations_s=[100.0, 200.0],
        outage_durations_s=[10.0, 12.0, 14.0],
        idle_totals_s=[5.0, 15.0],
    )
    assert model.acquisition_scale == pytest.approx(500.0 / 3.0 / 25.0, rel=1.0e-12, abs=0.0)
    assert model.acquisition_shape == pytest.approx(
        25.0 / (500.0 / 3.0 / 25.0), rel=1.0e-12, abs=0.0
    )
    # Poisson intensity: 3 outages over 300 s of observed window is 0.01 /s.
    assert model.outage_intensity_per_s == pytest.approx(0.01, rel=1.0e-14, abs=0.0)
    assert model.outage_mean_s == pytest.approx(12.0, rel=1.0e-14, abs=0.0)
    assert model.sample_passes == 4


def test_fit_event_model_rejects_an_empty_sample():
    """Input validation: there is nothing to fit."""
    with pytest.raises(ValueError, match="non-empty"):
        fit_event_model([], [], [], [], [])


def test_fit_event_model_rejects_misaligned_counts_and_windows():
    """Input validation: the Poisson estimator needs a window per count."""
    with pytest.raises(ValueError, match="must align"):
        fit_event_model([10.0], [1, 2], [100.0], [10.0], [5.0])


def test_fit_event_model_rejects_a_non_positive_window():
    """Input validation: the Poisson intensity divides by total window time."""
    with pytest.raises(ValueError, match="window_durations_s"):
        fit_event_model([10.0], [1], [0.0], [10.0], [5.0])


@pytest.mark.parametrize(
    ("shape", "scale", "intensity", "outage", "idle_shape", "idle_scale", "passes"),
    [
        (0.0, 10.0, 0.004, 14.0, 3.0, 9.0, 10),
        (4.0, 0.0, 0.004, 14.0, 3.0, 9.0, 10),
        (4.0, 10.0, -0.1, 14.0, 3.0, 9.0, 10),
        (4.0, 10.0, 0.004, 0.0, 3.0, 9.0, 10),
        (4.0, 10.0, 0.004, 14.0, 0.0, 9.0, 10),
        (4.0, 10.0, 0.004, 14.0, 3.0, 0.0, 10),
        (4.0, 10.0, 0.004, 14.0, 3.0, 9.0, 0),
    ],
)
def test_event_model_rejects_out_of_range_parameters(
    shape, scale, intensity, outage, idle_shape, idle_scale, passes
):
    """Input validation on every model parameter."""
    with pytest.raises(ValueError):
        EventModel(shape, scale, intensity, outage, idle_shape, idle_scale, passes)


# ---------------------------------------------------------------- baseline 3


def _decompositions(satrec, record, count: int):
    """Decompose the first ``count`` rows of the committed record."""
    from passyield.dataset import events_of, station_by_name, window_of
    from passyield.decompose import decompose_pass as run

    return [
        run(satrec, station_by_name(row.station), window_of(row), events_of(row))
        for row in record[:count]
    ]


def test_baseline_three_declares_the_pass_as_the_resampling_unit(satrec, record):
    """The specification fixes the unit as the PASS; it is on the object.

    Asserted rather than trusted to a docstring. Resampling seconds or
    outages would treat within-pass structure as independent replication and
    shrink every interval, which is how a bootstrap interval becomes a
    fiction.
    """
    result = accounted_yield(_decompositions(satrec, record, 40), seed=99099, resamples=200)
    assert result.resampling_unit == "pass"
    assert result.empirical_distribution.resampling_unit == "pass"
    assert result.passes == 40


def test_baseline_three_interval_contains_the_point_estimate(satrec, record):
    """A percentile bootstrap interval brackets the statistic it is built from."""
    result = accounted_yield(_decompositions(satrec, record, 60), seed=99099, resamples=400)
    low, high = result.mean_shortfall_ci_mbit
    assert low <= result.mean_shortfall_mbit <= high
    low, high = result.realised_fraction_ci
    assert low <= result.realised_fraction <= high


def test_baseline_three_shares_total_one_up_to_the_residual(satrec, record):
    """The four shares plus the residual share total 1.

    The residual share is reported as its own number, so a reader can see how
    much of the shortfall the identity did not close rather than inferring it
    from a rounding.
    """
    result = accounted_yield(_decompositions(satrec, record, 60), seed=99099, resamples=200)
    total = sum(result.shortfall_share.values()) + result.residual_share
    assert total == pytest.approx(1.0, rel=1.0e-9, abs=1.0e-12)


def test_baseline_three_is_reproducible_at_a_declared_seed(satrec, record):
    """Two bootstraps at the same seed give identical intervals."""
    first = accounted_yield(_decompositions(satrec, record, 40), seed=99099, resamples=200)
    second = accounted_yield(_decompositions(satrec, record, 40), seed=99099, resamples=200)
    assert first.mean_shortfall_ci_mbit == second.mean_shortfall_ci_mbit


@pytest.mark.parametrize(
    ("resamples", "confidence"), [(200, 0.0), (200, 1.0), (10, 0.95), (200, -0.5)]
)
def test_baseline_three_rejects_out_of_range_settings(satrec, record, resamples, confidence):
    """Input validation on the bootstrap."""
    with pytest.raises(ValueError):
        accounted_yield(
            _decompositions(satrec, record, 20), seed=1, resamples=resamples,
            confidence=confidence,
        )


def test_baseline_three_rejects_a_single_pass(satrec, record):
    """A pass-level bootstrap of one pass has nothing to resample."""
    with pytest.raises(ValueError, match="at least 2 passes"):
        accounted_yield(_decompositions(satrec, record, 1), seed=1)


@given(level=st.floats(min_value=0.6, max_value=0.99))
@settings(max_examples=12, deadline=None)
def test_wider_confidence_gives_a_wider_interval(satrec, record, level):
    """Property: a 99 per cent interval is at least as wide as the requested one."""
    narrow = accounted_yield(
        _decompositions(satrec, record, 40), seed=99099, resamples=400, confidence=level
    )
    wide = accounted_yield(
        _decompositions(satrec, record, 40), seed=99099, resamples=400, confidence=0.995
    )
    narrow_width = narrow.mean_shortfall_ci_mbit[1] - narrow.mean_shortfall_ci_mbit[0]
    wide_width = wide.mean_shortfall_ci_mbit[1] - wide.mean_shortfall_ci_mbit[0]
    assert wide_width >= narrow_width - 1.0e-9
