"""Fade-duration distribution fitting and goodness-of-fit tests."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from linkoutage.distributions import (
    compare_fade_duration_models,
    exponential_mle_with_censoring,
    fit_geometric_samples,
    geometric_chi_square,
    kaplan_meier_survival,
    ks_fitted,
)
from linkoutage.fade import fade_durations


def test_exponential_mle_hand_calculation():
    # complete = [1, 2, 3] (sum 6), censored = [4]; n_c = 3
    # theta = (6 + 4) / 3 = 10/3
    # log L = -3 ln(10/3) - 10 / (10/3) = -3 ln(10/3) - 3
    theta, ll = exponential_mle_with_censoring([1.0, 2.0, 3.0], [4.0])
    assert theta == pytest.approx(10.0 / 3.0)
    assert ll == pytest.approx(-3.0 * math.log(10.0 / 3.0) - 3.0)


def test_exponential_mle_without_censoring_is_the_sample_mean():
    t = [0.5, 1.5, 2.5, 3.5]
    theta, _ = exponential_mle_with_censoring(t)
    assert theta == pytest.approx(float(np.mean(t)))


def test_censoring_raises_the_fitted_scale():
    without, _ = exponential_mle_with_censoring([1.0, 1.0, 1.0])
    with_cens, _ = exponential_mle_with_censoring([1.0, 1.0, 1.0], [5.0])
    assert with_cens > without


def test_exponential_mle_recovers_the_scale():
    rng = np.random.default_rng(4)
    t = rng.exponential(2.5, 20_000)
    theta, _ = exponential_mle_with_censoring(t)
    assert theta == pytest.approx(2.5, rel=0.03)


def test_geometric_mle_hand_calculation():
    # mean of [1, 2, 3, 4] is 2.5 -> p = 1 / 2.5 = 0.4
    p, _ = fit_geometric_samples([1, 2, 3, 4])
    assert p == pytest.approx(0.4)


def test_geometric_mle_log_likelihood_hand_calculation():
    # L = p * (1-p) p = p^2 (1-p) for lengths [1, 2] with p = 1/1.5 = 2/3
    p, ll = fit_geometric_samples([1, 2])
    assert p == pytest.approx(2.0 / 3.0)
    assert ll == pytest.approx(2.0 * math.log(2.0 / 3.0) + math.log(1.0 / 3.0))


def test_geometric_chi_square_does_not_reject_geometric_data():
    rng = np.random.default_rng(12)
    lengths = rng.geometric(0.08, 30_000)
    p, _ = fit_geometric_samples(lengths)
    stat, dof, pv, n_bins, note = geometric_chi_square(lengths, p)
    assert dof >= 5
    assert note == ""
    assert pv > 0.01, (stat, dof, pv, n_bins)


def test_geometric_chi_square_rejects_heavy_tailed_data():
    rng = np.random.default_rng(13)
    lengths = np.maximum(1, np.rint(rng.lognormal(2.0, 1.2, 30_000)).astype(int))
    p, _ = fit_geometric_samples(lengths)
    stat, dof, pv, _, _ = geometric_chi_square(lengths, p)
    assert pv < 1e-6
    assert stat > dof


def test_geometric_chi_square_degenerate_sample():
    stat, dof, pv, n_bins, note = geometric_chi_square([1, 1, 1, 1, 1], 1.0)
    assert n_bins == 1
    assert dof < 1
    assert math.isnan(pv)
    assert "degrees of freedom" in note


def test_geometric_chi_square_rejects_bad_p():
    with pytest.raises(ValueError, match="p must lie"):
        geometric_chi_square([1, 2, 3], 0.0)


def test_fit_geometric_rejects_non_integer():
    with pytest.raises(ValueError, match="integer"):
        fit_geometric_samples([1.5, 2.0])


def test_fit_geometric_rejects_zero_length():
    with pytest.raises(ValueError, match=">= 1"):
        fit_geometric_samples([0, 1, 2])


def test_ks_fitted_small_on_correct_model():
    rng = np.random.default_rng(14)
    t = rng.exponential(1.3, 20_000)
    stat, pv = ks_fitted(t, stats.expon, (0.0, 1.3))
    assert stat < 0.02
    assert pv > 0.01


def test_ks_fitted_large_on_wrong_model():
    rng = np.random.default_rng(15)
    t = rng.lognormal(0.0, 1.0, 5_000)
    stat, _ = ks_fitted(t, stats.expon, (0.0, 1.0))
    assert stat > 0.1


def test_kaplan_meier_without_censoring_is_one_minus_ecdf():
    t = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    times, surv = kaplan_meier_survival(t)
    assert times.tolist() == [1.0, 2.0, 3.0, 5.0]
    expected = [1.0 - np.mean(t <= tt) for tt in times]
    assert np.allclose(surv, expected)


def test_kaplan_meier_with_censoring_lies_above_naive():
    t = np.array([1.0, 2.0, 3.0])
    _, surv_plain = kaplan_meier_survival(t)
    _, surv_cens = kaplan_meier_survival(t, [10.0, 10.0])
    assert np.all(surv_cens >= surv_plain - 1e-12)
    assert surv_cens[-1] > surv_plain[-1]


def test_kaplan_meier_ends_at_zero_without_censoring():
    _, surv = kaplan_meier_survival([1.0, 2.0, 3.0])
    assert surv[-1] == pytest.approx(0.0)


def test_compare_models_rejects_exponential_on_the_lognormal_channel(short_channel):
    complete, censored = fade_durations(short_channel.amplitude, 0.6, 1.0e6)
    result = compare_fade_duration_models(complete, fs_hz=1.0e6, censored_s=censored)
    assert result.exponential_rejected is True
    assert result.geometric.valid is True
    assert result.geometric.p_value < 1e-10
    # A correlated channel's fade durations are over-dispersed relative to the
    # exponential, whose coefficient of variation is exactly 1.
    assert result.extra["coefficient_of_variation"] > 1.5


def test_compare_models_does_not_reject_true_geometric_durations():
    rng = np.random.default_rng(16)
    lengths = rng.geometric(0.05, 20_000)
    result = compare_fade_duration_models(lengths / 1.0e6, fs_hz=1.0e6)
    assert result.exponential_rejected is False


def test_compare_models_report_mentions_the_verdict(short_channel):
    complete, _ = fade_durations(short_channel.amplitude, 0.6, 1.0e6)
    text = compare_fade_duration_models(complete, fs_hz=1.0e6).report()
    assert "Memoryless" in text
    assert "geometric chi-square" in text


def test_compare_models_names_all_five_laws(short_channel):
    complete, _ = fade_durations(short_channel.amplitude, 0.6, 1.0e6)
    result = compare_fade_duration_models(complete, fs_hz=1.0e6)
    names = [f.name for f in result.fits]
    assert names[0] == "geometric (samples)"
    assert "exponential (cens MLE)" in names
    assert {"lognormal", "weibull", "gamma"} <= set(names)


def test_best_by_aic_is_one_of_the_fits(short_channel):
    complete, _ = fade_durations(short_channel.amplitude, 0.6, 1.0e6)
    result = compare_fade_duration_models(complete, fs_hz=1.0e6)
    assert result.best_by_aic in result.fits


def test_every_fit_carries_a_note_or_a_valid_flag(short_channel):
    complete, _ = fade_durations(short_channel.amplitude, 0.6, 1.0e6)
    result = compare_fade_duration_models(complete, fs_hz=1.0e6)
    for fit in result.fits:
        assert fit.notes or not fit.valid
        assert fit.line()


def test_interval_count_durations_are_rejected_by_the_comparison():
    from linkoutage.fade import FadeDefinitions

    a = np.array([1.0, 0.4, 1.0, 0.4, 1.0, 0.4, 1.0] * 5)
    complete, _ = fade_durations(
        a, 0.6, 1.0, definitions=FadeDefinitions(duration_convention="interval_count")
    )
    with pytest.raises(ValueError, match="interval_count"):
        compare_fade_duration_models(complete, fs_hz=1.0)


def test_empty_durations_raise():
    with pytest.raises(ValueError, match="empty"):
        compare_fade_duration_models([], fs_hz=1.0e6)


def test_negative_durations_raise():
    with pytest.raises(ValueError, match="negative"):
        compare_fade_duration_models([-1.0, 1.0], fs_hz=1.0e6)


def test_bad_fs_raises():
    with pytest.raises(ValueError, match="fs_hz"):
        compare_fade_duration_models([1e-6], fs_hz=0.0)
