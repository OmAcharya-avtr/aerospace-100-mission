"""Unit and edge-case tests for the synthetic generator and its known truth."""

from __future__ import annotations

import numpy as np
import pytest

from transfergap import synthetic


def test_generate_is_deterministic_from_its_seed():
    first = synthetic.generate(40, "random", seed=7)
    second = synthetic.generate(40, "random", seed=7)
    assert np.array_equal(first.gaps, second.gaps)
    assert np.array_equal(first.features, second.features)


def test_generate_differs_between_seeds():
    first = synthetic.generate(40, "random", seed=7)
    second = synthetic.generate(40, "random", seed=8)
    assert not np.array_equal(first.gaps, second.gaps)


def test_generate_respects_the_declared_range():
    sample = synthetic.generate(2000, "random", seed=11)
    assert sample.gaps.min() >= 0.0
    assert sample.gaps.max() <= synthetic.BOUND_RANGE_KNM


def test_generate_carries_the_unit_and_names():
    sample = synthetic.generate(20, "random", seed=1)
    assert sample.unit == "kN.m"
    assert sample.feature_names == ("mach", "altitude_km", "alpha_deg", "load_factor")


def test_features_stay_inside_the_declared_envelope():
    sample = synthetic.generate(5000, "random", seed=13)
    for column, (low, high) in enumerate(synthetic.ENVELOPE):
        assert sample.features[:, column].min() >= low
        assert sample.features[:, column].max() <= high


def test_swept_campaign_has_a_monotone_mach_column():
    sample = synthetic.generate(200, "swept", seed=3)
    mach = sample.features[:, 0]
    assert np.all(np.diff(mach) >= 0.0)


def test_random_campaign_mach_column_is_not_monotone():
    sample = synthetic.generate(200, "random", seed=3)
    assert not np.all(np.diff(sample.features[:, 0]) >= 0.0)


def test_generate_rejects_an_unknown_campaign():
    with pytest.raises(ValueError, match="campaign must be one of"):
        synthetic.generate(20, "spiral", seed=1)


def test_generate_rejects_n_below_one():
    with pytest.raises(ValueError, match="at least 1"):
        synthetic.generate(0, "random", seed=1)


def test_conditional_mean_gap_matches_a_direct_monte_carlo():
    """Equation (22), the folded-normal mean, against brute force.

    The closed form is checked against 200000 draws of |b + s Z| at a fixed
    condition. The tolerance is four Monte Carlo standard errors of the
    brute-force mean, which is the conditioning of the comparison and not an
    absolute constant.
    """
    rng = np.random.default_rng(21)
    condition = np.array([[0.88, 9.0, 5.0, 2.0]])
    closed = float(synthetic.conditional_mean_gap(condition)[0])
    bias = float(synthetic._bias(condition)[0])
    scale = float(synthetic._scale(condition)[0])
    draws = np.abs(bias + scale * rng.standard_normal(200_000))
    standard_error = float(np.std(draws, ddof=1) / np.sqrt(draws.size))
    assert closed == pytest.approx(
        float(np.mean(draws)), rel=0.0, abs=4.0 * standard_error
    )


def test_conditional_mean_gap_rejects_a_wrong_column_count():
    with pytest.raises(ValueError, match="shape"):
        synthetic.conditional_mean_gap(np.zeros((3, 2)))


def test_population_truth_is_deterministic_and_reports_its_error():
    first = synthetic.population_truth(draws=20_000, seed=99)
    second = synthetic.population_truth(draws=20_000, seed=99)
    assert first.mean == second.mean
    assert first.standard_error > 0.0
    assert first.clip_fraction == 0.0


def test_population_truth_quantiles_increase():
    truth = synthetic.population_truth(draws=50_000, seed=98)
    values = [truth.quantiles[p] for p in sorted(truth.quantiles)]
    assert all(later > earlier for earlier, later in zip(values[:-1], values[1:], strict=True))


def test_population_mean_is_below_the_declared_range():
    truth = synthetic.population_truth(draws=20_000, seed=97)
    assert 0.0 < truth.mean < synthetic.BOUND_RANGE_KNM
