"""The declared synthetic conjunction population."""

from __future__ import annotations

import numpy as np
import pytest

from conjprob.analytic import pc_foster
from conjprob.dataset import DEFAULT_PC_THRESHOLD, REGIMES, generate_population
from conjprob.features import FEATURE_NAMES


def test_population_is_reproducible_for_a_seed():
    first = generate_population(300, seed=17)
    second = generate_population(300, seed=17)
    np.testing.assert_allclose(first.pc, second.pc, rtol=1e-15, atol=1e-300)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_different_seeds_give_different_populations():
    first = generate_population(300, seed=1)
    second = generate_population(300, seed=2)
    assert not np.allclose(first.pc, second.pc)


def test_shapes_and_label_definition():
    population = generate_population(200, seed=3)
    assert len(population) == 200
    assert population.features.shape == (200, len(FEATURE_NAMES))
    np.testing.assert_array_equal(population.labels, population.pc > population.threshold)
    assert population.threshold == DEFAULT_PC_THRESHOLD


def test_pc_is_the_foster_value_for_the_recorded_geometry():
    population = generate_population(40, seed=4)
    recomputed = np.array(
        [
            pc_foster(float(sx), float(sy), float(xb), float(yb), float(r))
            for sx, sy, xb, yb, r in zip(
                population.sigma_x_m,
                population.sigma_y_m,
                population.xbar_m,
                population.ybar_m,
                population.radius_m,
                strict=True,
            )
        ]
    )
    np.testing.assert_allclose(recomputed, population.pc, rtol=1e-12, atol=1e-300)


def test_sigma_x_is_the_larger_principal_sigma():
    population = generate_population(500, seed=5)
    assert np.all(population.sigma_x_m >= population.sigma_y_m)


def test_mahalanobis_feature_matches_the_recorded_geometry():
    population = generate_population(200, seed=6)
    index = FEATURE_NAMES.index("mahalanobis")
    expected = np.hypot(
        population.xbar_m / population.sigma_x_m, population.ybar_m / population.sigma_y_m
    )
    np.testing.assert_allclose(population.features[:, index], expected, rtol=1e-12)


@pytest.mark.parametrize("regime", REGIMES)
def test_every_declared_regime_generates(regime):
    population = generate_population(400, seed=7, regime=regime)
    assert population.regime == regime
    assert np.all(np.isfinite(population.features))
    assert 0.0 <= population.positive_rate <= 1.0


def test_dilution_regime_is_mostly_on_the_dilution_branch():
    population = generate_population(2000, seed=8, regime="dilution")
    index = FEATURE_NAMES.index("mahalanobis")
    fraction = float(np.mean(population.features[:, index] < np.sqrt(2.0)))
    assert fraction > 0.5


def test_homogeneous_regime_has_a_narrow_radius_over_sigma_range():
    population = generate_population(2000, seed=9, regime="homogeneous")
    index = FEATURE_NAMES.index("log10_radius_over_sigma_eff")
    spread = population.features[:, index].max() - population.features[:, index].min()
    assert spread < np.log10(3.0)


def test_nominal_regime_has_a_wide_radius_over_sigma_range():
    population = generate_population(2000, seed=10)
    index = FEATURE_NAMES.index("log10_radius_over_sigma_eff")
    spread = population.features[:, index].max() - population.features[:, index].min()
    assert spread > 2.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_samples": 0}, "n_samples"),
        ({"n_samples": 10, "regime": "nope"}, "regime"),
        ({"n_samples": 10, "threshold": 0.0}, "threshold"),
        ({"n_samples": 10, "threshold": 1.0}, "threshold"),
    ],
)
def test_rejects_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        generate_population(**kwargs)
