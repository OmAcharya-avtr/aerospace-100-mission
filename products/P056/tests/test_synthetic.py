"""Synthetic generator: determinism, registry, and the shape of the draws."""

from __future__ import annotations

import numpy as np
import pytest

from calibaudit.synthetic import (
    SPEC_NAMES,
    analytic_truth,
    get_spec,
    sample_forecast,
    spec_names,
)


class TestRegistry:
    def test_registry_is_not_empty(self):
        assert len(SPEC_NAMES) == 6

    def test_spec_names_returns_a_mutable_copy(self):
        names = spec_names()
        names.append("intruder")
        assert "intruder" not in spec_names()

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_every_spec_resolves_and_describes_itself(self, name):
        spec = get_spec(name)
        assert spec.name == name
        assert len(spec.description) > 10

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_every_spec_has_a_finite_analytic_truth(self, name):
        t = analytic_truth(get_spec(name))
        for value in (t.base_rate, t.uncertainty, t.resolution, t.reliability, t.brier,
                      t.ece, t.log_score):
            assert np.isfinite(value)

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_brier_equals_the_sum_of_its_terms(self, name):
        t = analytic_truth(get_spec(name))
        assert t.brier == pytest.approx(t.brier_from_terms, abs=1e-15)


class TestDeterminism:
    def test_same_seed_gives_the_same_draw(self):
        a = sample_forecast(get_spec("calibrated"), 500, seed=11)
        b = sample_forecast(get_spec("calibrated"), 500, seed=11)
        assert np.array_equal(a.forecasts, b.forecasts)
        assert np.array_equal(a.outcomes, b.outcomes)
        assert np.array_equal(a.true_probabilities, b.true_probabilities)

    def test_different_seeds_give_different_draws(self):
        a = sample_forecast(get_spec("calibrated"), 500, seed=11)
        b = sample_forecast(get_spec("calibrated"), 500, seed=12)
        assert not np.array_equal(a.forecasts, b.forecasts)

    def test_a_longer_sample_is_not_an_extension_of_a_shorter_one(self):
        # Measured consequence of drawing each array in one call from one
        # stream: the latent probabilities do share a prefix, because they are
        # drawn first, but the outcomes do not, because the outcome draw
        # starts further along the stream. So a size change invalidates every
        # number, and sample sizes are part of the seed in this package.
        a = sample_forecast(get_spec("calibrated"), 100, seed=13)
        b = sample_forecast(get_spec("calibrated"), 200, seed=13)
        assert np.array_equal(a.forecasts, b.forecasts[:100])
        assert not np.array_equal(a.outcomes, b.outcomes[:100])


class TestDrawShape:
    @pytest.mark.parametrize("n", [1, 2, 17, 1000])
    def test_shapes_match_the_request(self, n):
        s = sample_forecast(get_spec("overconfident"), n, seed=14)
        assert s.forecasts.shape == (n,)
        assert s.outcomes.shape == (n,)
        assert s.true_probabilities.shape == (n,)
        assert s.n_samples == n

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_forecasts_are_probabilities(self, name):
        s = sample_forecast(get_spec(name), 2000, seed=15)
        assert np.all((s.forecasts >= 0.0) & (s.forecasts <= 1.0))

    @pytest.mark.parametrize("name", SPEC_NAMES)
    def test_outcomes_are_binary(self, name):
        s = sample_forecast(get_spec(name), 2000, seed=16)
        assert set(np.unique(s.outcomes).tolist()) <= {0.0, 1.0}

    def test_identity_spec_forecasts_equal_the_latent_probability(self):
        s = sample_forecast(get_spec("calibrated"), 500, seed=17)
        assert np.array_equal(s.forecasts, s.true_probabilities)

    def test_base_rate_of_the_rare_spec_is_near_one_tenth(self):
        s = sample_forecast(get_spec("calibrated_rare"), 40000, seed=18)
        # sem of a mean of Bernoulli(0.1) over 40000 draws is 1.5e-3; 5 sem.
        assert float(s.outcomes.mean()) == pytest.approx(0.1, abs=5 * 1.5e-3)

    def test_overconfident_spec_is_sharper_than_its_latent_probability(self):
        s = sample_forecast(get_spec("overconfident"), 20000, seed=19)
        assert float(s.forecasts.std()) > float(s.true_probabilities.std())

    def test_underconfident_spec_is_flatter_than_its_latent_probability(self):
        s = sample_forecast(get_spec("underconfident"), 20000, seed=20)
        assert float(s.forecasts.std()) < float(s.true_probabilities.std())
