"""Uncertainty budget: GUM terms, bootstrap, and combination."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinfer.uncertainty import (
    bootstrap_quantile_uncertainty,
    clock_quantisation_uncertainty_s,
    combine_standard_uncertainties,
    mean_uncertainty,
    timer_overhead_s,
    uncertainty_budget,
)


class TestClockQuantisation:
    def test_known_answer_for_one_read(self) -> None:
        # GUM 4.3.7: a rectangular distribution of full width d has standard
        # uncertainty d/sqrt(12). For d = 1 ns: 1e-9/3.4641 = 2.8868e-10 s.
        assert clock_quantisation_uncertainty_s(1e-9, reads=1) == pytest.approx(
            1e-9 / np.sqrt(12.0)
        )

    def test_two_reads_combine_in_quadrature(self) -> None:
        one = clock_quantisation_uncertainty_s(1e-9, reads=1)
        two = clock_quantisation_uncertainty_s(1e-9, reads=2)
        assert two == pytest.approx(one * np.sqrt(2.0))

    def test_a_perfect_clock_contributes_nothing(self) -> None:
        assert clock_quantisation_uncertainty_s(0.0) == 0.0

    def test_negative_resolution_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            clock_quantisation_uncertainty_s(-1e-9)

    def test_zero_reads_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="reads must be >= 1"):
            clock_quantisation_uncertainty_s(1e-9, reads=0)


class TestMeanUncertainty:
    def test_known_answer(self) -> None:
        # samples 1..5: s = sqrt(2.5) = 1.58114, n = 5, s/sqrt(5) = 0.70711.
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mean_uncertainty(samples) == pytest.approx(0.7071067811865476)

    def test_scales_as_one_over_root_n(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(0.0, 1.0, size=100)
        wide = np.tile(base, 4)
        assert mean_uncertainty(wide) == pytest.approx(mean_uncertainty(base) / 2.0, rel=0.1)

    def test_single_sample_gives_nan(self) -> None:
        assert np.isnan(mean_uncertainty(np.array([1.0])))


class TestBootstrap:
    def test_is_deterministic_in_the_seed(self) -> None:
        rng = np.random.default_rng(2)
        samples = rng.lognormal(0.0, 0.5, size=200)
        a = bootstrap_quantile_uncertainty(samples, 0.99, 200, seed=7)
        b = bootstrap_quantile_uncertainty(samples, 0.99, 200, seed=7)
        c = bootstrap_quantile_uncertainty(samples, 0.99, 200, seed=8)
        assert a == b
        assert a != c

    def test_uncertainty_of_the_median_shrinks_with_sample_size(self) -> None:
        rng = np.random.default_rng(3)
        small = bootstrap_quantile_uncertainty(rng.normal(size=50), 0.5, 400, seed=0)
        rng = np.random.default_rng(3)
        large = bootstrap_quantile_uncertainty(rng.normal(size=2000), 0.5, 400, seed=0)
        assert large < small

    def test_tail_quantile_is_less_certain_than_the_median(self) -> None:
        rng = np.random.default_rng(4)
        samples = rng.lognormal(0.0, 1.0, size=400)
        median = bootstrap_quantile_uncertainty(samples, 0.5, 400, seed=0)
        tail = bootstrap_quantile_uncertainty(samples, 0.99, 400, seed=0)
        assert tail > median

    def test_single_sample_gives_nan(self) -> None:
        assert np.isnan(bootstrap_quantile_uncertainty(np.array([1.0]), 0.5))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"quantile": 0.0}, "quantile"), ({"quantile": 1.0}, "quantile"),
         ({"n_resamples": 5}, "n_resamples")],
    )
    def test_invalid_arguments_are_rejected(self, kwargs: dict, match: str) -> None:
        base = {"samples": np.arange(10.0), "quantile": 0.5, "n_resamples": 100}
        base.update(kwargs)
        with pytest.raises(ValueError, match=match):
            bootstrap_quantile_uncertainty(**base)


class TestCombination:
    def test_root_sum_square(self) -> None:
        # 3 and 4 combine to 5.
        assert combine_standard_uncertainties(3.0, 4.0) == pytest.approx(5.0)

    def test_no_components_gives_zero(self) -> None:
        assert combine_standard_uncertainties() == 0.0

    def test_negative_component_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            combine_standard_uncertainties(1.0, -1.0)


class TestTimerOverhead:
    def test_is_small_and_non_negative(self) -> None:
        bias = timer_overhead_s(500)
        assert 0.0 <= bias < 1e-5

    def test_too_few_samples_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n must be >= 10"):
            timer_overhead_s(2)


class TestUncertaintyBudget:
    def test_mean_budget_combines_type_a_and_clock(self) -> None:
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e-6
        budget = uncertainty_budget(samples, "mean", 1e-9, method="test")
        type_a = mean_uncertainty(samples)
        clock = clock_quantisation_uncertainty_s(1e-9)
        assert budget.value == pytest.approx(3e-6)
        assert budget.type_a_s == pytest.approx(type_a)
        assert budget.clock_s == pytest.approx(clock)
        assert budget.combined_s == pytest.approx(np.hypot(type_a, clock))
        assert budget.n_samples == 5

    def test_quantile_budget_records_the_bootstrap_settings(self) -> None:
        rng = np.random.default_rng(5)
        samples = rng.lognormal(0.0, 0.3, size=300) * 1e-6
        budget = uncertainty_budget(
            samples, "p99", 1e-9, quantile=0.99, n_resamples=100, seed=3, method="test"
        )
        assert budget.value == pytest.approx(float(np.quantile(samples, 0.99)))
        assert "seed=3" in budget.method
        assert "100 resamples" in budget.method

    def test_a_single_sample_does_not_produce_a_nan_combined_uncertainty(self) -> None:
        budget = uncertainty_budget(np.array([1e-6]), "mean", 1e-9)
        assert np.isnan(budget.type_a_s)
        assert np.isfinite(budget.combined_s)

    def test_empty_samples_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            uncertainty_budget(np.array([]), "mean", 1e-9)

    def test_summary_reports_the_bias_as_stated_not_corrected(self) -> None:
        text = "\n".join(
            uncertainty_budget(
                np.arange(1.0, 11.0) * 1e-6, "mean", 1e-9, timer_bias_s=4e-8
            ).summary_lines()
        )
        assert "stated, not corrected" in text
