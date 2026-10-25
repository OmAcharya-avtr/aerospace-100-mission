"""Known-answer and precision tests for the hazard/unreliability arithmetic."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from reliablock.numerics import (
    MAX_HAZARD,
    complement,
    hazard_from_reliability,
    hazard_from_unreliability,
    log1p_safe,
    parallel_unreliability,
    reliability_from_hazard,
    series_hazard,
    series_unreliability,
    unreliability_from_hazard,
    weighted_binomial_survival,
)

from conftest import abs_, rel


class TestConversions:
    def test_reliability_from_hazard_hand_calculated(self) -> None:
        # H = 0.1 -> R = exp(-0.1) = 0.9048374180359595
        assert reliability_from_hazard(0.1) == rel(0.9048374180359595, 1e-15)

    def test_unreliability_from_hazard_hand_calculated(self) -> None:
        # H = 0.1 -> Q = 1 - exp(-0.1) = 0.09516258196404048
        assert unreliability_from_hazard(0.1) == rel(0.09516258196404048, 1e-15)

    def test_round_trip_through_unreliability_keeps_tiny_hazards(self) -> None:
        # The point of the module: a hazard of 1e-18 survives the round trip
        # through unreliability space, and is destroyed by reliability space.
        hazard = 1.0e-18
        q = unreliability_from_hazard(hazard)
        assert q == rel(hazard, 1e-15)
        assert hazard_from_unreliability(q) == rel(hazard, 1e-15)
        # Via reliability: R is exactly 1.0 and the hazard is lost entirely.
        assert reliability_from_hazard(hazard) == 1.0
        assert hazard_from_reliability(reliability_from_hazard(hazard)) == 0.0

    def test_hazard_above_cap_saturates(self) -> None:
        assert reliability_from_hazard(MAX_HAZARD + 1.0) == 0.0
        assert unreliability_from_hazard(MAX_HAZARD + 1.0) == 1.0

    def test_infinite_hazard_is_accepted(self) -> None:
        assert reliability_from_hazard(math.inf) == 0.0
        assert unreliability_from_hazard(math.inf) == 1.0

    def test_hazard_from_unreliability_at_one_is_infinite(self) -> None:
        assert hazard_from_unreliability(1.0) == math.inf

    def test_log1p_safe_at_minus_one(self) -> None:
        assert log1p_safe(-1.0) == -math.inf

    def test_complement_is_exact_on_the_upper_half(self) -> None:
        for p in (0.5, 0.75, 0.9, 1.0):
            assert complement(p) + p == 1.0


class TestSeries:
    def test_series_hazard_is_the_exact_sum(self) -> None:
        assert series_hazard([0.1, 0.2, 0.3]) == rel(0.6000000000000001, 1e-16)

    def test_series_hazard_of_nothing_is_zero(self) -> None:
        assert series_hazard([]) == 0.0

    def test_series_hazard_with_an_infinite_term(self) -> None:
        assert series_hazard([0.1, math.inf, 0.2]) == math.inf

    def test_series_unreliability_beats_the_naive_expression(self) -> None:
        # Ten blocks at q = 1e-12. The exact value, from the first two
        # Bonferroni orders, lies in [1e-11 - 45e-24, 1e-11].
        q = 1.0e-12
        n = 10
        computed = series_unreliability([q] * n)
        naive = 1.0 - (1.0 - q) ** n
        exact_upper = n * q
        exact_lower = n * q - (n * (n - 1) / 2) * q * q
        assert exact_lower <= computed <= exact_upper
        # The naive expression lands outside the bound, which is the finding.
        assert not exact_lower <= naive <= exact_upper
        assert abs(naive - computed) / computed > 1.0e-6

    def test_series_unreliability_known_answer(self) -> None:
        # q1 = 0.1, q2 = 0.2 -> Q = 1 - 0.9*0.8 = 0.28
        assert series_unreliability([0.1, 0.2]) == rel(0.28, 1e-14)


class TestParallel:
    def test_parallel_unreliability_is_the_exact_product(self) -> None:
        assert parallel_unreliability([1.0e-9, 1.0e-9]) == 1.0e-18

    def test_parallel_unreliability_survives_where_reliability_space_cannot(self) -> None:
        # 1 - prod(1 - q_i) is exactly 0.0 here; the product is not.
        q = 1.0e-9
        assert parallel_unreliability([q, q]) == 1.0e-18
        assert 1.0 - (1.0 - q * q) == 0.0

    def test_parallel_unreliability_of_nothing_is_zero(self) -> None:
        assert parallel_unreliability([]) == 0.0

    def test_parallel_unreliability_short_circuits_on_zero(self) -> None:
        assert parallel_unreliability([0.0, 0.5, 0.5]) == 0.0

    def test_parallel_unreliability_known_answer(self) -> None:
        assert parallel_unreliability([0.1, 0.2, 0.3]) == rel(0.006, 1e-15)


class TestPoissonBinomial:
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    def test_identical_blocks_match_the_binomial_survival(self, k: int) -> None:
        p = 0.9
        n = 5
        assert weighted_binomial_survival([p] * n, k) == rel(
            float(binom.sf(k - 1, n, p)), 1e-13
        )

    def test_hand_calculated_two_of_three(self) -> None:
        # p = 0.9: R = 3 p^2 (1-p) + p^3 = 3*0.81*0.1 + 0.729 = 0.243 + 0.729
        assert weighted_binomial_survival([0.9] * 3, 2) == rel(0.972, 1e-14)

    def test_non_identical_against_brute_force(self) -> None:
        from itertools import product

        probabilities = [0.9, 0.8, 0.7, 0.6]
        for k in range(len(probabilities) + 2):
            brute = 0.0
            for state in product((0, 1), repeat=len(probabilities)):
                if sum(state) < k:
                    continue
                term = 1.0
                for up, p in zip(state, probabilities, strict=True):
                    term *= p if up else 1.0 - p
                brute += term
            assert weighted_binomial_survival(probabilities, k) == abs_(brute, 1e-14)

    def test_k_zero_is_certain_success_and_k_above_n_certain_failure(self) -> None:
        assert weighted_binomial_survival([0.5, 0.5], 0) == 1.0
        assert weighted_binomial_survival([0.5, 0.5], 3) == 0.0

    def test_distribution_sums_to_one(self) -> None:
        probabilities = [0.3, 0.5, 0.95, 0.01, 0.7]
        total = weighted_binomial_survival(probabilities, 0)
        assert total == 1.0
        # survival at k minus survival at k+1 is P(exactly k), all non-negative
        for k in range(len(probabilities) + 1):
            mass = weighted_binomial_survival(probabilities, k) - weighted_binomial_survival(
                probabilities, k + 1
            )
            assert mass >= -1e-18

    def test_numpy_integer_k_is_accepted(self) -> None:
        assert weighted_binomial_survival([0.9, 0.9], np.int64(1)) == rel(0.99, 1e-14)


class TestValidation:
    @pytest.mark.parametrize("bad", [-0.1, 1.1, 2.0])
    def test_probability_out_of_range_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            complement(bad)

    def test_negative_hazard_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            reliability_from_hazard(-1.0)

    def test_nan_hazard_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="nan"):
            reliability_from_hazard(float("nan"))

    def test_boolean_k_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="integer"):
            weighted_binomial_survival([0.9], True)

    def test_negative_k_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            weighted_binomial_survival([0.9], -1)

    def test_log1p_below_minus_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            log1p_safe(-1.5)
