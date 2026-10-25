"""Known-answer, round-trip and validation tests for the failure distributions."""

from __future__ import annotations

import math

import pytest
from scipy.integrate import quad
from scipy.stats import lognorm, weibull_min

from reliablock.distributions import (
    Exponential,
    FixedProbability,
    Lognormal,
    Weibull,
    distribution_from_mapping,
)

from conftest import rel


class TestExponential:
    def test_hand_calculated_reliability(self) -> None:
        # lambda = 1e-5 /h, t = 1000 h -> H = 0.01, R = exp(-0.01)
        d = Exponential(1.0e-5)
        assert d.cumulative_hazard(1000.0) == rel(0.01, 1e-15)
        assert d.reliability(1000.0) == rel(0.990049833749168, 1e-14)
        assert d.unreliability(1000.0) == rel(0.009950166250831947, 1e-14)

    def test_mttf_is_the_reciprocal_rate(self) -> None:
        assert Exponential(2.5e-4).mttf_hours() == rel(4000.0, 1e-14)

    def test_zero_rate_never_fails(self) -> None:
        d = Exponential(0.0)
        assert d.reliability(1.0e9) == 1.0
        assert d.unreliability(1.0e9) == 0.0
        assert d.mttf_hours() == math.inf
        assert d.inverse_cumulative_hazard(1.0) == math.inf

    def test_density_integrates_to_the_unreliability(self) -> None:
        d = Exponential(3.0e-4)
        value, _ = quad(d.density, 0.0, 5000.0)
        assert value == rel(d.unreliability(5000.0), 1e-9)

    def test_ages_with_time(self) -> None:
        assert Exponential(1e-5).ages_with_time is True


class TestWeibull:
    def test_hand_calculated_hazard(self) -> None:
        # beta = 2, eta = 5000 h, t = 1000 h -> H = (1/5)^2 = 0.04
        d = Weibull(shape=2.0, scale_hours=5000.0)
        assert d.cumulative_hazard(1000.0) == rel(0.04, 1e-14)
        assert d.reliability(1000.0) == rel(math.exp(-0.04), 1e-14)

    def test_shape_one_reduces_to_the_exponential(self) -> None:
        eta = 7000.0
        weibull = Weibull(shape=1.0, scale_hours=eta)
        exponential = Exponential(1.0 / eta)
        assert weibull.cumulative_hazard(0.0) == exponential.cumulative_hazard(0.0) == 0.0
        for t in (1.0, 100.0, 7000.0, 50_000.0):
            assert weibull.cumulative_hazard(t) == rel(
                exponential.cumulative_hazard(t), 1e-14
            )

    def test_matches_scipy_weibull_min(self) -> None:
        d = Weibull(shape=2.3, scale_hours=5.0e4)
        reference = weibull_min(2.3, scale=5.0e4)
        for t in (100.0, 5.0e3, 5.0e4, 2.0e5):
            assert d.reliability(t) == rel(float(reference.sf(t)), 1e-12)
            assert d.density(t) == rel(float(reference.pdf(t)), 1e-11)

    def test_inverse_cumulative_hazard_round_trips(self) -> None:
        d = Weibull(shape=1.7, scale_hours=1.0e4)
        for h in (1e-9, 1e-3, 1.0, 20.0):
            assert d.cumulative_hazard(d.inverse_cumulative_hazard(h)) == rel(h, 1e-12)

    def test_density_at_zero_for_increasing_hazard(self) -> None:
        assert Weibull(shape=2.0, scale_hours=100.0).density(0.0) == 0.0

    def test_density_at_zero_for_constant_hazard(self) -> None:
        assert Weibull(shape=1.0, scale_hours=100.0).density(0.0) == rel(0.01, 1e-15)

    def test_large_shape_does_not_overflow(self) -> None:
        d = Weibull(shape=60.0, scale_hours=100.0)
        assert d.cumulative_hazard(1.0e6) <= 745.0
        assert d.reliability(1.0e6) == 0.0


class TestLognormal:
    def test_matches_scipy_lognorm(self) -> None:
        d = Lognormal(log_median=10.0, log_sigma=0.6)
        reference = lognorm(0.6, scale=math.exp(10.0))
        for t in (100.0, 5.0e3, 2.2e4, 1.0e5):
            assert d.reliability(t) == rel(float(reference.sf(t)), 1e-11)

    def test_upper_tail_hazard_stays_finite(self) -> None:
        d = Lognormal(log_median=8.0, log_sigma=0.5)
        # 1 - F cancels to zero in the far tail; logsf does not.
        assert 0.0 < d.cumulative_hazard(1.0e9) < 745.0

    def test_inverse_cumulative_hazard_round_trips(self) -> None:
        d = Lognormal(log_median=9.0, log_sigma=0.8)
        for h in (1e-6, 1e-2, 1.0, 10.0):
            assert d.cumulative_hazard(d.inverse_cumulative_hazard(h)) == rel(h, 1e-9)

    def test_hazard_at_zero_is_zero(self) -> None:
        assert Lognormal(log_median=9.0, log_sigma=0.8).cumulative_hazard(0.0) == 0.0


class TestFixedProbability:
    def test_unreliability_is_independent_of_time(self) -> None:
        d = FixedProbability(1.5e-3)
        for t in (0.0, 1.0, 1.0e6):
            assert d.unreliability(t) == rel(1.5e-3, 1e-15)

    def test_does_not_age(self) -> None:
        assert FixedProbability(0.1).ages_with_time is False

    def test_density_is_zero(self) -> None:
        assert FixedProbability(0.1).density(10.0) == 0.0

    def test_certain_failure(self) -> None:
        d = FixedProbability(1.0)
        assert d.reliability(1.0) == 0.0
        assert d.unreliability(1.0) == 1.0


class TestDeclarationLoading:
    def test_round_trip_through_mapping(self) -> None:
        for d in (
            Exponential(1.0e-5),
            Weibull(shape=2.0, scale_hours=3000.0),
            Lognormal(log_median=9.0, log_sigma=0.4),
            FixedProbability(1.0e-3),
        ):
            assert distribution_from_mapping(d.to_mapping()) == d

    def test_unknown_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown distribution kind"):
            distribution_from_mapping({"kind": "gamma", "shape": 2.0})

    def test_missing_kind_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="needs a 'kind' key"):
            distribution_from_mapping({"rate_per_hour": 1.0})

    def test_misspelled_parameter_is_rejected_not_defaulted(self) -> None:
        # The defect this rejection exists for: a typo that fell through to a
        # default would report a reliability of 1.0 and no complaint.
        with pytest.raises(ValueError, match="does not accept"):
            distribution_from_mapping({"kind": "exponential", "rate_per_hr": 1.0e-5})

    def test_missing_parameter_is_rejected_not_defaulted(self) -> None:
        with pytest.raises(ValueError, match="requires"):
            distribution_from_mapping({"kind": "exponential"})

    def test_non_mapping_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a mapping"):
            distribution_from_mapping([1, 2, 3])  # type: ignore[arg-type]


class TestParameterValidation:
    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_bad_exponential_rate(self, bad: float) -> None:
        with pytest.raises(ValueError, match="rate_per_hour"):
            Exponential(bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_bad_weibull_shape(self, bad: float) -> None:
        with pytest.raises(ValueError, match="shape"):
            Weibull(shape=bad, scale_hours=100.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_bad_weibull_scale(self, bad: float) -> None:
        with pytest.raises(ValueError, match="scale_hours"):
            Weibull(shape=1.0, scale_hours=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_bad_lognormal_sigma(self, bad: float) -> None:
        with pytest.raises(ValueError, match="log_sigma"):
            Lognormal(log_median=1.0, log_sigma=bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_bad_fixed_probability(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            FixedProbability(bad)

    @pytest.mark.parametrize(
        "distribution",
        [
            Exponential(1e-5),
            Weibull(shape=2.0, scale_hours=100.0),
            Lognormal(log_median=5.0, log_sigma=1.0),
            FixedProbability(0.1),
        ],
    )
    def test_negative_time_is_rejected(self, distribution: object) -> None:
        with pytest.raises(ValueError, match="t_hours must be >= 0"):
            distribution.cumulative_hazard(-1.0)  # type: ignore[attr-defined]

    def test_positional_construction_sets_the_parameter_not_the_flag(self) -> None:
        # ages_with_time is a ClassVar, not a dataclass field, so the first
        # positional argument is the rate. Getting this wrong would make
        # Exponential(1e-5) a zero-rate block that never fails.
        assert Exponential(1.0e-5).rate_per_hour == 1.0e-5

    def test_describe_mentions_units(self) -> None:
        assert "/h" in Exponential(1e-5).describe()
        assert " h" in Weibull(shape=1.0, scale_hours=10.0).describe()
