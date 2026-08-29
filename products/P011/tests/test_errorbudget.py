"""Tests for waveforge.errorbudget."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.errorbudget import (
    HUDGIN_FITTING_COEFFICIENT,
    ErrorBudget,
    bandwidth_error,
    delay_error,
    fitting_error,
    ideal_filter_fitting_coefficient,
    noise_error,
    strehl_marechal,
    strehl_marechal_quadratic,
    variance_from_strehl,
)


class TestFittingError:
    def test_ideal_filter_coefficient(self):
        # 0.022903 * 2 pi * 0.6 * 2^(5/3) = 0.27412
        assert ideal_filter_fitting_coefficient() == pytest.approx(0.27412, abs=1e-5)

    def test_ideal_coefficient_close_to_hudgin(self):
        # The engineering value 0.28 (Hudgin 1977) is the same number rounded
        assert ideal_filter_fitting_coefficient() == pytest.approx(
            HUDGIN_FITTING_COEFFICIENT, rel=0.03
        )

    def test_five_thirds_scaling(self):
        a = fitting_error(0.1, 0.1)
        b = fitting_error(0.2, 0.1)
        assert b / a == pytest.approx(2.0 ** (5 / 3), rel=1e-12)

    def test_value_at_unit_ratio(self):
        assert fitting_error(0.1, 0.1) == pytest.approx(HUDGIN_FITTING_COEFFICIENT)

    def test_r0_scaling(self):
        assert fitting_error(0.1, 0.05) / fitting_error(0.1, 0.1) == pytest.approx(
            2.0 ** (5 / 3), rel=1e-12
        )

    def test_custom_coefficient(self):
        assert fitting_error(0.1, 0.1, 0.34) == pytest.approx(0.34)

    @pytest.mark.parametrize("pitch", [0.0, -0.1, float("nan")])
    def test_bad_pitch(self, pitch):
        with pytest.raises(ValueError, match="actuator_pitch_m"):
            fitting_error(pitch, 0.1)

    def test_bad_r0(self):
        with pytest.raises(ValueError, match="r0_m"):
            fitting_error(0.1, 0.0)

    def test_bad_coefficient(self):
        with pytest.raises(ValueError, match="coefficient"):
            fitting_error(0.1, 0.1, 0.0)


class TestTemporalError:
    def test_unit_at_tau0(self):
        # tau0 = 0.314 * 0.1 / 10 = 3.14 ms; delay = tau0 -> variance 1 rad^2
        assert delay_error(3.14e-3, 0.1, 10.0) == pytest.approx(1.0, rel=1e-6)

    def test_five_thirds_scaling(self):
        a = delay_error(1e-3, 0.1, 10.0)
        b = delay_error(2e-3, 0.1, 10.0)
        assert b / a == pytest.approx(2.0 ** (5 / 3), rel=1e-12)

    def test_zero_delay(self):
        assert delay_error(0.0, 0.1, 10.0) == 0.0

    def test_bandwidth_unit_at_greenwood_frequency(self):
        # f_G = 0.427 * 10 / 0.1 = 42.7 Hz
        assert bandwidth_error(42.7, 0.1, 10.0) == pytest.approx(1.0, rel=1e-9)

    def test_bandwidth_scaling(self):
        a = bandwidth_error(100.0, 0.1, 10.0)
        b = bandwidth_error(200.0, 0.1, 10.0)
        assert a / b == pytest.approx(2.0 ** (5 / 3), rel=1e-12)

    def test_bad_delay(self):
        with pytest.raises(ValueError, match="delay_s"):
            delay_error(-1e-3, 0.1, 10.0)

    def test_bad_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth_hz"):
            bandwidth_error(0.0, 0.1, 10.0)


class TestNoiseError:
    def test_zero_noise_gives_zero(self):
        p = np.eye(4)
        assert noise_error(0.0, p) == 0.0

    def test_quadratic_in_sigma(self):
        p = np.random.default_rng(0).normal(size=(20, 6))
        a = noise_error(1.0, p)
        b = noise_error(2.0, p)
        assert b / a == pytest.approx(4.0, rel=1e-12)

    def test_linear_in_noise_gain(self):
        p = np.random.default_rng(1).normal(size=(20, 6))
        assert noise_error(1.0, p, 3.0) == pytest.approx(3.0 * noise_error(1.0, p, 1.0))

    def test_known_answer_for_identity_propagation(self):
        # P = I(4) with the column means removed leaves each column with
        # variance sum 1 - 1/4 = 0.75, so mean diag(PP^T) = 0.75
        assert noise_error(1.0, np.eye(4)) == pytest.approx(0.75)

    def test_constant_propagation_is_pure_piston(self):
        # A propagation matrix with identical rows produces only piston, which
        # is removed, so it contributes no residual variance.
        p = np.ones((10, 3))
        assert noise_error(5.0, p) == pytest.approx(0.0, abs=1e-24)

    def test_bad_sigma(self):
        with pytest.raises(ValueError, match="slope_noise_sigma"):
            noise_error(-1.0, np.eye(3))

    def test_bad_gain(self):
        with pytest.raises(ValueError, match="noise_gain"):
            noise_error(1.0, np.eye(3), -1.0)

    def test_bad_matrix_rank(self):
        with pytest.raises(ValueError, match="2-D"):
            noise_error(1.0, np.zeros(4))


class TestStrehl:
    def test_zero_variance_gives_unity(self):
        assert float(strehl_marechal(0.0)) == pytest.approx(1.0)
        assert float(strehl_marechal_quadratic(0.0)) == pytest.approx(1.0)

    def test_known_answers(self):
        # exp(-0.5) = 0.606531, 1 - 0.5 = 0.5
        assert float(strehl_marechal(0.5)) == pytest.approx(0.606531, abs=1e-6)
        assert float(strehl_marechal_quadratic(0.5)) == pytest.approx(0.5)

    def test_quadratic_agrees_for_small_variance(self):
        var = 0.01
        assert float(strehl_marechal(var)) == pytest.approx(
            float(strehl_marechal_quadratic(var)), rel=1e-4
        )

    def test_quadratic_diverges_for_large_variance(self):
        var = 1.5
        assert float(strehl_marechal_quadratic(var)) == 0.0
        assert float(strehl_marechal(var)) > 0.0

    def test_round_trip(self):
        var = 0.37
        assert float(variance_from_strehl(float(strehl_marechal(var)))) == pytest.approx(var)

    def test_array_input(self):
        assert strehl_marechal(np.array([0.0, 1.0])).shape == (2,)

    def test_negative_variance_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            strehl_marechal(-0.1)
        with pytest.raises(ValueError, match="non-negative"):
            strehl_marechal_quadratic(-0.1)

    @pytest.mark.parametrize("value", [0.0, -0.5, 1.5])
    def test_bad_strehl(self, value):
        with pytest.raises(ValueError, match="Strehl"):
            variance_from_strehl(value)


class TestErrorBudget:
    def test_total_is_the_sum(self):
        budget = ErrorBudget(fitting=0.1, temporal=0.2, noise=0.05, other=0.01)
        assert budget.total == pytest.approx(0.36)

    def test_rms(self):
        assert ErrorBudget(fitting=0.25).rms_rad == pytest.approx(0.5)

    def test_strehl(self):
        assert ErrorBudget(fitting=0.5).strehl == pytest.approx(np.exp(-0.5))

    def test_dominant_term(self):
        assert ErrorBudget(fitting=0.1, temporal=0.4).dominant_term() == "temporal"
        assert ErrorBudget(fitting=0.4, temporal=0.1).dominant_term() == "fitting"
        assert ErrorBudget(noise=1.0).dominant_term() == "noise"
        assert ErrorBudget(other=1.0).dominant_term() == "other"

    def test_as_dict_keys(self):
        keys = set(ErrorBudget().as_dict())
        assert keys == {
            "fitting_rad2",
            "temporal_rad2",
            "noise_rad2",
            "other_rad2",
            "total_rad2",
            "rms_rad",
            "strehl_marechal",
        }

    def test_empty_budget_is_perfect(self):
        assert ErrorBudget().strehl == pytest.approx(1.0)

    @pytest.mark.parametrize("field", ["fitting", "temporal", "noise", "other"])
    def test_negative_terms_rejected(self, field):
        with pytest.raises(ValueError, match=field):
            ErrorBudget(**{field: -0.1})

    def test_is_frozen(self):
        budget = ErrorBudget(fitting=0.1)
        with pytest.raises(AttributeError):
            budget.fitting = 0.2
