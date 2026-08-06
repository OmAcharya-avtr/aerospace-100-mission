"""Input-validation tests: documented policy is ValueError/TypeError with
actionable messages; negative snr_db is VALID (dB scale), NaN/inf is not."""

import numpy as np
import pytest

from berbench import analytic_ber, mc_ber, wilson_interval


class TestModulationAndChannel:
    def test_unknown_modulation(self):
        with pytest.raises(ValueError, match="unknown modulation"):
            analytic_ber("qam", 5.0)

    def test_unknown_channel(self):
        with pytest.raises(ValueError, match="unknown channel"):
            analytic_ber("bpsk", 5.0, channel="rayleigh")

    def test_lognormal_requires_sigma_i2(self):
        with pytest.raises(ValueError, match="sigma_i2"):
            analytic_ber("ook", 5.0, channel="lognormal")

    def test_sigma_i2_must_be_positive(self):
        for bad in (0.0, -0.3, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="sigma_i2"):
                analytic_ber("ook", 5.0, channel="lognormal", sigma_i2=bad)

    def test_sigma_i2_rejected_for_awgn(self):
        with pytest.raises(ValueError, match="lognormal"):
            analytic_ber("ook", 5.0, channel="awgn", sigma_i2=0.3)


class TestSNRPolicy:
    def test_negative_snr_db_is_valid(self):
        # Documented policy: negative dB is physically meaningful.
        res = analytic_ber("bpsk", -10.0)
        assert 0.3 < res.ber[0] < 0.5

    def test_nan_snr_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            analytic_ber("bpsk", float("nan"))

    def test_inf_snr_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            mc_ber("bpsk", float("inf"), n=1000)

    def test_2d_snr_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            analytic_ber("bpsk", np.zeros((2, 2)))


class TestPPMParams:
    def test_m_not_power_of_two(self):
        with pytest.raises(ValueError, match="power of two"):
            analytic_ber("ppm", 5.0, M=3)

    def test_m_too_small(self):
        with pytest.raises(ValueError, match="power of two"):
            analytic_ber("ppm", 5.0, M=1)

    def test_m_not_integer(self):
        with pytest.raises(TypeError, match="integer"):
            analytic_ber("ppm", 5.0, M=4.0)

    def test_m_too_large(self):
        with pytest.raises(ValueError, match="not supported"):
            analytic_ber("ppm", 5.0, M=8192)

    def test_bad_ppm_method(self):
        with pytest.raises(ValueError, match="ppm_method"):
            analytic_ber("ppm", 5.0, ppm_method="chernoff")


class TestOOKThreshold:
    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5, float("nan")])
    def test_threshold_out_of_range(self, bad):
        with pytest.raises(ValueError, match="threshold"):
            analytic_ber("ook", 5.0, threshold=bad)

    def test_threshold_string_other_than_optimal(self):
        with pytest.raises(ValueError):
            analytic_ber("ook", 5.0, threshold="midpoint")


class TestMCParams:
    def test_n_must_be_positive_int(self):
        for bad in (0, -5, 2.5, True):
            with pytest.raises(ValueError, match="n must be"):
                mc_ber("bpsk", 5.0, n=bad)

    def test_seed_must_be_nonnegative_int(self):
        with pytest.raises(ValueError, match="seed"):
            mc_ber("bpsk", 5.0, n=1000, seed=-1)

    def test_ci_level_range(self):
        for bad in (0.0, 1.0, 1.5):
            with pytest.raises(ValueError, match="ci_level"):
                mc_ber("bpsk", 5.0, n=1000, ci_level=bad)

    def test_max_seconds_positive(self):
        with pytest.raises(ValueError, match="max_seconds"):
            mc_ber("bpsk", 5.0, n=1000, max_seconds=0.0)


class TestWilson:
    def test_wilson_known_value(self):
        # k=10, n=100, 95%: Wilson interval = (0.0552, 0.1744)
        # (hand-computed from Wilson 1927 formula with z = 1.959964)
        lo, hi = wilson_interval(10, 100, 0.95)
        assert lo == pytest.approx(0.05522, abs=2e-4)
        assert hi == pytest.approx(0.17437, abs=2e-4)

    def test_wilson_zero_errors(self):
        lo, hi = wilson_interval(0, 1000, 0.95)
        assert lo == 0.0
        assert 0.0 < hi < 0.01

    def test_wilson_invalid(self):
        with pytest.raises(ValueError):
            wilson_interval(5, 0)
        with pytest.raises(ValueError):
            wilson_interval(-1, 10)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)
