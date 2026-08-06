"""Monte Carlo engine tests: reproducibility and statistical agreement.

Statistical tests use a 99.9% Wilson CI so the per-assertion false-alarm
probability is ~1e-3; seeds are fixed so reruns are deterministic anyway.
"""

import numpy as np
import pytest

from berbench import analytic_ber, mc_ber, n_bits_for_target


class TestReproducibility:
    def test_same_seed_same_result(self):
        a = mc_ber("bpsk", 4.0, n=100_000, seed=42)
        b = mc_ber("bpsk", 4.0, n=100_000, seed=42)
        np.testing.assert_array_equal(a.n_errors, b.n_errors)
        np.testing.assert_array_equal(a.n_bits, b.n_bits)

    def test_different_seed_different_result(self):
        a = mc_ber("bpsk", 4.0, n=100_000, seed=1)
        b = mc_ber("bpsk", 4.0, n=100_000, seed=2)
        assert a.n_errors[0] != b.n_errors[0]


class TestStatisticalAgreement:
    """Analytic BER must lie inside the 99.9% Wilson CI of the MC estimate.

    Sample sizes are chosen with n_bits_for_target so each point observes
    >> 100 errors, per the documented sizing rule.
    """

    @pytest.mark.parametrize(
        "mod,kwargs",
        [
            ("bpsk", {}),
            ("ook", {}),
            ("ppm", {"M": 4}),
            ("ppm", {"M": 16}),
            ("ook", {"threshold": 0.4}),
        ],
    )
    def test_awgn_agreement(self, mod, kwargs):
        snr = np.array([0.0, 4.0, 7.0])
        ana = analytic_ber(mod, snr, **kwargs).ber
        n = min(20_000_000, n_bits_for_target(float(ana.min()), min_errors=300))
        mc = mc_ber(mod, snr, n=n, seed=123, ci_level=0.999, **kwargs)
        assert np.all(mc.n_errors >= 100), f"undersized run: {mc.n_errors}"
        assert np.all((ana >= mc.ci_low) & (ana <= mc.ci_high)), (
            f"{mod} {kwargs}: analytic {ana} outside CI [{mc.ci_low}, {mc.ci_high}]"
        )

    @pytest.mark.parametrize(
        "mod,kwargs",
        [
            ("ook", {"sigma_i2": 0.3}),
            ("ook", {"sigma_i2": 0.3, "threshold": 0.5}),
            ("bpsk", {"sigma_i2": 0.1}),
            ("ppm", {"sigma_i2": 0.2, "M": 4}),
        ],
    )
    def test_lognormal_agreement(self, mod, kwargs):
        snr = np.array([4.0, 8.0])
        ana = analytic_ber(mod, snr, channel="lognormal", **kwargs).ber
        n = min(2_000_000, n_bits_for_target(float(ana.min()), min_errors=300))
        mc = mc_ber(mod, snr, n=n, seed=321, channel="lognormal", ci_level=0.999, **kwargs)
        assert np.all(mc.n_errors >= 100)
        assert np.all((ana >= mc.ci_low) & (ana <= mc.ci_high))


class TestEngineBehaviour:
    def test_ppm_rounds_up_to_whole_symbols(self):
        res = mc_ber("ppm", 4.0, n=1001, seed=0, M=8)  # k = 3 bits/symbol
        assert res.n_bits[0] == 1002  # ceil(1001/3)*3

    def test_ci_brackets_estimate(self):
        res = mc_ber("ook", 2.0, n=50_000, seed=5)
        assert res.ci_low[0] <= res.ber[0] <= res.ci_high[0]
        assert res.ci_method == "wilson"

    def test_max_seconds_budget(self):
        # A absurdly small budget must stop early and flag it, not hang.
        res = mc_ber("bpsk", np.arange(0.0, 10.0), n=50_000_000, seed=0, max_seconds=0.3)
        assert res.budget_exhausted
        assert res.n_bits.sum() < 10 * 50_000_000
        assert res.elapsed_s < 5.0

    def test_n_bits_for_target(self):
        assert n_bits_for_target(1e-3, min_errors=100) == 100_000
        with pytest.raises(ValueError):
            n_bits_for_target(0.0)
        with pytest.raises(ValueError):
            n_bits_for_target(1e-3, min_errors=0)
