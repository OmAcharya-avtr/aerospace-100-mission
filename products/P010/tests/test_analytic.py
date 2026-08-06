"""Known-answer and identity tests for the analytic BER expressions."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from berbench import analytic_ber, qfunc


class TestKnownAnswers:
    def test_bpsk_0db(self):
        # Hand calculation: gamma = 1, Pb = Q(sqrt(2)) = 1 - Phi(1.41421356)
        # Phi(1.41421356) = 0.92135040 (Abramowitz & Stegun Table 26.1)
        # => Pb = 0.07864960
        res = analytic_ber("bpsk", 0.0)
        assert res.ber[0] == pytest.approx(0.0786496035, rel=1e-8)

    def test_bpsk_10db(self):
        # gamma = 10, Pb = Q(sqrt(20)) = Q(4.47213595) = 3.87210822e-6
        # (Proakis & Salehi 2008, Eq. (4.3-13); Q evaluated via erfc)
        res = analytic_ber("bpsk", 10.0)
        assert res.ber[0] == pytest.approx(3.872108216e-6, rel=1e-6)

    def test_ook_0db(self):
        # gamma = 1, Pb = Q(1) = 0.15865525 (standard normal table)
        res = analytic_ber("ook", 0.0)
        assert res.ber[0] == pytest.approx(0.1586552539, rel=1e-8)

    def test_ook_is_3db_worse_than_bpsk(self):
        # Q(sqrt(gamma)) = Q(sqrt(2*(gamma/2))): OOK at gamma equals BPSK at
        # gamma/2, i.e. exactly 3.0103 dB penalty (Proakis Sec. 4.3).
        snr = np.array([2.0, 6.0, 10.0])
        ook = analytic_ber("ook", snr).ber
        bpsk = analytic_ber("bpsk", snr - 10 * np.log10(2.0)).ber
        np.testing.assert_allclose(ook, bpsk, rtol=1e-12)

    def test_ppm_m2_equals_binary_orthogonal(self):
        # M=2 orthogonal signalling: Pb = Q(sqrt(Eb/N0)) exactly
        # (Proakis & Salehi 2008, Eq. (4.2-90) binary orthogonal).
        snr = np.array([0.0, 4.0, 8.0])
        ppm2 = analytic_ber("ppm", snr, M=2).ber
        expected = qfunc(np.sqrt(10 ** (snr / 10)))
        np.testing.assert_allclose(ppm2, expected, rtol=1e-9)

    def test_ook_fixed_midpoint_equals_optimal_awgn(self):
        # Under AWGN the optimal threshold IS the midpoint, so
        # threshold=0.5 must reproduce threshold="optimal".
        snr = np.array([0.0, 5.0, 10.0])
        opt = analytic_ber("ook", snr, threshold="optimal").ber
        mid = analytic_ber("ook", snr, threshold=0.5).ber
        np.testing.assert_allclose(mid, opt, rtol=1e-12)

    def test_ook_asymmetric_threshold_hand_value(self):
        # gamma = 1, t = 0.25: Pb = 0.5*(Q(0.5) + Q(1.5))
        # Q(0.5) = 0.30853754, Q(1.5) = 0.06680720 => Pb = 0.18767237
        res = analytic_ber("ook", 0.0, threshold=0.25)
        expected = 0.5 * (0.3085375387 + 0.0668072013)
        assert res.ber[0] == pytest.approx(expected, rel=1e-8)


class TestPPMBoundVsExact:
    @settings(max_examples=40, deadline=None)
    @given(
        snr_db=st.floats(min_value=-5.0, max_value=14.0),
        m_exp=st.integers(min_value=1, max_value=6),
    )
    def test_union_bound_upper_bounds_exact(self, snr_db, m_exp):
        m = 2**m_exp
        exact = analytic_ber("ppm", snr_db, M=m, ppm_method="exact").ber[0]
        bound = analytic_ber("ppm", snr_db, M=m, ppm_method="union").ber[0]
        assert bound >= exact - 1e-12

    def test_bound_tightens_at_high_snr(self):
        exact = analytic_ber("ppm", np.array([2.0, 10.0]), M=4).ber
        bound = analytic_ber("ppm", np.array([2.0, 10.0]), M=4, ppm_method="union").ber
        ratio = bound / exact
        assert ratio[1] < ratio[0]  # tighter at higher SNR
        assert ratio[1] == pytest.approx(1.0, abs=0.02)


class TestLognormal:
    def test_small_scintillation_approaches_awgn(self):
        awgn = analytic_ber("bpsk", 6.0).ber[0]
        faded = analytic_ber("bpsk", 6.0, channel="lognormal", sigma_i2=1e-4).ber[0]
        assert faded == pytest.approx(awgn, rel=5e-2)
        assert faded > awgn  # fading always hurts at these SNRs (Jensen)

    def test_fading_degrades_ber(self):
        snr = np.array([4.0, 8.0, 12.0])
        for mod in ("ook", "bpsk"):
            awgn = analytic_ber(mod, snr).ber
            faded = analytic_ber(mod, snr, channel="lognormal", sigma_i2=0.3).ber
            assert np.all(faded > awgn)

    def test_more_scintillation_is_worse(self):
        b1 = analytic_ber("ook", 10.0, channel="lognormal", sigma_i2=0.1).ber[0]
        b3 = analytic_ber("ook", 10.0, channel="lognormal", sigma_i2=0.3).ber[0]
        assert b3 > b1

    def test_fixed_threshold_has_error_floor(self):
        # Fixed (no-CSI) threshold: as SNR -> inf, BER -> 0.5*P(I < t) > 0.
        # For sigma_I^2=0.3, t=0.5: P(I<0.5) = Phi((ln 0.5 + sz^2/2)/sz)
        # with sz^2 = ln(1.3) = 0.262364, sz = 0.512215:
        # = Phi(-1.097063) = 0.136312 => floor = 0.068156.
        res = analytic_ber(
            "ook", 30.0, channel="lognormal", sigma_i2=0.3, threshold=0.5, n_gh_nodes=256
        )
        assert res.ber[0] == pytest.approx(0.068156, rel=0.02)
        # while the adaptive threshold keeps falling
        adap = analytic_ber("ook", 30.0, channel="lognormal", sigma_i2=0.3).ber[0]
        assert adap < 1e-3

    def test_weak_fluctuation_warning(self):
        with pytest.warns(UserWarning, match="weak-fluctuation"):
            analytic_ber("ook", 6.0, channel="lognormal", sigma_i2=1.5)


class TestMonotonicity:
    @settings(max_examples=25, deadline=None)
    @given(mod=st.sampled_from(["ook", "bpsk", "ppm"]))
    def test_ber_decreases_with_snr(self, mod):
        snr = np.linspace(-5.0, 15.0, 21)
        ber = analytic_ber(mod, snr).ber
        assert np.all(np.diff(ber) < 0)
        assert np.all((ber >= 0) & (ber <= 0.5 + 1e-12))
