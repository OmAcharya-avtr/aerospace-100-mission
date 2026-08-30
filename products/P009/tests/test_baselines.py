"""Known-answer, property, and input-validation tests for the Kim/Kruse baselines.

Hand calculations use:
    alpha_dB = (10/ln 10) * (3.912 / V) * (lambda/550)^(-q),  10/ln 10 = 4.342944819.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fogcast import (
    kim_attenuation_db_km,
    kim_q,
    kruse_attenuation_db_km,
    kruse_q,
)


class TestKimQBranches:
    """Kim et al. 2001 (SPIE 4214) piecewise q(V), every branch hand-checked."""

    def test_dense_fog_branch(self):
        # V <= 0.5 km: q = 0.
        assert kim_q(0.3) == pytest.approx(0.0)
        assert kim_q(0.5) == pytest.approx(0.0)

    def test_fog_branch(self):
        # 0.5 < V <= 1: q = V - 0.5. V = 0.75 -> 0.25; V = 1.0 -> 0.5.
        assert kim_q(0.75) == pytest.approx(0.25)
        assert kim_q(1.0) == pytest.approx(0.5)

    def test_haze_branch(self):
        # 1 < V <= 6: q = 0.16 V + 0.34. V = 3 -> 0.82; V = 6 -> 1.30 (continuous).
        assert kim_q(3.0) == pytest.approx(0.82)
        assert kim_q(6.0) == pytest.approx(1.30)

    def test_clear_branch(self):
        # 6 < V <= 50: q = 1.3.
        assert kim_q(10.0) == pytest.approx(1.3)
        assert kim_q(50.0) == pytest.approx(1.3)

    def test_very_clear_branch(self):
        # V > 50: q = 1.6.
        assert kim_q(50.1) == pytest.approx(1.6)
        assert kim_q(80.0) == pytest.approx(1.6)


class TestKruseQBranches:
    """Kruse et al. 1962 q(V) branches."""

    def test_low_visibility_branch(self):
        # V <= 6: q = 0.585 V^(1/3). V = 0.3 -> 0.585 * 0.669433 = 0.391618.
        assert kruse_q(0.3) == pytest.approx(0.5850 * 0.3 ** (1 / 3), rel=1e-12)
        assert kruse_q(0.3) == pytest.approx(0.391618, abs=1e-6)

    def test_mid_and_high_branches(self):
        assert kruse_q(10.0) == pytest.approx(1.3)
        assert kruse_q(60.0) == pytest.approx(1.6)


class TestKnownAnswers:
    def test_v10_1550nm_both_models(self):
        # V = 10 km, lambda = 1550 nm, q = 1.3 (both models):
        # sigma = (3.912/10) * (1550/550)^(-1.3) = 0.3912 * 0.260026 = 0.101722 /km
        # alpha = 4.342944819 * 0.101722 = 0.441798 dB/km
        assert kim_attenuation_db_km(10.0, 1550.0) == pytest.approx(0.441798, abs=1e-4)
        assert kruse_attenuation_db_km(10.0, 1550.0) == pytest.approx(0.441798, abs=1e-4)

    def test_dense_fog_kim_wavelength_independent(self):
        # V = 0.3 km, Kim q = 0: alpha = 4.342944819 * 3.912 / 0.3 = 56.632 dB/km
        # at ANY wavelength (Kim's key claim for dense fog).
        expected = 4.342944819 * 3.912 / 0.3
        assert kim_attenuation_db_km(0.3, 850.0) == pytest.approx(expected, rel=1e-9)
        assert kim_attenuation_db_km(0.3, 1550.0) == pytest.approx(expected, rel=1e-9)
        assert expected == pytest.approx(56.632, abs=1e-3)

    def test_fog_branch_kim_v075_850nm(self):
        # V = 0.75, lambda = 850: q = 0.25,
        # (850/550)^(-0.25) = exp(-0.25 * ln 1.545455) = exp(-0.108876) = 0.896840
        # alpha = 4.342944819 * (3.912/0.75) * 0.896840 = 20.3169 dB/km
        assert kim_attenuation_db_km(0.75, 850.0) == pytest.approx(20.3169, abs=1e-3)

    def test_kruse_dense_fog_1550nm(self):
        # V = 0.3, Kruse q = 0.391618:
        # (1550/550)^(-0.391618) = exp(-0.391618 * 1.036092) = 0.666486
        # alpha = 56.632 * 0.666486 = 37.744 dB/km
        assert kruse_attenuation_db_km(0.3, 1550.0) == pytest.approx(37.744, abs=1e-2)

    def test_kim_kruse_agree_above_6km(self):
        # Identical q for V > 6 km -> identical attenuation.
        for v in (6.5, 20.0, 60.0):
            for lam in (850.0, 1310.0, 1550.0):
                assert kim_attenuation_db_km(v, lam) == pytest.approx(
                    kruse_attenuation_db_km(v, lam), rel=1e-12
                )

    def test_kim_kruse_disagree_in_dense_fog(self):
        # Documented disagreement: at V = 0.3 km, 1550 nm, Kruse predicts a large
        # long-wavelength advantage (q = 0.39) while Kim predicts none (q = 0).
        kim = kim_attenuation_db_km(0.3, 1550.0)
        kruse = kruse_attenuation_db_km(0.3, 1550.0)
        assert kim > 1.4 * kruse  # 56.63 vs 37.74 dB/km


class TestMonotonicity:
    @settings(max_examples=200, deadline=None)
    @given(
        v1=st.floats(min_value=0.05, max_value=100.0),
        v2=st.floats(min_value=0.05, max_value=100.0),
        lam=st.floats(min_value=550.0, max_value=2000.0),
    )
    def test_attenuation_decreases_with_visibility(self, v1, v2, lam):
        """Property: better visibility never increases attenuation, for lambda >= 550 nm.

        The 550 nm lower bound is load-bearing, not cosmetic. Attenuation scales as
        (lambda / 550)^(-q), and q is piecewise in V with upward steps at the band
        boundaries (V = 50 km for Kim, V = 6 km for Kruse). Below 550 nm the base is
        less than 1, so a step UP in q is a step UP in attenuation, and the models are
        genuinely non-monotone across those boundaries. That is a property of the
        published empirical fits, not a defect in this implementation, and it is
        pinned by test_band_boundary_reversal_below_550nm below rather than hidden.

        Hypothesis found this at lambda = 500 nm; the original bound of 500.0 asserted
        a property the Kim and Kruse models do not have.
        """
        lo, hi = sorted((v1, v2))
        assert kim_attenuation_db_km(lo, lam) >= kim_attenuation_db_km(hi, lam) - 1e-12
        assert kruse_attenuation_db_km(lo, lam) >= kruse_attenuation_db_km(hi, lam) - 1e-12

    def test_band_boundary_reversal_below_550nm(self):
        """Pin the documented non-monotonicity below 550 nm so it cannot vanish silently.

        Known-answer values computed from the model definitions in baselines.py:
        at 500 nm, crossing Kim's q step at V = 50 km (q: 1.3 -> 1.6) and Kruse's at
        V = 6 km (q: 1.3 -> 1.6) both INCREASE attenuation as visibility improves.
        """
        below, above = 49.9999, 50.0001
        assert kim_attenuation_db_km(above, 500.0) > kim_attenuation_db_km(below, 500.0)
        # ... and the reversal is absent at and above the 550 nm reference wavelength.
        assert kim_attenuation_db_km(above, 550.0) <= kim_attenuation_db_km(below, 550.0)

        below, above = 5.9999, 6.0001
        assert kruse_attenuation_db_km(above, 500.0) > kruse_attenuation_db_km(below, 500.0)
        assert kruse_attenuation_db_km(above, 550.0) <= kruse_attenuation_db_km(below, 550.0)

    @settings(max_examples=100, deadline=None)
    @given(
        v=st.floats(min_value=0.05, max_value=100.0),
        lam=st.floats(min_value=550.0, max_value=2000.0),
    )
    def test_longer_wavelength_never_worse_than_550(self, v, lam):
        """For lambda >= 550 nm, q >= 0 implies attenuation <= the 550 nm value."""
        assert kim_attenuation_db_km(v, lam) <= kim_attenuation_db_km(v, 550.0) + 1e-12
        assert kruse_attenuation_db_km(v, lam) <= kruse_attenuation_db_km(v, 550.0) + 1e-12


class TestInputValidation:
    @pytest.mark.parametrize("fn", [kim_attenuation_db_km, kruse_attenuation_db_km])
    def test_rejects_nonpositive_and_out_of_range_visibility(self, fn):
        for bad_v in (-1.0, 0.0, 0.01, 150.0, np.nan, np.inf):
            with pytest.raises(ValueError):
                fn(bad_v, 1550.0)

    @pytest.mark.parametrize("fn", [kim_attenuation_db_km, kruse_attenuation_db_km])
    def test_rejects_out_of_range_wavelength(self, fn):
        for bad_lam in (-500.0, 0.0, 100.0, 3000.0, np.nan):
            with pytest.raises(ValueError):
                fn(10.0, bad_lam)

    def test_array_inputs_broadcast(self):
        v = np.array([0.3, 1.0, 10.0])
        out = kim_attenuation_db_km(v, 1550.0)
        assert out.shape == (3,)
        assert np.all(np.diff(out) < 0)  # decreasing with visibility
