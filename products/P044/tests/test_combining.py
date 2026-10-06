"""Tests for the combiners, outage probability and diversity order."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from aperturediv.channel import lognormal_cdf
from aperturediv.combining import (
    COMBINERS,
    NORMALISATIONS,
    branch_mean_snr,
    combined_gain,
    diversity_order,
    egc_gain,
    mrc_gain,
    outage_probability,
    sc_gain,
    weighted_snr,
)
from aperturediv.correlation import sample_correlated_lognormal

_IRRADIANCE = hnp.arrays(
    dtype=float,
    shape=hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
    elements=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
)


class TestKnownAnswers:
    def test_single_branch_all_schemes_agree(self):
        arr = np.array([[0.25]])
        assert float(mrc_gain(arr)[0]) == pytest.approx(0.25, rel=1e-15)
        assert float(egc_gain(arr)[0]) == pytest.approx(0.25, rel=1e-12)
        assert float(sc_gain(arr)[0]) == pytest.approx(0.25, rel=1e-15)

    def test_two_equal_branches(self):
        # I = (1, 1): MRC = 2, EGC = (1+1)^2/2 = 2, SC = 1.
        arr = np.array([[1.0, 1.0]])
        assert float(mrc_gain(arr)[0]) == pytest.approx(2.0, rel=1e-15)
        assert float(egc_gain(arr)[0]) == pytest.approx(2.0, rel=1e-15)
        assert float(sc_gain(arr)[0]) == pytest.approx(1.0, rel=1e-15)

    def test_one_branch_in_total_fade(self):
        # I = (1, 0): MRC = 1, EGC = (1+0)^2/2 = 0.5, SC = 1.
        # EGC loses to SC here, which is why EGC >= SC is not asserted anywhere.
        arr = np.array([[1.0, 0.0]])
        assert float(mrc_gain(arr)[0]) == pytest.approx(1.0, rel=1e-15)
        assert float(egc_gain(arr)[0]) == pytest.approx(0.5, rel=1e-15)
        assert float(sc_gain(arr)[0]) == pytest.approx(1.0, rel=1e-15)
        assert float(egc_gain(arr)[0]) < float(sc_gain(arr)[0])

    def test_unequal_branches(self):
        # I = (4, 1): MRC = 5, EGC = (2+1)^2/2 = 4.5, SC = 4.
        arr = np.array([[4.0, 1.0]])
        assert float(mrc_gain(arr)[0]) == pytest.approx(5.0, rel=1e-15)
        assert float(egc_gain(arr)[0]) == pytest.approx(4.5, rel=1e-15)
        assert float(sc_gain(arr)[0]) == pytest.approx(4.0, rel=1e-15)

    def test_four_equal_branches(self):
        # I = (0.5 x 4): MRC = 2, EGC = (4 sqrt 0.5)^2/4 = 2, SC = 0.5.
        arr = np.full((1, 4), 0.5)
        assert float(mrc_gain(arr)[0]) == pytest.approx(2.0, rel=1e-15)
        assert float(egc_gain(arr)[0]) == pytest.approx(2.0, rel=1e-14)
        assert float(sc_gain(arr)[0]) == pytest.approx(0.5, rel=1e-15)

    def test_weighted_snr_known_answer(self):
        # w = (1,1), h = (2,1), gamma_bar = 1: (w.h)^2/||w||^2 = 9/2 = 4.5.
        assert float(weighted_snr(np.array([[1.0, 1.0]]), np.array([[2.0, 1.0]]), 1.0)[0]) == (
            pytest.approx(4.5, rel=1e-15)
        )

    def test_weighted_snr_matches_mrc_at_optimal_weights(self):
        h = np.array([[2.0, 1.0, 0.5]])
        got = float(weighted_snr(h, h, 1.0)[0])
        assert got == pytest.approx(float(mrc_gain(h**2)[0]), rel=1e-14)


class TestOrdering:
    @given(_IRRADIANCE)
    def test_mrc_dominates_egc_and_sc(self, arr):
        g_mrc, g_egc, g_sc = mrc_gain(arr), egc_gain(arr), sc_gain(arr)
        assert np.all(g_mrc >= g_egc - 1e-9 * np.maximum(1.0, g_mrc))
        assert np.all(g_mrc >= g_sc - 1e-9 * np.maximum(1.0, g_mrc))

    @given(_IRRADIANCE)
    def test_weighted_snr_never_exceeds_mrc(self, arr):
        h = np.sqrt(arr)
        w = np.ones_like(h)
        assert np.all(weighted_snr(w, h, 1.0) <= mrc_gain(arr) * (1.0 + 1e-9))

    def test_sc_sometimes_beats_egc(self, rng):
        irr = sample_correlated_lognormal(200_000, 1.0, np.eye(3), rng)
        assert float(np.mean(sc_gain(irr) > egc_gain(irr) * (1.0 + 1e-12))) > 0.0

    def test_mrc_gain_grows_with_branch_count(self, rng):
        irr = sample_correlated_lognormal(50_000, 0.6, np.eye(4), rng)
        means = [mrc_gain(irr[:, :n]).mean() for n in (1, 2, 3, 4)]
        assert all(y > x for x, y in zip(means, means[1:], strict=False))


class TestInputValidation:
    def test_one_dimensional_input_is_a_single_branch(self):
        got = mrc_gain(np.array([0.3, 0.7]))
        assert got.shape == (2,)
        assert np.allclose(got, [0.3, 0.7])

    @pytest.mark.parametrize("bad", [np.array([[-1.0]]), np.array([[np.nan]]),
                                     np.array([[np.inf]])])
    def test_rejects_bad_irradiance(self, bad):
        with pytest.raises(ValueError):
            mrc_gain(bad)

    def test_rejects_three_dimensional(self):
        with pytest.raises(ValueError):
            mrc_gain(np.zeros((2, 2, 2)))

    def test_rejects_empty_aperture_axis(self):
        with pytest.raises(ValueError):
            mrc_gain(np.zeros((4, 0)))

    def test_combined_gain_rejects_unknown_scheme(self):
        with pytest.raises(ValueError, match="scheme must be one of"):
            combined_gain(np.array([[1.0]]), "mmse")

    def test_combined_gain_dispatches(self):
        arr = np.array([[4.0, 1.0]])
        assert float(combined_gain(arr, "mrc")[0]) == pytest.approx(5.0)
        assert float(combined_gain(arr, "egc")[0]) == pytest.approx(4.5)
        assert float(combined_gain(arr, "sc")[0]) == pytest.approx(4.0)

    def test_weighted_snr_rejects_zero_weights(self):
        with pytest.raises(ValueError, match="non-zero norm"):
            weighted_snr(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]), 1.0)

    def test_weighted_snr_rejects_mismatched_axis(self):
        with pytest.raises(ValueError, match="last axis"):
            weighted_snr(np.array([[1.0, 1.0]]), np.array([[1.0]]), 1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_weighted_snr_rejects_bad_mean_snr(self, bad):
        with pytest.raises(ValueError):
            weighted_snr(np.array([[1.0]]), np.array([[1.0]]), bad)


class TestBranchMeanSnr:
    def test_fixed_total_divides(self):
        assert branch_mean_snr(100.0, 4, "fixed_total") == pytest.approx(25.0, rel=1e-15)

    def test_fixed_branch_keeps(self):
        assert branch_mean_snr(100.0, 4, "fixed_branch") == pytest.approx(100.0, rel=1e-15)

    def test_both_normalisations_listed(self):
        assert set(NORMALISATIONS) == {"fixed_total", "fixed_branch"}

    def test_rejects_unknown_normalisation(self):
        with pytest.raises(ValueError, match="normalisation must be one of"):
            branch_mean_snr(100.0, 2, "per_photon")

    @pytest.mark.parametrize("g,n", [(0.0, 2), (-1.0, 2), (10.0, 0)])
    def test_rejects_bad_input(self, g, n):
        with pytest.raises(ValueError):
            branch_mean_snr(g, n)


class TestOutageProbability:
    def test_single_branch_matches_lognormal_cdf(self, rng):
        irr = sample_correlated_lognormal(400_000, 0.6, np.eye(1), rng)
        gain = mrc_gain(irr)
        for snr_db in (5.0, 12.0, 20.0):
            want = float(lognormal_cdf(10.0 ** ((5.0 - snr_db) / 10.0), 0.6))
            got = float(outage_probability(gain, snr_db, 5.0))
            assert got == pytest.approx(want, abs=4e-3)

    def test_decreasing_in_mean_snr(self, rng):
        gain = mrc_gain(sample_correlated_lognormal(50_000, 0.6, np.eye(2), rng))
        p = outage_probability(gain, np.arange(0.0, 40.0, 1.0), 5.0)
        assert np.all(np.diff(p) <= 0.0)

    def test_array_shape_preserved(self, rng):
        gain = mrc_gain(sample_correlated_lognormal(1000, 0.6, np.eye(2), rng))
        p = outage_probability(gain, np.array([[0.0, 10.0], [20.0, 30.0]]), 5.0)
        assert p.shape == (2, 2)

    def test_scalar_returns_scalar(self, rng):
        gain = mrc_gain(sample_correlated_lognormal(1000, 0.6, np.eye(2), rng))
        assert np.ndim(outage_probability(gain, 10.0, 5.0)) == 0

    def test_resolution_floor(self, rng):
        gain = mrc_gain(sample_correlated_lognormal(1000, 0.6, np.eye(1), rng))
        assert float(outage_probability(gain, 200.0, 5.0)) == 0.0

    def test_rejects_empty_gain(self):
        with pytest.raises(ValueError):
            outage_probability(np.array([]), 10.0, 5.0)

    def test_rejects_two_dimensional_gain(self):
        with pytest.raises(ValueError):
            outage_probability(np.ones((2, 2)), 10.0, 5.0)


class TestDiversityOrder:
    def test_exact_power_law(self):
        # Construct P_out = C gamma_bar^-3 exactly; the slope must be 3.
        snr_db = np.arange(10.0, 40.0, 0.5)
        p = 1e-1 * 10.0 ** (-3.0 * (snr_db - 10.0) / 10.0)
        res = diversity_order(snr_db, p, window=(1e-6, 1e-1))
        assert res.order == pytest.approx(3.0, rel=1e-9)
        assert res.residual_rms < 1e-12

    def test_window_and_metadata_reported(self):
        snr_db = np.arange(10.0, 40.0, 0.5)
        p = 1e-1 * 10.0 ** (-2.0 * (snr_db - 10.0) / 10.0)
        res = diversity_order(snr_db, p, window=(1e-4, 1e-2))
        assert res.window == (1e-4, 1e-2)
        assert res.n_points >= 4
        assert res.snr_db_range[0] < res.snr_db_range[1]

    def test_zero_outage_points_dropped(self):
        snr_db = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
        p = np.array([1e-2, 5e-3, 2e-3, 1e-3, 0.0, 0.0])
        res = diversity_order(snr_db, p, window=(1e-4, 1e-1))
        assert res.n_points == 4

    def test_raises_when_too_few_points(self):
        snr_db = np.array([10.0, 20.0])
        p = np.array([1e-3, 1e-5])
        with pytest.raises(ValueError, match="usable points"):
            diversity_order(snr_db, p, window=(1e-4, 1e-2))

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            diversity_order(np.arange(5.0), np.arange(4.0))

    @pytest.mark.parametrize("window", [(0.0, 0.1), (0.1, 0.01), (0.1, 2.0)])
    def test_rejects_bad_window(self, window):
        snr_db = np.arange(10.0, 30.0, 0.5)
        p = 1e-1 * 10.0 ** (-2.0 * (snr_db - 10.0) / 10.0)
        with pytest.raises(ValueError, match="window"):
            diversity_order(snr_db, p, window=window)

    def test_correlated_order_below_independent(self, rng, line_array_correlation):
        snr_db = np.arange(0.0, 50.0, 0.5)
        ind = mrc_gain(sample_correlated_lognormal(400_000, 0.9, np.eye(4), rng))
        cor = mrc_gain(sample_correlated_lognormal(400_000, 0.9, line_array_correlation, rng))
        o_ind = diversity_order(snr_db, outage_probability(ind, snr_db, 5.0),
                                window=(1e-4, 1e-2)).order
        o_cor = diversity_order(snr_db, outage_probability(cor, snr_db, 5.0),
                                window=(1e-4, 1e-2)).order
        assert o_cor < o_ind

    def test_all_combiners_listed(self):
        assert set(COMBINERS) == {"mrc", "egc", "sc"}
