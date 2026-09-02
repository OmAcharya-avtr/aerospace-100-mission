"""Residual-window features: hand arithmetic and reference-vs-fast agreement."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fdiscope.features import (
    N_FEATURES,
    NIS_EXCEEDANCE_THRESHOLD,
    feature_matrix,
    feature_names,
    window_features,
)

NAMES = feature_names()


def index_of(name: str) -> int:
    return NAMES.index(name)


class TestNames:
    def test_count_matches_the_constant(self):
        assert len(NAMES) == N_FEATURES == 16

    def test_names_are_unique(self):
        assert len(set(NAMES)) == len(NAMES)

    def test_exceedance_threshold_known_answer(self):
        # 99th percentile of chi2 with 2 dof is -2 ln(0.01) = 9.2103403720
        assert np.isclose(NIS_EXCEEDANCE_THRESHOLD, 9.2103403719761836, rtol=1e-12)


class TestKnownAnswers:
    def test_mean_and_std_by_hand(self):
        # channel 0 = (1, 2, 3): mean 2, population std sqrt(2/3) = 0.8164966
        # channel 1 = (0, 0, 0): mean 0, std 0
        f = window_features([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert np.isclose(f[index_of("mean_ch0")], 2.0)
        assert np.isclose(f[index_of("std_ch0")], np.sqrt(2.0 / 3.0))
        assert np.isclose(f[index_of("mean_ch1")], 0.0)
        assert np.isclose(f[index_of("std_ch1")], 0.0)

    def test_slope_by_hand(self):
        # channel 0 = (1, 2, 3), t = (0, 1, 2), t_c = (-1, 0, 1), sum t_c^2 = 2
        # x_c = (-1, 0, 1) -> raw slope = (1 + 0 + 1)/2 = 1 per sample,
        # scaled by the window length 3 -> 3.
        f = window_features([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert np.isclose(f[index_of("slope_ch0")], 3.0)

    def test_max_abs_and_nis_by_hand(self):
        # rows (3, 4), (0, 0), (-5, 0): NIS = 25, 0, 25
        #   mean_nis = 50 / 3 = 16.6667, max_nis = 25
        #   max_abs_ch0 = 5, max_abs_ch1 = 4
        f = window_features([[3.0, 4.0], [0.0, 0.0], [-5.0, 0.0]])
        assert np.isclose(f[index_of("mean_nis")], 50.0 / 3.0)
        assert np.isclose(f[index_of("max_nis")], 25.0)
        assert np.isclose(f[index_of("max_abs_ch0")], 5.0)
        assert np.isclose(f[index_of("max_abs_ch1")], 4.0)

    def test_exceed_fraction_by_hand(self):
        # NIS values 25, 0, 25 against the 9.2103 threshold -> 2 of 3.
        f = window_features([[3.0, 4.0], [0.0, 0.0], [-5.0, 0.0]])
        assert np.isclose(f[index_of("exceed_frac")], 2.0 / 3.0)

    def test_cusum_range_by_hand(self):
        # channel 0 = (1, 2, 3): centred (-1, 0, 1), cumsum (-1, -1, 0)
        #   range = 0 - (-1) = 1, divided by sqrt(3) = 0.5773503
        f = window_features([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        assert np.isclose(f[index_of("cusum_range_ch0")], 1.0 / np.sqrt(3.0))

    def test_perfect_correlation(self):
        f = window_features([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [-1.0, -2.0]])
        assert np.isclose(f[index_of("corr_01")], 1.0)

    def test_perfect_anticorrelation(self):
        f = window_features([[1.0, -2.0], [2.0, -4.0], [3.0, -6.0], [-1.0, 2.0]])
        assert np.isclose(f[index_of("corr_01")], -1.0)

    def test_constant_channel_gives_zero_correlation_not_nan(self):
        f = window_features([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        assert f[index_of("corr_01")] == 0.0

    def test_lag1_autocorrelation_of_an_alternating_sequence(self):
        # x = (1, -1, 1, -1): mean 0, lag-1 products sum = -3, denominator 4
        # -> rho = -0.75
        f = window_features([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        assert np.isclose(f[index_of("autocorr1_ch0")], -0.75)

    def test_constant_channel_gives_zero_autocorrelation(self):
        f = window_features([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
        assert f[index_of("autocorr1_ch0")] == 0.0


class TestExpectations:
    def test_standard_normal_window_hits_the_documented_expectations(self):
        r = np.random.default_rng(0).standard_normal((20000, 2))
        f = window_features(r)
        assert abs(f[index_of("mean_ch0")]) < 0.03
        assert abs(f[index_of("std_ch0")] - 1.0) < 0.03
        assert abs(f[index_of("autocorr1_ch0")]) < 0.03
        assert abs(f[index_of("mean_nis")] - 2.0) < 0.05
        assert abs(f[index_of("corr_01")]) < 0.03
        assert abs(f[index_of("exceed_frac")] - 0.01) < 0.005


class TestValidation:
    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match=r"\(W, 2\)"):
            window_features(np.zeros((10, 3)))

    def test_rejects_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 3 samples"):
            window_features(np.zeros((2, 2)))

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="finite"):
            window_features([[1.0, 2.0], [np.inf, 0.0], [0.0, 0.0]])

    @pytest.mark.parametrize("kwargs", [{"window": 2}, {"stride": 0}])
    def test_feature_matrix_rejects_bad_parameters(self, kwargs):
        with pytest.raises(ValueError):
            feature_matrix(np.zeros((50, 2)), **{"window": 10, "stride": 1, **kwargs})

    def test_feature_matrix_rejects_wrong_width(self):
        with pytest.raises(ValueError, match=r"\(N, 2\)"):
            feature_matrix(np.zeros((50, 3)), 10)

    def test_feature_matrix_rejects_non_finite(self):
        r = np.zeros((50, 2))
        r[7, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            feature_matrix(r, 10)


class TestFeatureMatrix:
    def test_end_indices_and_shape(self):
        feats, ends = feature_matrix(np.random.default_rng(1).standard_normal((50, 2)), 10, 5)
        assert ends[0] == 9
        assert np.all(np.diff(ends) == 5)
        assert feats.shape == (ends.size, N_FEATURES)

    def test_start_below_the_window_is_clamped(self):
        _, ends = feature_matrix(np.zeros((30, 2)), 10, 1, start=0)
        assert ends[0] == 9

    def test_empty_when_the_sequence_is_shorter_than_the_window(self):
        feats, ends = feature_matrix(np.zeros((5, 2)), 10, 1)
        assert feats.shape == (0, N_FEATURES)
        assert ends.size == 0

    def test_fast_path_matches_the_reference_implementation(self):
        # The batched implementation is used everywhere for speed; this is the
        # test that licenses that.
        r = np.random.default_rng(2).standard_normal((400, 2))
        feats, ends = feature_matrix(r, 37, 3)
        for row, end in zip(feats, ends, strict=True):
            reference = window_features(r[end - 36 : end + 1])
            assert np.allclose(row, reference, atol=1e-12, rtol=1e-12)

    def test_fast_path_matches_on_degenerate_windows(self):
        r = np.zeros((60, 2))
        r[:, 0] = 3.0
        feats, ends = feature_matrix(r, 20, 7)
        for row, end in zip(feats, ends, strict=True):
            assert np.allclose(row, window_features(r[end - 19 : end + 1]), atol=1e-12)

    @settings(max_examples=15, deadline=None)
    @given(window=st.integers(3, 40), stride=st.integers(1, 9))
    def test_fast_path_agreement_over_random_window_and_stride(self, window, stride):
        r = np.random.default_rng(window * 100 + stride).standard_normal((120, 2))
        feats, ends = feature_matrix(r, window, stride)
        for row, end in zip(feats, ends, strict=True):
            assert np.allclose(
                row, window_features(r[end - window + 1 : end + 1]), atol=1e-11, rtol=1e-11
            )
