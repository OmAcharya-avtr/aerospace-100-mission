"""Tests for the synthetic combiner dataset and its splits."""

from __future__ import annotations

import numpy as np
import pytest

from aperturediv.datasets import (
    N_EXTRA_FEATURES,
    build_features,
    make_combiner_dataset,
    split_dataset,
)


class TestBuildFeatures:
    def test_shape(self):
        est = np.array([[1.0, 2.0, 4.0], [0.5, 0.25, 1.0]])
        sig = np.array([0.0, 1.0])
        feats = build_features(est, sig)
        assert feats.shape == (2, 3 + N_EXTRA_FEATURES)

    def test_known_answer(self):
        # Ihat = (1, e, e^2): log = (0, 1, 2), mean 1, population std sqrt(2/3).
        est = np.array([[1.0, np.e, np.e**2]])
        feats = build_features(est, np.array([0.5]))
        assert feats[0, 0] == pytest.approx(0.0, abs=1e-15)
        assert feats[0, 1] == pytest.approx(1.0, rel=1e-14)
        assert feats[0, 2] == pytest.approx(2.0, rel=1e-14)
        assert feats[0, 3] == pytest.approx(1.0, rel=1e-14)
        assert feats[0, 4] == pytest.approx(np.sqrt(2.0 / 3.0), rel=1e-12)
        assert feats[0, 5] == pytest.approx(0.5, rel=1e-15)

    def test_takes_no_truth_argument(self):
        import inspect

        names = set(inspect.signature(build_features).parameters)
        assert names == {"irradiance_estimated", "sigma_e"}
        assert not any("true" in n for n in names)

    def test_rejects_nonpositive_estimate(self):
        with pytest.raises(ValueError, match="strictly positive"):
            build_features(np.array([[0.0]]), np.array([0.0]))

    def test_rejects_wrong_sigma_shape(self):
        with pytest.raises(ValueError, match=r"shape \(n,\)"):
            build_features(np.ones((3, 2)), np.ones(2))

    def test_rejects_one_dimensional_estimate(self):
        with pytest.raises(ValueError, match=r"shape \(n, L\)"):
            build_features(np.ones(3), np.ones(3))


class TestMakeCombinerDataset:
    def test_shapes_and_metadata(self):
        d = make_combiner_dataset(2000, n_apertures=3, si=0.5, seed=1)
        assert len(d) == 2000
        assert d.n_apertures == 3
        assert d.irradiance_true.shape == (2000, 3)
        assert d.irradiance_estimated.shape == (2000, 3)
        assert d.features.shape == (2000, 3 + N_EXTRA_FEATURES)
        assert d.weights_optimal.shape == (2000, 3)
        assert d.log_correlation.shape == (3, 3)
        assert d.si == 0.5
        assert d.seed == 1

    def test_optimal_weights_are_unit_norm(self):
        d = make_combiner_dataset(1000, seed=2)
        assert np.allclose(np.linalg.norm(d.weights_optimal, axis=1), 1.0, rtol=1e-12)

    def test_optimal_weights_are_the_mrc_direction(self):
        d = make_combiner_dataset(500, seed=3)
        h = d.amplitude_true
        assert np.allclose(
            d.weights_optimal, h / np.linalg.norm(h, axis=1, keepdims=True), rtol=1e-12
        )

    def test_features_reproducible_from_the_estimate(self):
        d = make_combiner_dataset(1500, seed=4)
        assert np.array_equal(build_features(d.irradiance_estimated, d.sigma_e), d.features)

    def test_sigma_e_inside_the_requested_range(self):
        d = make_combiner_dataset(3000, sigma_e_range=(0.2, 1.1), seed=5)
        assert d.sigma_e.min() >= 0.2
        assert d.sigma_e.max() <= 1.1

    def test_zero_error_range_reproduces_the_truth(self):
        d = make_combiner_dataset(400, sigma_e_range=(0.0, 0.0), seed=6)
        assert np.allclose(d.irradiance_estimated, d.irradiance_true, rtol=1e-12)

    def test_marginals(self):
        d = make_combiner_dataset(200_000, si=0.6, seed=7)
        assert np.allclose(d.irradiance_true.mean(axis=0), 1.0, atol=0.02)

    def test_sigma_e_db_property(self):
        d = make_combiner_dataset(100, seed=8)
        assert np.allclose(d.sigma_e_db / d.sigma_e, 10.0 / np.log(10.0), rtol=1e-12)

    def test_deterministic_in_seed(self):
        a = make_combiner_dataset(500, seed=9)
        b = make_combiner_dataset(500, seed=9)
        assert np.array_equal(a.irradiance_true, b.irradiance_true)
        assert np.array_equal(a.irradiance_estimated, b.irradiance_estimated)

    def test_different_seed_gives_different_data(self):
        a = make_combiner_dataset(500, seed=9)
        b = make_combiner_dataset(500, seed=10)
        assert not np.array_equal(a.irradiance_true, b.irradiance_true)

    def test_wider_spacing_lowers_correlation(self):
        tight = make_combiner_dataset(100, aperture_spacing_m=0.02, seed=11)
        wide = make_combiner_dataset(100, aperture_spacing_m=0.30, seed=11)
        assert wide.log_correlation[0, 1] < tight.log_correlation[0, 1]

    @pytest.mark.parametrize("n", [0, -3])
    def test_rejects_bad_n(self, n):
        with pytest.raises(ValueError, match="n_samples"):
            make_combiner_dataset(n)

    def test_rejects_bad_sigma_range(self):
        with pytest.raises(ValueError, match="sigma_e_range"):
            make_combiner_dataset(100, sigma_e_range=(1.0, 0.5))

    def test_rejects_negative_sigma_range(self):
        with pytest.raises(ValueError, match="sigma_e_range"):
            make_combiner_dataset(100, sigma_e_range=(-0.1, 0.5))


class TestSplitDataset:
    def test_sizes_and_disjointness(self):
        d = make_combiner_dataset(1000, seed=12)
        tr, va, te = split_dataset(d, n_train=600, n_validation=200)
        assert (len(tr), len(va), len(te)) == (600, 200, 200)
        rows = np.vstack([tr.irradiance_true, va.irradiance_true, te.irradiance_true])
        assert np.array_equal(rows, d.irradiance_true)

    def test_metadata_carried_through(self):
        d = make_combiner_dataset(1000, si=0.4, seed=13)
        tr, _, _ = split_dataset(d, n_train=500, n_validation=200)
        assert tr.si == 0.4
        assert tr.seed == 13
        assert tr.log_correlation.shape == d.log_correlation.shape

    def test_subset_keeps_feature_consistency(self):
        d = make_combiner_dataset(1000, seed=14)
        _, _, te = split_dataset(d, n_train=600, n_validation=200)
        assert np.array_equal(build_features(te.irradiance_estimated, te.sigma_e), te.features)

    def test_rejects_oversized_split(self):
        d = make_combiner_dataset(100, seed=15)
        with pytest.raises(ValueError, match="leaves no test rows"):
            split_dataset(d, n_train=90, n_validation=10)

    @pytest.mark.parametrize("tr,va", [(0, 10), (10, 0), (-1, 10)])
    def test_rejects_bad_sizes(self, tr, va):
        d = make_combiner_dataset(100, seed=16)
        with pytest.raises(ValueError, match=">= 1"):
            split_dataset(d, n_train=tr, n_validation=va)
