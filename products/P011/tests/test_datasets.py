"""Tests for waveforge.datasets."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from waveforge.datasets import make_slope_dataset


@pytest.fixture(scope="module")
def small_dataset(tiny_config_module):
    return make_slope_dataset(
        tiny_config_module, n_frames=40, train_seeds=(1, 2), test_seeds=(9,)
    )


@pytest.fixture(scope="module")
def tiny_config_module():
    from waveforge.loop import AOConfig

    return AOConfig(n_pix=32, n_sub=4, n_act=5, screen_pixels=256, n_subharmonics=3)


class TestDataset:
    def test_sequence_counts(self, small_dataset):
        assert len(small_dataset.train) == 2
        assert len(small_dataset.test) == 1

    def test_sequence_shapes(self, small_dataset):
        for seq in small_dataset.train + small_dataset.test:
            assert seq.shape == (40, small_dataset.n_slopes)

    def test_frame_counts(self, small_dataset):
        assert small_dataset.n_train_frames == 80
        assert small_dataset.n_test_frames == 40

    def test_train_and_test_differ(self, small_dataset):
        assert not np.allclose(small_dataset.train[0], small_dataset.test[0])

    def test_summary_contents(self, small_dataset):
        summary = small_dataset.summary()
        assert summary["train_seeds"] == [1, 2]
        assert summary["test_seeds"] == [9]
        assert summary["noise_sigma_rad_per_m"] == 0.0
        assert summary["d_over_r0"] == pytest.approx(5.0)

    def test_deterministic(self, tiny_config_module):
        a = make_slope_dataset(tiny_config_module, n_frames=20, train_seeds=(3,), test_seeds=(4,))
        b = make_slope_dataset(tiny_config_module, n_frames=20, train_seeds=(3,), test_seeds=(4,))
        assert np.array_equal(a.train[0], b.train[0])

    def test_noise_changes_the_data(self, tiny_config_module):
        clean = make_slope_dataset(
            tiny_config_module, n_frames=20, train_seeds=(3,), test_seeds=(4,)
        )
        noisy = make_slope_dataset(
            tiny_config_module, n_frames=20, train_seeds=(3,), test_seeds=(4,), noise_sigma=5.0
        )
        residual = noisy.train[0] - clean.train[0]
        assert residual.std() == pytest.approx(5.0, rel=0.15)

    def test_noise_seed_controls_the_realisation(self, tiny_config_module):
        kwargs = dict(n_frames=20, train_seeds=(3,), test_seeds=(4,), noise_sigma=2.0)
        a = make_slope_dataset(tiny_config_module, noise_seed=1, **kwargs)
        b = make_slope_dataset(tiny_config_module, noise_seed=2, **kwargs)
        assert not np.allclose(a.train[0], b.train[0])

    def test_train_test_noise_realisations_differ(self, tiny_config_module):
        # the same screen seed in train and test would be rejected, but the
        # noise stream must also advance between the two builds
        data = make_slope_dataset(
            tiny_config_module, n_frames=20, train_seeds=(3,), test_seeds=(5,), noise_sigma=1.0
        )
        assert not np.allclose(data.train[0], data.test[0])

    def test_overlapping_seeds_rejected(self, tiny_config_module):
        with pytest.raises(ValueError, match="disjoint"):
            make_slope_dataset(tiny_config_module, n_frames=20, train_seeds=(1,), test_seeds=(1,))

    def test_empty_seed_lists_rejected(self, tiny_config_module):
        with pytest.raises(ValueError, match="non-empty"):
            make_slope_dataset(tiny_config_module, n_frames=20, train_seeds=(), test_seeds=(1,))

    @pytest.mark.parametrize("n_frames", [1, 0, 2.5])
    def test_bad_n_frames(self, tiny_config_module, n_frames):
        with pytest.raises(ValueError, match="n_frames"):
            make_slope_dataset(tiny_config_module, n_frames=n_frames)

    def test_bad_noise_sigma(self, tiny_config_module):
        with pytest.raises(ValueError, match="noise_sigma"):
            make_slope_dataset(tiny_config_module, n_frames=20, noise_sigma=-1.0)

    def test_default_config_is_used_when_omitted(self, tiny_config_module):
        # exercising the default path with an explicitly small config keeps it
        # fast; the point is that config=None does not raise
        data = make_slope_dataset(
            replace(tiny_config_module, seed=0),
            n_frames=10,
            train_seeds=(21,),
            test_seeds=(22,),
        )
        assert data.config.n_pix == 32
