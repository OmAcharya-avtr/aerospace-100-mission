"""Synthetic pipeline population generation and splitting."""

from __future__ import annotations

import numpy as np
import pytest

from latencynet.dataset import (
    TARGET_PROBABILITIES,
    build_dataset,
    feature_matrix,
    generate_record,
    log_target,
)
from latencynet.features import N_FEATURES


def test_record_is_deterministic_in_seed():
    a = generate_record(0, "independent", 5, n_probe=32, n_reference=2000)
    b = generate_record(0, "independent", 5, n_probe=32, n_reference=2000)
    assert a.probe.features == b.probe.features
    assert a.reference_quantile_s == b.reference_quantile_s


def test_record_contents():
    rec = generate_record(3, "correlated", 7, n_probe=32, n_reference=2000)
    assert rec.index == 3
    assert rec.regime == "correlated"
    assert 2 <= rec.n_stages <= 6
    assert rec.latent_rho > 0.0
    assert set(rec.reference_quantile_s) == set(TARGET_PROBABILITIES)
    assert rec.reference_quantile_s[0.999] > rec.reference_quantile_s[0.99]
    assert rec.n_reference == 2000
    assert rec.spec().n_stages == rec.n_stages


def test_independent_regime_has_zero_rho():
    for i in range(6):
        rec = generate_record(i, "independent", 100 + i, n_probe=32, n_reference=2000)
        assert rec.latent_rho == 0.0


def test_reference_quantile_se_is_reported_and_small():
    rec = generate_record(0, "independent", 9, n_probe=32, n_reference=40_000)
    for p in TARGET_PROBABILITIES:
        se = rec.reference_quantile_se_s[p]
        assert np.isfinite(se)
        # At n = 40000 the relative uncertainty of p99 is under 2 % and of
        # p99.9 under 6 %; both are reported, never assumed away.
        assert 0.0 < se / rec.reference_quantile_s[p] < 0.06


def test_reference_quantile_se_is_nan_when_unresolvable():
    # The one-sigma rank interval must fit inside the sample. For p = 0.9999
    # at n = 2000 the upper rank is 2000 * 0.9999 + sqrt(2000 * 0.9999 *
    # 0.0001) = 1999.8 + 0.45, which rounds up to 2001 and does not fit, so
    # the honest answer is nan rather than a number. generate_record cannot
    # reach this branch because n_reference >= 1000 always resolves p99.9, so
    # the private helper is exercised directly.
    from latencynet.dataset import _quantile_se

    assert np.isnan(_quantile_se(np.arange(1.0, 2001.0), 0.9999))
    assert np.isfinite(_quantile_se(np.arange(1.0, 2001.0), 0.999))


def test_probe_and_reference_use_different_streams():
    rec = generate_record(0, "independent", 11, n_probe=2000, n_reference=2000)
    # Probe mean and reference mean are independent estimates of the same
    # quantity; they must be close but must not be identical.
    assert sum(rec.probe.stage_mean_s) != rec.reference_mean_s
    assert sum(rec.probe.stage_mean_s) == pytest.approx(rec.reference_mean_s, rel=0.1)


def test_build_dataset_splits(tiny_dataset):
    assert len(tiny_dataset.train) == 32
    assert len(tiny_dataset.calibration) == 14
    assert len(tiny_dataset.test) == 14
    assert tiny_dataset.n_total == 60
    indices = [r.index for r in tiny_dataset.all_records()]
    assert indices == sorted(indices)
    assert len(set(indices)) == 60


def test_feature_matrix_and_target(tiny_dataset):
    x = feature_matrix(tiny_dataset.test)
    y = log_target(tiny_dataset.test, 0.99)
    assert x.shape == (14, N_FEATURES)
    assert y.shape == (14,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))


def test_test_split_gets_the_larger_reference_sample():
    ds = build_dataset(
        "independent",
        n_train=2,
        n_calibration=1,
        n_test=1,
        seed=3,
        n_probe=16,
        n_reference=2000,
        n_reference_test=8000,
    )
    assert ds.train[0].n_reference == 2000
    assert ds.test[0].n_reference == 8000


def test_validation():
    with pytest.raises(ValueError, match="regime must be"):
        generate_record(0, "mixed", 1)
    with pytest.raises(ValueError, match="n_probe"):
        generate_record(0, "independent", 1, n_probe=4)
    with pytest.raises(ValueError, match="n_reference"):
        generate_record(0, "independent", 1, n_reference=10)
    with pytest.raises(ValueError, match="n_stages_range"):
        generate_record(0, "independent", 1, n_reference=2000, n_stages_range=(5, 2))
    with pytest.raises(ValueError, match="regime must be"):
        build_dataset("nonsense")
    with pytest.raises(ValueError, match="n_train"):
        build_dataset("independent", n_train=0)
    with pytest.raises(ValueError, match="non-empty"):
        feature_matrix(())
    with pytest.raises(ValueError, match="non-empty"):
        log_target((), 0.99)


def test_log_target_rejects_an_unavailable_probability(tiny_dataset):
    with pytest.raises(ValueError, match="no reference quantile"):
        log_target(tiny_dataset.test, 0.95)
