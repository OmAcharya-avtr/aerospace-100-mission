"""Tests for the synthetic telemetry generator."""

from __future__ import annotations

import numpy as np
import pytest

from telemetryool.synthetic import (
    ANOMALY_KINDS,
    Anomaly,
    NominalModel,
    apply_anomaly,
    equicorrelation,
    generate_anomalous,
    generate_nominal,
)


def test_equicorrelation_structure() -> None:
    mat = equicorrelation(3, 0.4)
    assert np.allclose(np.diag(mat), 1.0)
    assert mat[0, 1] == pytest.approx(0.4)
    assert np.allclose(mat, mat.T)
    np.linalg.cholesky(mat)


def test_equicorrelation_rejects_non_positive_definite() -> None:
    with pytest.raises(ValueError, match="non-positive-definite"):
        equicorrelation(4, -0.5)
    with pytest.raises(ValueError, match="n_channels must be >= 1"):
        equicorrelation(0, 0.1)


def test_nominal_is_unit_variance_without_serial_correlation() -> None:
    rng = np.random.default_rng(0)
    block = generate_nominal(NominalModel(3), 2000, 100, rng)
    assert block.shape == (2000, 100, 3)
    # 600000 samples per channel: the standard error of the sample sd is
    # 1 / sqrt(2 N) = 0.00091, so a 5-sigma band is 1 +/- 0.0046.
    assert np.allclose(block.std(axis=(0, 1)), 1.0, atol=0.005)
    assert np.allclose(block.mean(axis=(0, 1)), 0.0, atol=0.01)


def test_nominal_is_unit_variance_with_serial_correlation() -> None:
    """The AR(1) parameterisation e_t = rho e_{t-1} + sqrt(1 - rho^2) n_t keeps
    the marginal variance at 1 for any rho, and the lag-1 autocorrelation at rho."""
    rng = np.random.default_rng(1)
    block = generate_nominal(NominalModel(2, rho_time=0.6), 1500, 100, rng)
    assert np.allclose(block.std(axis=(0, 1)), 1.0, atol=0.01)
    lag1 = np.corrcoef(block[:, :-1, 0].ravel(), block[:, 1:, 0].ravel())[0, 1]
    assert lag1 == pytest.approx(0.6, abs=0.01)


def test_cross_correlation_is_reproduced() -> None:
    rng = np.random.default_rng(2)
    block = generate_nominal(NominalModel(4, correlation=equicorrelation(4, 0.6)), 2000, 80, rng)
    corr = np.corrcoef(block.reshape(-1, 4).T)
    off = corr[~np.eye(4, dtype=bool)]
    assert np.allclose(off, 0.6, atol=0.01)


def test_generation_is_deterministic_for_a_fixed_seed() -> None:
    model = NominalModel(3, rho_time=0.4)
    a = generate_nominal(model, 20, 30, np.random.default_rng(99))
    b = generate_nominal(model, 20, 30, np.random.default_rng(99))
    assert np.array_equal(a, b)


def test_step_anomaly_offsets_only_the_named_channels_after_onset() -> None:
    rng = np.random.default_rng(3)
    base = generate_nominal(NominalModel(3), 10, 20, rng)
    out = apply_anomaly(base, Anomaly("step", 5, 1.5, channels=[1]), rng)
    diff = out - base
    assert np.allclose(diff[:, :5, :], 0.0)
    assert np.allclose(diff[:, 5:, 1], 1.5)
    assert np.allclose(diff[:, :, 0], 0.0)
    assert np.allclose(diff[:, :, 2], 0.0)


def test_drift_anomaly_is_a_ramp() -> None:
    """slope 0.05 from onset 5 over a window of 20: the offset at the last sample
    (index 19) is (19 - 5) * 0.05 = 0.7."""
    rng = np.random.default_rng(4)
    base = generate_nominal(NominalModel(1), 5, 20, rng)
    out = apply_anomaly(base, Anomaly("drift", 5, 0.05), rng)
    diff = (out - base)[0, :, 0]
    assert np.allclose(diff[:5], 0.0)
    assert diff[-1] == pytest.approx(0.7, abs=1e-12)
    assert np.allclose(np.diff(diff[5:]), 0.05)


def test_spike_anomaly_affects_exactly_one_sample() -> None:
    rng = np.random.default_rng(5)
    base = generate_nominal(NominalModel(2), 8, 15, rng)
    out = apply_anomaly(base, Anomaly("spike", 7, 4.0), rng)
    diff = out - base
    assert np.allclose(diff[:, 7, 0], 4.0)
    assert np.count_nonzero(diff) == 8


def test_stuck_anomaly_freezes_the_value() -> None:
    rng = np.random.default_rng(6)
    base = generate_nominal(NominalModel(2), 6, 20, rng)
    out = apply_anomaly(base, Anomaly("stuck", 10), rng)
    assert np.allclose(out[:, 10:, 0], base[:, 10, 0][:, None])
    assert np.allclose(out[:, :, 1], base[:, :, 1])
    # std of a frozen channel is zero up to the floating-point noise of the
    # variance computation itself, hence atol rather than an exact comparison.
    assert np.all(out[:, 10:, 0].std(axis=1) < 1e-15)


def test_decorrelate_leaves_marginals_and_breaks_correlation() -> None:
    rng = np.random.default_rng(7)
    model = NominalModel(4, correlation=equicorrelation(4, 0.7))
    base = generate_nominal(model, 2000, 60, rng)
    out = apply_anomaly(base, Anomaly("decorrelate", 20, channels=[0, 1]), rng)
    tail = out[:, 20:, :]
    assert np.allclose(tail[:, :, 0].std(), 1.0, atol=0.02)
    assert abs(np.corrcoef(tail[:, :, 0].ravel(), tail[:, :, 2].ravel())[0, 1]) < 0.03
    # the untouched channels keep their correlation
    assert np.corrcoef(tail[:, :, 2].ravel(), tail[:, :, 3].ravel())[0, 1] == pytest.approx(
        0.7, abs=0.03
    )


def test_generate_anomalous_shape_and_determinism() -> None:
    model = NominalModel(3)
    a = generate_anomalous(model, Anomaly("step", 4, 2.0), 7, 11, np.random.default_rng(8))
    b = generate_anomalous(model, Anomaly("step", 4, 2.0), 7, 11, np.random.default_rng(8))
    assert a.shape == (7, 11, 3)
    assert np.array_equal(a, b)


def test_model_validation() -> None:
    with pytest.raises(ValueError, match="n_channels must be an integer >= 1"):
        NominalModel(0)
    with pytest.raises(ValueError, match=r"rho_time must lie in \[0, 1\)"):
        NominalModel(2, rho_time=1.0)
    with pytest.raises(ValueError, match="burn_in must be >= 0"):
        NominalModel(2, burn_in=-1)
    with pytest.raises(ValueError, match="correlation must have shape"):
        NominalModel(2, correlation=np.eye(3))
    with pytest.raises(ValueError, match="must be symmetric"):
        NominalModel(2, correlation=np.array([[1.0, 0.5], [0.2, 1.0]]))
    with pytest.raises(ValueError, match="unit diagonal"):
        NominalModel(2, correlation=np.array([[2.0, 0.0], [0.0, 2.0]]))
    with pytest.raises(ValueError, match="positive definite"):
        NominalModel(2, correlation=np.array([[1.0, 1.0], [1.0, 1.0]]))


def test_anomaly_validation() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        Anomaly("explode", 0, 1.0)
    with pytest.raises(ValueError, match="onset must be >= 0"):
        Anomaly("step", -1, 1.0)
    with pytest.raises(ValueError, match="takes no magnitude"):
        Anomaly("stuck", 0, 1.0)
    with pytest.raises(ValueError, match="requires a non-zero magnitude"):
        Anomaly("step", 0, 0.0)
    with pytest.raises(ValueError, match="out of range"):
        Anomaly("step", 0, 1.0, channels=[5]).channel_indices(3)
    with pytest.raises(ValueError, match="at least one channel"):
        Anomaly("step", 0, 1.0, channels=[]).channel_indices(3)


def test_apply_anomaly_validation() -> None:
    rng = np.random.default_rng(9)
    base = generate_nominal(NominalModel(2), 4, 10, rng)
    with pytest.raises(ValueError, match="must be 3-D"):
        apply_anomaly(base[0], Anomaly("step", 1, 1.0), rng)
    with pytest.raises(ValueError, match="outside a window"):
        apply_anomaly(base, Anomaly("step", 10, 1.0), rng)
    with pytest.raises(ValueError, match="must both be >= 1"):
        generate_nominal(NominalModel(2), 0, 10, rng)


def test_all_anomaly_kinds_are_applicable() -> None:
    rng = np.random.default_rng(10)
    base = generate_nominal(NominalModel(3), 5, 20, rng)
    for kind in ANOMALY_KINDS:
        magnitude = 0.0 if kind in ("stuck", "decorrelate") else 1.0
        out = apply_anomaly(base, Anomaly(kind, 5, magnitude), rng)
        assert out.shape == base.shape
        assert not np.array_equal(out, base)
