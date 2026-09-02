"""Regression suite: pinned seeded outputs.

Every number below was produced by this repository on 2026-09-02 with
Python 3.11.15, numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0 on two x86-64
cores, and is pinned so that a change in behaviour has to be deliberate.
Tolerances are tight where the quantity is a closed-form or purely linear
result and looser where a long simulation accumulates round-off, and each
relaxation says why.
"""

import numpy as np
import pytest

from cmgsteer.arrays import pyramid_array
from cmgsteer.dataset import generate_policy_dataset, manoeuvre_suite
from cmgsteer.ml import LearnedNullMotion
from cmgsteer.nullmotion import GradientNullMotion
from cmgsteer.simulate import run_steering
from cmgsteer.singularity import momentum_envelope, singularity_measure
from cmgsteer.steering import sr_inverse_steer

SEED = 20260902
D_TEST = np.array([0.4, -0.2, 0.7, 0.1])
TAU = np.array([0.10, -0.05, 0.20])


@pytest.fixture(scope="module")
def suite():
    return manoeuvre_suite(
        pyramid_array(), 3, seed=SEED, n_segments=3, segment_duration=5.0, dt=0.02
    )


class TestPinnedGeometry:
    def test_measure_at_the_origin(self):
        assert singularity_measure(pyramid_array().jacobian(np.zeros(4))) == pytest.approx(
            1.152, abs=1e-13
        )

    def test_envelope_extremes(self):
        momenta, _ = momentum_envelope(pyramid_array(), n_points=1024)
        radii = np.linalg.norm(momenta, axis=1)
        assert float(radii.max()) == pytest.approx(3.298479922877764, rel=1e-9)
        assert float(radii.min()) == pytest.approx(2.9868188204267803, rel=1e-9)


class TestPinnedSRClosedForm:
    @pytest.mark.parametrize(
        ("lam", "expected"),
        [
            (1e-6, 8.286174745999156e-07),
            (1e-3, 8.240351038057901e-04),
            (1e-1, 5.336615612172257e-02),
        ],
    )
    def test_torque_error_norm(self, lam, expected):
        result = sr_inverse_steer(pyramid_array(), D_TEST, TAU, lam=lam)
        assert result.torque_error_norm == pytest.approx(expected, rel=1e-11)


class TestPinnedManoeuvre:
    def test_suite_start_and_peak(self, suite):
        assert np.allclose(
            suite.initial_deltas[0],
            [0.076735583997935, -0.121821385165675, 0.016472191464704, 0.014381547282059],
            atol=1e-14,
        )
        assert suite.profiles[0].peak_momentum == pytest.approx(2.5423154052216606, rel=1e-12)

    # Long explicit-Euler runs accumulate round-off over 750 steps, so these are
    # pinned to 1e-8 relative rather than to machine precision.
    @pytest.mark.parametrize(
        ("method", "rms", "path", "min_measure", "n_sat"),
        [
            ("pinv", 2.478642128067748e-01, 6.335720486729631e-01, 1.413220501699178e-02, 21),
            ("sr", 7.965322189970177e-02, 3.379071964501034e-01, 2.341074957689643e-02, 14),
            ("gsr", 7.815992652752507e-02, 3.340543648718894e-01, 2.296943831832126e-02, 13),
        ],
    )
    def test_pinned_run(self, suite, method, rms, path, min_measure, n_sat):
        history = run_steering(
            pyramid_array(),
            suite.initial_deltas[0],
            suite.profiles[0],
            method=method,
            max_gimbal_rate=2.0,
        )
        assert history.rms_torque_error == pytest.approx(rms, rel=1e-8)
        assert history.total_momentum_error_path == pytest.approx(path, rel=1e-8)
        assert history.min_measure == pytest.approx(min_measure, rel=1e-8)
        assert history.n_rate_limited == n_sat

    def test_pinned_gradient_null_motion_run(self, suite):
        history = run_steering(
            pyramid_array(),
            suite.initial_deltas[0],
            suite.profiles[0],
            method="sr",
            max_gimbal_rate=2.0,
            null_policy=GradientNullMotion(gain=1.0, max_rate=0.5),
        )
        assert history.total_momentum_error_path == pytest.approx(
            5.757828466981026e-02, rel=1e-8
        )
        assert history.min_measure == pytest.approx(9.482032839199543e-01, rel=1e-8)


class TestPinnedDataset:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        return generate_policy_dataset(
            pyramid_array(), 25, seed=4242, horizon=10, n_candidates=5, stride=29, n_manoeuvres=3
        )

    def test_feature_and_label_checksums(self, data):
        assert float(data.features.sum()) == pytest.approx(1.709495187753689e02, rel=1e-9)
        assert float(data.coefficients.sum()) == pytest.approx(1.213996906540890e00, rel=1e-8)

    def test_oracle_beats_zero_and_gradient_loses(self, data):
        # The pinned finding on this seed: the clairvoyant oracle improves on
        # the plain SR inverse by 20%, and the classical gradient policy is
        # worse than doing nothing at all.
        assert float(data.zero_scores.mean()) == pytest.approx(4.764659395480546e-04, rel=1e-8)
        assert float(data.best_scores.mean()) == pytest.approx(3.807598460805049e-04, rel=1e-8)
        assert float(data.gradient_scores.mean()) == pytest.approx(
            5.024113384935569e-04, rel=1e-8
        )
        assert data.best_scores.mean() < data.zero_scores.mean()
        assert data.gradient_scores.mean() > data.zero_scores.mean()

    def test_training_is_reproducible_from_the_seed(self, data):
        first = LearnedNullMotion(n_estimators=2, hidden_layer_sizes=(16,), max_iter=40)
        second = LearnedNullMotion(n_estimators=2, hidden_layer_sizes=(16,), max_iter=40)
        first.fit(data.features, data.coefficients)
        second.fit(data.features, data.coefficients)
        assert np.allclose(*[m.predict(data.features)[0] for m in (first, second)])
