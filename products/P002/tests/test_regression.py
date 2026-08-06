"""Regression tests against pinned seeded outputs.

Every value in ``PINNED`` was produced by running this package in the build
session on Python 3.11.15 / numpy 2.4.4 / scipy 1.17.1 and is reproduced by
``validation/v6_regression_baseline.py``. A change here means the numerical
behaviour of the package changed: investigate before re-pinning.

Tolerances: exact for integers and pure-geometry quantities; 1e-9 relative
for deterministic float pipelines; 1e-6 relative where BLAS/LAPACK ordering
can move the last bits (LQR Riccati solution).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from trackforge.control import (
    LQRController,
    PIDController,
    lqr_weights_from_bandwidth,
    pid_gains_from_bandwidth,
    step_response,
)
from trackforge.dynamics import GimbalAxis, JitterPSD, synthesize_jitter
from trackforge.reacq import (
    AlwaysFullPolicy,
    AlwaysLocalPolicy,
    ReacqConfig,
    evaluate_policy,
    train_q_learning,
)
from trackforge.scan import (
    GaussianUncertainty,
    coverage_fraction,
    expected_acquisition_time_spiral,
    raster_scan,
    simulate_acquisition,
    spiral_scan,
)
from trackforge.sim import DEFAULT_SCENARIO, run_episode

PINNED = {
    "spiral_n_points": 9987,
    "spiral_max_radius": 9.765741784312373e-4,
    "spiral_track_spacing": 3.0000000000000004e-05,
    "spiral_scan_speed": 0.010001857529517351,
    "spiral_coverage": 0.99465,
    "raster_n_points": 13199,
    "acq_time_seed7": 0.58,
    "expected_acq_time": 2.943899355196817,
    "jitter_std": 2.0244600254078307e-06,
    "jitter_first": 7.103931036519614e-07,
    "jitter_last": 5.073865055327964e-07,
    "pd_overshoot": 0.04054674667531096,
    "pd_rise": 0.0689,
    "pd_peak": 0.14250000000000002,
    "lqr_gain": (49.34802200544659, 2.2015314988864194),
    "episode99_acq": 2.742,
    "episode99_rms": 1.4059145964416993e-06,
    "episode99_peak": 8.55564332117158e-05,
    "episode99_loss": 1.1840000000000002,
    "episode99_reacq": 2.2289623972172654,
    "episode99_attempts": 4,
    "episode99_total": 6.154962397217266,
    "q_sum_2000ep_seed777": -1924.1545316336012,
    "q_visits_2000ep_seed777": 6883,
    "full_mean_300ep_seed555": 8.248721293383273,
    "full_success_300ep_seed555": 0.8966666666666666,
    "local_mean_300ep_seed555": 6.259058156368156,
    "local_success_300ep_seed555": 0.85,
}


@pytest.fixture(scope="module")
def pattern():
    """Pinned reference spiral: sigma = 300 urad, R_beam = 20 urad, overlap 0.25."""
    u = GaussianUncertainty(3e-4)
    return u, spiral_scan(u, 2e-5, overlap=0.25, containment=0.995, dwell_time=1e-3)


# --------------------------------------------------------------------------
# scan geometry
# --------------------------------------------------------------------------
def test_regression_spiral_point_count(pattern):
    _, p = pattern
    assert p.n_points == PINNED["spiral_n_points"]


def test_regression_spiral_geometry(pattern):
    _, p = pattern
    assert p.max_radius == pytest.approx(PINNED["spiral_max_radius"], rel=1e-12)
    assert p.track_spacing == pytest.approx(PINNED["spiral_track_spacing"], rel=1e-12)
    assert p.scan_speed == pytest.approx(PINNED["spiral_scan_speed"], rel=1e-9)


def test_regression_raster_point_count():
    u = GaussianUncertainty(3e-4)
    assert raster_scan(u, 2e-5).n_points == PINNED["raster_n_points"]


def test_regression_spiral_coverage(pattern):
    u, p = pattern
    cov = coverage_fraction(p, u, n_samples=20000, rng=np.random.default_rng(123))
    assert cov == pytest.approx(PINNED["spiral_coverage"], abs=1e-12)


def test_regression_acquisition_time(pattern):
    _, p = pattern
    t = simulate_acquisition(
        p, np.array([1.5e-4, -2.0e-4]), p_dwell=0.9, rng=np.random.default_rng(7)
    )
    assert t == pytest.approx(PINNED["acq_time_seed7"], rel=1e-12)


def test_regression_expected_acquisition_time(pattern):
    u, p = pattern
    t = expected_acquisition_time_spiral(
        u, 2e-5, 0.25, p.scan_speed, containment=0.995, p_pass=0.9
    )
    assert t == pytest.approx(PINNED["expected_acq_time"], rel=1e-9)


# --------------------------------------------------------------------------
# jitter synthesis
# --------------------------------------------------------------------------
def test_regression_jitter_series():
    x = synthesize_jitter(JitterPSD(1e-12, 3.0, 2.0), 4096, 5000.0, np.random.default_rng(31))
    assert float(np.std(x)) == pytest.approx(PINNED["jitter_std"], rel=1e-12)
    assert float(x[0]) == pytest.approx(PINNED["jitter_first"], rel=1e-12)
    assert float(x[-1]) == pytest.approx(PINNED["jitter_last"], rel=1e-12)


# --------------------------------------------------------------------------
# control
# --------------------------------------------------------------------------
def test_regression_pd_step_metrics():
    kp, _, kd = pid_gains_from_bandwidth(0.05, 2 * math.pi * 5, 0.707)
    _, _, m = step_response(
        GimbalAxis(0.05, 0.02, 2.0, 1.0), PIDController(kp, 0.0, kd, 2.0), 1e-4, 1e-4, 1.0
    )
    assert m.overshoot == pytest.approx(PINNED["pd_overshoot"], rel=1e-9)
    assert m.rise_time == pytest.approx(PINNED["pd_rise"], rel=1e-9)
    assert m.peak_time == pytest.approx(PINNED["pd_peak"], rel=1e-9)


def test_regression_lqr_gain():
    q, qr, r = lqr_weights_from_bandwidth(0.05, 2 * math.pi * 5)
    lqr = LQRController(GimbalAxis(0.05, 0.02, 2.0, 1.0), q_angle=q, q_rate=qr, r_torque=r)
    assert lqr.gain[0] == pytest.approx(PINNED["lqr_gain"][0], rel=1e-6)
    assert lqr.gain[1] == pytest.approx(PINNED["lqr_gain"][1], rel=1e-6)


# --------------------------------------------------------------------------
# end-to-end episode
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def episode99():
    """Pinned end-to-end episode: default scenario, seed 99."""
    return run_episode(DEFAULT_SCENARIO, seed=99, keep_series=False)


def test_regression_episode_acquisition(episode99):
    assert episode99.acquisition_time_s == pytest.approx(PINNED["episode99_acq"], rel=1e-12)


def test_regression_episode_tracking(episode99):
    assert episode99.track_rms_rad == pytest.approx(PINNED["episode99_rms"], rel=1e-9)
    assert episode99.track_peak_rad == pytest.approx(PINNED["episode99_peak"], rel=1e-9)


def test_regression_episode_loss_time(episode99):
    assert episode99.loss_time_s == pytest.approx(PINNED["episode99_loss"], rel=1e-12)


def test_regression_episode_reacquisition(episode99):
    assert episode99.reacq_time_s == pytest.approx(PINNED["episode99_reacq"], rel=1e-9)
    assert episode99.reacq_attempts == PINNED["episode99_attempts"]
    assert episode99.reacq_success


def test_regression_episode_total(episode99):
    assert episode99.total_time_s == pytest.approx(PINNED["episode99_total"], rel=1e-9)


# --------------------------------------------------------------------------
# learning and policy evaluation
# --------------------------------------------------------------------------
def test_regression_q_table_checksum():
    pol = train_q_learning(ReacqConfig(), episodes=2000, seed=777)
    assert float(pol.q.sum()) == pytest.approx(PINNED["q_sum_2000ep_seed777"], rel=1e-9)
    assert int(pol.visits.sum()) == PINNED["q_visits_2000ep_seed777"]


def test_regression_baseline_full():
    r = evaluate_policy(AlwaysFullPolicy(), ReacqConfig(), n_episodes=300, seed=555)
    assert r["mean_time_s"] == pytest.approx(PINNED["full_mean_300ep_seed555"], rel=1e-9)
    assert r["success_rate"] == pytest.approx(PINNED["full_success_300ep_seed555"], rel=1e-12)


def test_regression_baseline_local():
    r = evaluate_policy(AlwaysLocalPolicy(), ReacqConfig(), n_episodes=300, seed=555)
    assert r["mean_time_s"] == pytest.approx(PINNED["local_mean_300ep_seed555"], rel=1e-9)
    assert r["success_rate"] == pytest.approx(PINNED["local_success_300ep_seed555"], rel=1e-12)
