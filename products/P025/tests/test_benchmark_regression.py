"""Pinned seeded values and a wall-clock budget check.

Every number below was produced by running this repository in the build
session and is pinned so that an unintended change to the dynamics, the
filter, the feature definitions or the signature construction fails a test
rather than quietly moving a plot.  If a change here is intended, update the
constants **and** rerun the validation scripts, because the published numbers
move with them.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from fdiscope.analytic import (
    chi2_threshold,
    cusum_threshold_for_arl0,
    normalised_bias_signature,
)
from fdiscope.detectors import ChiSquaredDetector, CusumDetector, detection_delay
from fdiscope.evaluate import BenchmarkConfig, build_default_bank
from fdiscope.faults import FaultSpec, FaultType
from fdiscope.features import window_features
from fdiscope.plant import PlantConfig, loop_matrices
from fdiscope.simulate import LoopConfig, build_filter, simulate_loop

REFERENCE_SEED = 424242
REFERENCE_STEPS = 2000
BIAS_ONSET = 800

# --- pinned reference values, measured on this repository -------------------
HEALTHY_MEAN_NIS = 2.023532661040
HEALTHY_LAST_RESIDUAL = (0.33426130417330774, -0.00936914377697487)
HEALTHY_TORQUE_RMS_NM = 2.328015136837e-03
INNOVATION_COV = (
    (7.828273532189517e-07, 3.40286880177596e-09),
    (3.40286880177596e-09, 1.2312920752839758e-07),
)
FAULTED_MEAN_NIS = 17.775459331005
CUSUM_THRESHOLD = 5.020236626686
CUSUM_MU = 3.999962006045
CUSUM_DIRECTION = (-0.6837331650427059, 0.7297321145603255)
CUSUM_DELAY = 2.0
CHI2_DELAY = 2.0
WINDOW_FEATURES = (
    -1.748155708,
    1.238919827,
    -2.331956902,
    0.403077109,
    4.967565962,
    3.520309771,
    3.095850286,
    1.110233694,
    -1.503534163,
    0.102927428,
    5.966766504,
    2.319630296,
    15.407878567,
    39.01452122,
    0.215909273,
    0.78,
)
GRAM_ROW_SENSOR_BIAS = (
    1.0,
    -0.308420117,
    -0.569017422,
    0.413149991,
    -0.235016859,
    0.248394249,
    -0.255433976,
)
WORST_SIGNATURE_COSINE = 0.996620026


@pytest.fixture(scope="module")
def healthy_run():
    return simulate_loop(LoopConfig(n_steps=REFERENCE_STEPS, seed=REFERENCE_SEED))


@pytest.fixture(scope="module")
def bias_setup():
    plant = PlantConfig()
    kf = build_filter(loop_matrices(plant))
    bias = 4.0 * float(np.sqrt(plant.gyro_var_rad2_s2))
    direction, mu = normalised_bias_signature(kf, [0.0, bias])
    run = simulate_loop(
        LoopConfig(n_steps=REFERENCE_STEPS, seed=REFERENCE_SEED),
        FaultSpec(FaultType.SENSOR_BIAS, BIAS_ONSET, bias, 1),
    )
    return run, direction, mu


class TestPinnedSimulation:
    def test_healthy_mean_nis(self, healthy_run):
        assert np.isclose(healthy_run.nis.mean(), HEALTHY_MEAN_NIS, rtol=1e-11)

    def test_healthy_last_residual(self, healthy_run):
        assert np.allclose(healthy_run.residual[-1], HEALTHY_LAST_RESIDUAL, rtol=1e-11)

    def test_healthy_torque_rms(self, healthy_run):
        rms = float(np.sqrt(np.mean(healthy_run.u_cmd_nm**2)))
        assert np.isclose(rms, HEALTHY_TORQUE_RMS_NM, rtol=1e-11)

    def test_innovation_covariance(self, healthy_run):
        assert np.allclose(healthy_run.innovation_cov, INNOVATION_COV, rtol=1e-11)

    def test_faulted_mean_nis(self, bias_setup):
        run, _, _ = bias_setup
        assert np.isclose(run.nis[BIAS_ONSET:].mean(), FAULTED_MEAN_NIS, rtol=1e-10)


class TestPinnedDesign:
    def test_cusum_design(self, bias_setup):
        _, direction, mu = bias_setup
        assert np.isclose(mu, CUSUM_MU, rtol=1e-11)
        assert np.allclose(direction, CUSUM_DIRECTION, rtol=1e-11)
        assert np.isclose(cusum_threshold_for_arl0(2000.0, mu), CUSUM_THRESHOLD, rtol=1e-10)

    def test_chi2_threshold_is_stable(self):
        # 50 degrees of freedom at alpha = 1e-3.
        assert np.isclose(chi2_threshold(1e-3, 50), 86.66081519040317, rtol=1e-12)


class TestPinnedDetection:
    def test_cusum_delay(self, bias_setup):
        run, direction, mu = bias_setup
        det = CusumDetector(direction=direction, mu=mu, threshold=CUSUM_THRESHOLD)
        assert detection_delay(det.run(run.residual).alarm, BIAS_ONSET) == CUSUM_DELAY

    def test_chi2_delay(self, bias_setup):
        run, _, _ = bias_setup
        chi = ChiSquaredDetector(window=25, dim=2, alpha=1e-3)
        assert detection_delay(chi.run(run.residual).alarm, BIAS_ONSET) == CHI2_DELAY


class TestPinnedFeatures:
    def test_window_features(self, bias_setup):
        run, _, _ = bias_setup
        feats = window_features(run.residual[BIAS_ONSET : BIAS_ONSET + 100])
        assert np.allclose(feats, WINDOW_FEATURES, atol=1e-8)


class TestPinnedSignatures:
    def test_gram_row_and_worst_pair(self):
        bank = build_default_bank(BenchmarkConfig(), n_onsets=8)
        gram = bank.gram()
        assert np.allclose(gram[0], GRAM_ROW_SENSOR_BIAS, atol=1e-8)
        worst = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
        assert np.isclose(worst, WORST_SIGNATURE_COSINE, atol=1e-8)

    def test_the_worst_pair_is_the_two_actuator_faults(self):
        bank = build_default_bank(BenchmarkConfig(), n_onsets=8)
        gram = np.abs(bank.gram() - np.eye(7))
        i, j = np.unravel_index(int(np.argmax(gram)), gram.shape)
        pair = {bank.faults[i], bank.faults[j]}
        assert pair == {FaultType.ACTUATOR_STUCK, FaultType.ACTUATOR_RUNAWAY}


class TestComputeBudget:
    def test_one_run_is_fast_enough(self):
        # Measured about 0.018 s per 2000-step run on the 2-core build machine;
        # the bound is loose so that a slower CI box does not fail the suite,
        # but tight enough to catch an order-of-magnitude regression.
        cfg = LoopConfig(n_steps=2000, seed=1)
        simulate_loop(cfg)  # warm the DARE solver's imports
        start = time.perf_counter()
        for i in range(10):
            simulate_loop(LoopConfig(n_steps=2000, seed=i))
        per_run = (time.perf_counter() - start) / 10.0
        assert per_run < 0.25, f"{per_run:.4f} s per run"

    def test_signature_bank_build_is_fast_enough(self):
        # Measured about 2.2 s for the default 8-onset, 100-sample bank.
        start = time.perf_counter()
        build_default_bank(BenchmarkConfig(), n_onsets=8)
        elapsed = time.perf_counter() - start
        assert elapsed < 20.0, f"{elapsed:.2f} s"
