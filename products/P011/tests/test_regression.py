"""Regression suite with pinned seeded outputs (Level 3 requirement).

Every number below was produced by this code in the 0.1.0 build session with
the seeds shown.  They are *pins*, not physics: a change here means the
numerical behaviour of the package changed, and the change must be explained
in CHANGELOG.md before the pin is updated.  Tolerances are set at the level of
double-precision reproducibility on the same platform, not at the level of
physical agreement (which is what ``validation/`` covers).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from waveforge.atmosphere import phase_screen
from waveforge.control import noise_variance_gain, stability_limit_gain
from waveforge.datasets import make_slope_dataset
from waveforge.errorbudget import ideal_filter_fitting_coefficient
from waveforge.loop import AOConfig, AOSystem
from waveforge.predictor import LinearSlopePredictor
from waveforge.statistics import noll_residual_variance, total_phase_variance

PINNED_CONFIG = AOConfig(
    n_pix=32, n_sub=4, n_act=5, screen_pixels=256, n_subharmonics=3, seed=11
)


class TestAnalyticPins:
    def test_total_phase_variance(self):
        assert total_phase_variance() == pytest.approx(1.032765212209, rel=1e-10)

    @pytest.mark.parametrize(
        ("j", "expected"),
        [
            (1, 1.032765212209),
            (2, 0.583736845702),
            (4, 0.111482874031),
            (11, 0.037802946104),
            (21, 0.020839134438),
        ],
    )
    def test_noll_residual_variance(self, j, expected):
        assert noll_residual_variance(j) == pytest.approx(expected, rel=1e-9)

    def test_ideal_fitting_coefficient(self):
        assert ideal_filter_fitting_coefficient() == pytest.approx(0.274122117297, rel=1e-10)

    @pytest.mark.parametrize(
        ("delay", "expected"),
        [(1, 2.0), (2, 1.0), (3, 0.618034), (4, 0.445042)],
    )
    def test_stability_limits(self, delay, expected):
        assert stability_limit_gain(delay) == pytest.approx(expected, abs=1e-6)

    def test_noise_variance_gain(self):
        assert noise_variance_gain(0.4, 2) == pytest.approx(0.388889, abs=1e-6)


class TestScreenPins:
    """Seed 42, 16 x 16 samples at 20 mm, r0 = 0.1 m, three subharmonic levels."""

    @pytest.fixture(scope="class")
    @staticmethod
    def screen():
        return phase_screen(16, 0.02, 0.1, n_subharmonics=3, rng=42)

    def test_variance(self, screen):
        assert np.var(screen) == pytest.approx(1.425359342574, rel=1e-10)

    def test_corner_sample(self, screen):
        assert screen[0, 0] == pytest.approx(-1.944378378078, rel=1e-10)

    def test_centre_sample(self, screen):
        assert screen[8, 8] == pytest.approx(0.943435300235, rel=1e-10)

    def test_piston_removed(self, screen):
        assert screen.sum() == pytest.approx(0.0, abs=1e-10)


class TestSystemPins:
    @pytest.fixture(scope="class")
    @staticmethod
    def system():
        return AOSystem(PINNED_CONFIG)

    def test_geometry_counts(self, system):
        assert system.sensor.n_valid == 12
        assert system.mirror.n_actuators == 49
        assert system.n_controlled_modes == 22

    def test_interaction_matrix_norm(self, system):
        assert np.linalg.norm(system.interaction_matrix) == pytest.approx(
            42.082451925586, rel=1e-9
        )

    def test_reconstructor_norm(self, system):
        assert np.linalg.norm(system.reconstructor) == pytest.approx(11.020278557806, rel=1e-8)

    def test_error_budget_total(self, system):
        assert system.error_budget().total == pytest.approx(0.877659523252, rel=1e-9)

    def test_noise_terms_with_finite_flux(self):
        noisy = AOSystem(replace(PINNED_CONFIG, photon_flux=500.0, read_noise_e=1.0))
        assert noisy.sensor.slope_noise_sigma() == pytest.approx(5.975733720669, rel=1e-9)
        assert noisy.error_budget().noise == pytest.approx(0.449140071971, rel=1e-8)


class TestClosedLoopPins:
    """150 frames, 50 warm-up, noise seed 777."""

    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        return AOSystem(PINNED_CONFIG).run(150, warmup_frames=50, rng=777)

    def test_open_loop_variance(self, result):
        assert result.mean_open_loop_variance == pytest.approx(9.438153065281, rel=1e-9)

    def test_residual_variance(self, result):
        assert result.mean_residual_variance == pytest.approx(0.865458938372, rel=1e-8)

    def test_strehl(self, result):
        assert result.mean_strehl == pytest.approx(0.468800314376, rel=1e-8)

    def test_rejection(self, result):
        assert result.rejection_db == pytest.approx(10.3766, abs=1e-3)


class TestPredictorPins:
    """Ridge ensemble, seeds 1 and 2 for training, seed 9 for test."""

    @pytest.fixture(scope="class")
    @staticmethod
    def fitted():
        data = make_slope_dataset(
            PINNED_CONFIG, n_frames=60, train_seeds=(1, 2), test_seeds=(9,)
        )
        model = LinearSlopePredictor(
            n_history=3, horizon=2, n_members=4, random_state=0
        ).fit(data.train)
        return model, data

    def test_alpha_selection(self, fitted):
        model, _ = fitted
        assert model.chosen_alpha == pytest.approx(100.0)

    def test_prediction_values(self, fitted):
        model, data = fitted
        mean, sigma = model.predict(data.test[0][:3])
        assert mean[0] == pytest.approx(15.971843317315, rel=1e-7)
        assert np.linalg.norm(mean) == pytest.approx(68.198562284822, rel=1e-7)
        assert np.linalg.norm(sigma) == pytest.approx(19.960694400972, rel=1e-7)
