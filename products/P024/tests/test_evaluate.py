"""Scoring, the gain oracle, and the training-set builder."""

from __future__ import annotations

import numpy as np
import pytest

from detumblesim.evaluate import (
    ENERGY_WEIGHT,
    FAILURE_PENALTY,
    fit_power_law_gain,
    oracle_gain,
    run_policy,
    score_run,
    training_rows,
)
from detumblesim.features import N_FEATURES
from detumblesim.policies import FixedGainPolicy
from detumblesim.scenarios import sample_scenario, sample_scenarios

FAST = {"duration_s": 8000.0, "control_dt_s": 4.0, "substeps": 1}


class TestScoring:
    def test_cost_decomposes_into_time_and_energy(self):
        s = sample_scenario(1000)
        res, sc = run_policy(s, FixedGainPolicy(1.5e5), **FAST)
        assert np.isclose(sc.cost, sc.time_orbits + sc.energy_term)
        m_max = float(np.min(s.magnetorquer.max_dipole_am2))
        assert np.isclose(
            sc.energy_term,
            ENERGY_WEIGHT * res.actuation_cost_a2m4s / (m_max**2 * s.orbit.period_s),
        )

    def test_failed_run_uses_the_penalised_span(self):
        s = sample_scenario(1000)
        # A tiny gain cannot detumble inside the span.
        res, sc = run_policy(s, FixedGainPolicy(1.0), **FAST)
        assert not sc.detumbled
        assert np.isnan(sc.detumble_time_s)
        assert np.isclose(sc.time_orbits, FAILURE_PENALTY * 8000.0 / s.orbit.period_s)

    def test_zero_energy_weight_drops_the_energy_term(self):
        s = sample_scenario(1001)
        res, _ = run_policy(s, FixedGainPolicy(1.5e5), **FAST)
        sc = score_run(res, s, 8000.0, energy_weight=0.0)
        assert sc.energy_term == 0.0
        assert np.isclose(sc.cost, sc.time_orbits)

    def test_rejects_negative_energy_weight(self):
        s = sample_scenario(1002)
        res, _ = run_policy(s, FixedGainPolicy(1.5e5), **FAST)
        with pytest.raises(ValueError, match="energy_weight"):
            score_run(res, s, 8000.0, energy_weight=-1.0)


class TestOracle:
    def test_picks_the_minimum_of_the_grid(self):
        s = sample_scenario(1000)
        gains = np.geomspace(1e4, 1e6, 5)
        best, cost, costs = oracle_gain(s, gains, **FAST)
        assert costs.shape == gains.shape
        assert np.isclose(cost, costs.min())
        assert best == gains[int(np.argmin(costs))]

    def test_grid_has_an_interior_optimum(self):
        # Too small a gain never detumbles, too large a gain wastes energy;
        # the cost curve must therefore not be monotone.
        s = sample_scenario(1003)
        _, _, costs = oracle_gain(s, np.geomspace(1e3, 1e7, 7), **FAST)
        assert int(np.argmin(costs)) not in (0, costs.size - 1)

    @pytest.mark.parametrize("bad", [np.array([]), np.array([[1e5]]), np.array([-1.0])])
    def test_rejects_bad_gain_grids(self, bad):
        with pytest.raises(ValueError):
            oracle_gain(sample_scenario(1000), bad, **FAST)


class TestTrainingRows:
    def test_shapes_and_constant_label(self):
        s = sample_scenario(1000)
        res, _ = run_policy(s, FixedGainPolicy(1e5), **FAST)
        x, y = training_rows(s, 3e5, 1e5, res, window_length=20, stride=10)
        assert x.shape[1] == N_FEATURES
        assert x.shape[0] == y.shape[0] > 0
        assert np.allclose(y, np.log10(3.0))
        assert np.all(np.isfinite(x))

    def test_hardware_features_match_the_scenario(self):
        s = sample_scenario(1004)
        res, _ = run_policy(s, FixedGainPolicy(1e5), **FAST)
        x, _ = training_rows(s, 1e5, 1e5, res, window_length=20, stride=10)
        assert np.allclose(x[:, 5], np.log10(float(np.min(s.magnetorquer.max_dipole_am2))))
        assert np.allclose(x[:, 6], np.log10(s.inertia_scale_kgm2))

    def test_returns_empty_when_the_run_is_shorter_than_the_window(self):
        s = sample_scenario(1000)
        res, _ = run_policy(s, FixedGainPolicy(1e5), **FAST)
        x, y = training_rows(s, 1e5, 1e5, res, window_length=10**6, stride=10)
        assert x.shape == (0, 8) and y.shape == (0,)

    @pytest.mark.parametrize(
        "kw,msg",
        [
            ({"best_gain": 0.0}, "best_gain"),
            ({"base_gain": -1.0}, "base_gain"),
            ({"stride": 0}, "stride"),
        ],
    )
    def test_rejects_bad_arguments(self, kw, msg):
        s = sample_scenario(1000)
        res, _ = run_policy(s, FixedGainPolicy(1e5), **FAST)
        args = {"best_gain": 1e5, "base_gain": 1e5, "window_length": 20, "stride": 10}
        args.update(kw)
        with pytest.raises(ValueError, match=msg):
            training_rows(s, result=res, **args)


class TestPowerLawFit:
    def test_recovers_an_exact_relationship(self):
        # Build gains that satisfy log10 k = 5 + 1.0 log10 m - 0.5 log10 j
        # exactly, and check the fit returns those coefficients.
        scens = sample_scenarios(8, 2000)
        m = np.array([float(np.min(s.magnetorquer.max_dipole_am2)) for s in scens])
        j = np.array([s.inertia_scale_kgm2 for s in scens])
        k = 10.0 ** (5.0 + 1.0 * np.log10(m) - 0.5 * np.log10(j))
        coef, rms = fit_power_law_gain(scens, k)
        assert np.allclose(coef, [5.0, 1.0, -0.5], atol=1e-8)
        assert rms < 1e-9

    def test_reports_a_residual_for_noisy_data(self):
        scens = sample_scenarios(10, 2100)
        rng = np.random.default_rng(0)
        k = 10.0 ** (5.0 + rng.normal(0.0, 0.3, 10))
        _, rms = fit_power_law_gain(scens, k)
        assert rms > 0.05

    @pytest.mark.parametrize(
        "n,gains,msg",
        [
            (5, np.ones(4), "one entry per scenario"),
            (5, -np.ones(5), "positive"),
            (2, np.ones(2), "at least three"),
        ],
    )
    def test_rejects_bad_inputs(self, n, gains, msg):
        with pytest.raises(ValueError, match=msg):
            fit_power_law_gain(sample_scenarios(n, 3000), gains)
