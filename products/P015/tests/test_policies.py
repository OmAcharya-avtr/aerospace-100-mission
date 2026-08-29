"""Tests for linkswitch.policies: fixed-threshold, hysteresis, learned."""

import numpy as np
import pytest

from linkswitch.learn import OutagePredictor
from linkswitch.policies import FixedThresholdPolicy, HysteresisPolicy, LearnedPolicy
from linkswitch.scenario import ScenarioConfig, generate_telemetry


def _fake_telemetry(irradiance):
    from linkswitch.scenario import Telemetry

    n = len(irradiance)
    irr = np.array(irradiance, dtype=float)
    return Telemetry(
        irradiance=irr,
        opt_available=irr >= 1.0,
        rain_rate_mm_hr=np.zeros(n),
        rf_atten_db=np.zeros(n),
        rf_available=np.ones(n, dtype=bool),
    )


class TestFixedThresholdPolicy:
    def test_known_answer_selection(self):
        tel = _fake_telemetry([2.0, 0.5, 3.0, 0.1, 5.0])
        p = FixedThresholdPolicy(tau=1.0)
        np.testing.assert_array_equal(
            p.select_channels(tel), [True, False, True, False, True]
        )

    def test_tau_at_boundary_is_inclusive(self):
        tel = _fake_telemetry([1.0])
        p = FixedThresholdPolicy(tau=1.0)
        assert p.select_channels(tel)[0]  # >= tau selects optical

    def test_invalid_tau_rejected(self):
        with pytest.raises(ValueError):
            FixedThresholdPolicy(tau=0.0)
        with pytest.raises(ValueError):
            FixedThresholdPolicy(tau=-1.0)
        with pytest.raises(ValueError):
            FixedThresholdPolicy(tau=float("nan"))

    def test_output_shape_matches_input(self):
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, 300, seed=0)
        p = FixedThresholdPolicy(tau=cfg.optical.tau_phys)
        assert p.select_channels(tel).shape == (300,)


class TestHysteresisPolicy:
    def test_known_answer_no_chatter(self):
        # tau_low=1.0, tau_high=2.0. Sequence dips just below tau_low then
        # bounces around inside the deadband without crossing tau_high --
        # must stay on RF the whole time (no premature return).
        irr = [3.0, 0.9, 1.5, 1.8, 1.5, 0.9, 2.5]
        tel = _fake_telemetry(irr)
        p = HysteresisPolicy(tau_low=1.0, tau_high=2.0)
        sel = p.select_channels(tel)
        # starts optical, drops to RF at t=1 (0.9 < 1.0), stays RF through the
        # deadband (values 1.5, 1.8, 1.5 are all < 2.0), returns at t=6 (2.5>2.0)
        np.testing.assert_array_equal(sel, [True, False, False, False, False, False, True])

    def test_equal_thresholds_degenerates_to_fixed(self):
        tel = _fake_telemetry([2.0, 0.5, 3.0, 0.1, 5.0])
        p = HysteresisPolicy(tau_low=1.0, tau_high=1.0)
        fixed = FixedThresholdPolicy(tau=1.0)
        np.testing.assert_array_equal(p.select_channels(tel), fixed.select_channels(tel))

    def test_tau_high_below_tau_low_rejected(self):
        with pytest.raises(ValueError):
            HysteresisPolicy(tau_low=2.0, tau_high=1.0)

    def test_invalid_taus_rejected(self):
        with pytest.raises(ValueError):
            HysteresisPolicy(tau_low=-1.0, tau_high=1.0)
        with pytest.raises(ValueError):
            HysteresisPolicy(tau_low=1.0, tau_high=-1.0)

    def test_fewer_switches_than_fixed_on_noisy_data(self):
        # Fundamental property motivating hysteresis: on data that chatters
        # around a single threshold, hysteresis switches less often.
        rng = np.random.default_rng(0)
        irr = 1.0 + 0.05 * rng.standard_normal(500)  # hovers right at tau=1.0
        irr = np.abs(irr) + 0.01
        tel = _fake_telemetry(irr)
        fixed = FixedThresholdPolicy(tau=1.0)
        hyst = HysteresisPolicy(tau_low=0.95, tau_high=1.05)

        def count_switches(sel):
            return int(np.sum(sel[1:] != sel[:-1]))

        assert count_switches(hyst.select_channels(tel)) <= count_switches(
            fixed.select_channels(tel)
        )


class TestLearnedPolicy:
    def test_invalid_confidence_threshold_rejected(self):
        model = OutagePredictor()
        with pytest.raises(ValueError):
            LearnedPolicy(model, tau_phys=1.0, confidence_threshold=1.5, window=5)
        with pytest.raises(ValueError):
            LearnedPolicy(model, tau_phys=1.0, confidence_threshold=-0.1, window=5)

    def test_invalid_tau_phys_rejected(self):
        model = OutagePredictor()
        with pytest.raises(ValueError):
            LearnedPolicy(model, tau_phys=0.0, confidence_threshold=0.5, window=5)

    def test_invalid_window_rejected(self):
        model = OutagePredictor()
        with pytest.raises(ValueError):
            LearnedPolicy(model, tau_phys=1.0, confidence_threshold=0.5, window=0)

    def test_always_predict_outage_forces_rf(self):
        model = OutagePredictor().fit(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([1, 1]),
        )
        tel = _fake_telemetry([5.0, 5.0, 5.0, 5.0])
        p = LearnedPolicy(model, tau_phys=1.0, confidence_threshold=0.5, window=2)
        sel = p.select_channels(tel)
        # constant-1 predictor: outage_confidence == 1.0 always >= threshold,
        # so RF is selected from the very first step onward.
        assert not sel.any()

    def test_persistent_pessimistic_predictor_does_not_flap(self):
        # Regression test: a model that keeps predicting outage while the
        # observed irradiance stays comfortably safe must not flap every
        # step -- it should switch away once and stay on RF (return also
        # requires the model's own confidence to clear, not just irradiance).
        model = OutagePredictor().fit(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([1, 1]),
        )
        tel = _fake_telemetry([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        p = LearnedPolicy(model, tau_phys=1.0, confidence_threshold=0.5, window=2)
        sel = p.select_channels(tel)
        assert not sel.any()  # never returns to optical: 0 -> 1 switches, not repeated flapping
        n_switches = int(np.sum(sel[1:] != sel[:-1]))
        assert n_switches == 0

    def test_always_predict_safe_stays_optical(self):
        model = OutagePredictor().fit(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([0, 0]),
        )
        tel = _fake_telemetry([5.0, 5.0, 5.0, 5.0])
        p = LearnedPolicy(model, tau_phys=1.0, confidence_threshold=0.5, window=2)
        sel = p.select_channels(tel)
        assert sel.all()

    def test_output_shape(self):
        cfg = ScenarioConfig()
        tel = generate_telemetry(cfg, 200, seed=0)
        model = OutagePredictor().fit(
            np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]]),
            np.array([0, 1]),
        )
        p = LearnedPolicy(model, tau_phys=cfg.optical.tau_phys, confidence_threshold=0.5, window=5)
        assert p.select_channels(tel).shape == (200,)
