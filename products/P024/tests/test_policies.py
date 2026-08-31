"""Gain-selection policies."""

from __future__ import annotations

import numpy as np
import pytest

from detumblesim.features import N_FEATURES
from detumblesim.policies import (
    FixedGainPolicy,
    PowerLawGainPolicy,
    ScheduledGainPolicy,
    SizedGainPolicy,
    wrap_with_saturation_feedback,
)
from detumblesim.scheduler import GainScheduler
from detumblesim.spacecraft import Magnetorquer


class TestFixedGain:
    def test_known_answer(self):
        p = FixedGainPolicy(2.0e5)
        assert np.allclose(p.command([3e-5, 0, 0], [1e-6, 0.0, -2e-6]), [-0.2, 0.0, 0.4])
        assert p.current_gain() == 2.0e5

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("inf")])
    def test_rejects_bad_gain(self, bad):
        with pytest.raises(ValueError, match="gain"):
            FixedGainPolicy(bad)

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            FixedGainPolicy(1.0).command([1e-5, 0, 0], [1.0, 2.0])

    def test_reset_is_a_no_op(self):
        p = FixedGainPolicy(1.0)
        p.reset()
        assert p.current_gain() == 1.0


class TestPowerLaw:
    def test_known_answer(self):
        # log10 k = 6 + 1*log10(0.2) + 0*log10(0.05) = 6 - 0.69897 = 5.30103
        # so k = 2.0e5 exactly.
        p = PowerLawGainPolicy([6.0, 1.0, 0.0], 0.2, 0.05)
        assert np.isclose(p.gain, 2.0e5, rtol=1e-12)

    def test_inertia_exponent_is_used(self):
        a = PowerLawGainPolicy([5.0, 0.0, 1.0], 0.2, 0.05).gain
        b = PowerLawGainPolicy([5.0, 0.0, 1.0], 0.2, 0.10).gain
        assert np.isclose(b / a, 2.0, rtol=1e-12)

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="shape"):
            PowerLawGainPolicy([1.0, 2.0], 0.2, 0.05)
        with pytest.raises(ValueError, match="max_dipole_am2"):
            PowerLawGainPolicy([1.0, 2.0, 3.0], 0.0, 0.05)
        with pytest.raises(ValueError, match="inertia_scale_kgm2"):
            PowerLawGainPolicy([1.0, 2.0, 3.0], 0.2, -1.0)


class TestSizedGain:
    def _drive(self, p, n, rate=0.1, bmag=3e-5):
        for _ in range(n):
            p.command(np.array([bmag, 0.0, 0.0]), np.array([0.0, rate * bmag, 0.0]))
        return p

    def test_known_answer_sizing(self):
        # k = c m_max / (<|B|> omega_est); with c = 1, m_max = 0.2,
        # |B| = 3e-5 T and omega_est = 0.1 rad/s: k = 0.2/(3e-6) = 66666.667
        p = SizedGainPolicy(Magnetorquer.isotropic(0.2), coefficient=1.0, window=5)
        self._drive(p, 5)
        assert p.sized
        assert np.isclose(p.current_gain(), 0.2 / (3e-5 * 0.1), rtol=1e-9)

    def test_uses_the_fallback_before_sizing(self):
        p = SizedGainPolicy(Magnetorquer.isotropic(0.2), window=10, fallback_gain=1234.0)
        self._drive(p, 3)
        assert not p.sized
        assert p.current_gain() == 1234.0

    def test_gain_freezes_after_the_window(self):
        p = SizedGainPolicy(Magnetorquer.isotropic(0.2), coefficient=1.0, window=5)
        self._drive(p, 5)
        g = p.current_gain()
        self._drive(p, 50, rate=0.001)
        assert p.current_gain() == g

    def test_max_gain_clamp(self):
        p = SizedGainPolicy(
            Magnetorquer.isotropic(0.5), coefficient=1.0, window=3, max_gain=100.0
        )
        self._drive(p, 3, rate=1e-6)
        assert p.current_gain() == 100.0

    def test_estimator_choice_changes_the_gain(self):
        gains = {}
        for est in ("max", "mean", "median"):
            p = SizedGainPolicy(
                Magnetorquer.isotropic(0.2), coefficient=1.0, window=4, rate_estimator=est
            )
            for r in (0.05, 0.2, 0.1, 0.1):
                p.command(np.array([3e-5, 0.0, 0.0]), np.array([0.0, r * 3e-5, 0.0]))
            gains[est] = p.current_gain()
        assert gains["max"] < gains["mean"]  # larger rate estimate -> smaller gain

    def test_reset_restores_the_fallback(self):
        p = SizedGainPolicy(Magnetorquer.isotropic(0.2), window=3, fallback_gain=99.0)
        self._drive(p, 3)
        p.reset()
        assert not p.sized and p.current_gain() == 99.0

    @pytest.mark.parametrize(
        "kw,msg",
        [
            ({"coefficient": 0.0}, "coefficient"),
            ({"window": 1}, "window"),
            ({"fallback_gain": -1.0}, "fallback_gain"),
            ({"max_gain": 0.0}, "max_gain"),
            ({"rate_estimator": "p90"}, "rate_estimator"),
        ],
    )
    def test_rejects_bad_parameters(self, kw, msg):
        with pytest.raises(ValueError, match=msg):
            SizedGainPolicy(Magnetorquer.isotropic(0.2), **kw)


class TestScheduledGain:
    def _scheduler(self, label=0.0):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(60, N_FEATURES))
        return GainScheduler(n_estimators=20).fit(x, np.full(60, label))

    def test_uses_the_base_gain_before_the_window_fills(self):
        p = ScheduledGainPolicy(self._scheduler(), 1e5, 0.2, 0.05, window=10, update_every=5)
        p.command(np.array([3e-5, 0, 0]), np.array([0.0, 3e-6, 0.0]))
        assert p.current_gain() == 1e5

    def test_updates_and_records_history(self):
        p = ScheduledGainPolicy(self._scheduler(0.3), 1e5, 0.2, 0.05, window=6, update_every=3)
        for _ in range(30):
            p.command(np.array([3e-5, 0, 0]), np.array([0.0, 3e-6, 0.0]))
        assert p.gain_history
        assert all(0.0 < c <= 1.0 for _, _, c in p.gain_history)
        assert p.current_gain() > 1e5  # positive label raises the gain

    def test_reset_clears_history(self):
        p = ScheduledGainPolicy(self._scheduler(0.3), 1e5, 0.2, 0.05, window=6, update_every=3)
        for _ in range(30):
            p.command(np.array([3e-5, 0, 0]), np.array([0.0, 3e-6, 0.0]))
        p.reset()
        assert p.gain_history == [] and p.current_gain() == 1e5

    @pytest.mark.parametrize(
        "kw,msg", [({"base_gain": 0.0}, "base_gain"), ({"update_every": 0}, "update_every")]
    )
    def test_rejects_bad_parameters(self, kw, msg):
        args = {
            "scheduler": self._scheduler(), "base_gain": 1e5, "max_dipole_am2": 0.2,
            "inertia_scale_kgm2": 0.05,
        }
        args.update(kw)
        with pytest.raises(ValueError, match=msg):
            ScheduledGainPolicy(**args)


class TestSaturationWrapper:
    def test_passes_the_command_through_unchanged(self):
        inner = FixedGainPolicy(1e5)
        w = wrap_with_saturation_feedback(inner, Magnetorquer.isotropic(0.2))
        b, bd = np.array([3e-5, 0, 0]), np.array([0.0, 1e-6, 0.0])
        assert np.allclose(w.command(b, bd), inner.command(b, bd))
        assert w.current_gain() == 1e5
        w.reset()

    def test_reports_saturation_to_a_scheduled_policy(self):
        rng = np.random.default_rng(0)
        sch = GainScheduler(n_estimators=10).fit(
            rng.normal(size=(40, N_FEATURES)), np.zeros(40)
        )
        p = ScheduledGainPolicy(sch, 1e7, 0.2, 0.05, window=4, update_every=2)
        w = wrap_with_saturation_feedback(p, Magnetorquer.isotropic(0.2))
        for _ in range(12):
            w.command(np.array([3e-5, 0, 0]), np.array([0.0, 1e-5, 0.0]))
        # gain 1e7 x 1e-5 T/s = 100 A m^2, far above the 0.2 A m^2 limit.
        assert p._last_saturated is True
