"""Tests for waveforge.control, including the analytic transfer functions."""

from __future__ import annotations

import numpy as np
import pytest

from waveforge.control import (
    Integrator,
    noise_transfer,
    noise_variance_gain,
    rejection_transfer,
    stability_limit_gain,
)


class TestStabilityLimits:
    def test_one_frame_delay(self):
        # z - 1 + g = 0 -> z = 1 - g, stable for 0 < g < 2
        assert stability_limit_gain(1) == pytest.approx(2.0, abs=1e-6)

    def test_two_frame_delay(self):
        # z^2 - z + g = 0 -> |z|^2 = g for complex roots, stable for g < 1
        assert stability_limit_gain(2) == pytest.approx(1.0, abs=1e-6)

    def test_three_frame_delay(self):
        # z^3 - z^2 + g = 0 first touches |z| = 1 at g = 2 sin(pi/10)
        assert stability_limit_gain(3) == pytest.approx(2 * np.sin(np.pi / 10), abs=1e-6)

    def test_limit_decreases_with_delay(self):
        limits = [stability_limit_gain(d) for d in range(1, 7)]
        assert all(a > b for a, b in zip(limits, limits[1:], strict=False))

    @pytest.mark.parametrize("leak", [0.8, 0.9, 0.95, 1.0])
    def test_leak_at_one_frame_delay(self, leak):
        # z - leak + g = 0 -> z = leak - g, so |z| < 1 gives g < 1 + leak
        assert stability_limit_gain(1, leak=leak) == pytest.approx(1.0 + leak, abs=1e-6)

    def test_leak_does_not_change_the_two_frame_limit(self):
        # z^2 - leak z + g has |z|^2 = g for its complex roots, so the limit
        # stays at g = 1 whatever the leak. Documented, not assumed.
        assert stability_limit_gain(2, leak=0.9) == pytest.approx(
            stability_limit_gain(2, leak=1.0), abs=1e-6
        )

    @pytest.mark.parametrize("delay", [0, -1, 2.5])
    def test_bad_delay(self, delay):
        with pytest.raises(ValueError, match="delay_frames"):
            stability_limit_gain(delay)

    @pytest.mark.parametrize("leak", [0.0, 1.1, -0.5])
    def test_bad_leak(self, leak):
        with pytest.raises(ValueError, match="leak"):
            stability_limit_gain(2, leak=leak)

    def test_bad_tolerance(self):
        with pytest.raises(ValueError, match="tol"):
            stability_limit_gain(2, tol=0.0)

    def test_poles_inside_unit_circle_below_limit(self):
        for d in (1, 2, 3, 4):
            g = 0.9 * stability_limit_gain(d)
            coeffs = np.zeros(d + 1)
            coeffs[0], coeffs[1] = 1.0, -1.0
            coeffs[-1] += g
            assert np.max(np.abs(np.roots(coeffs))) < 1.0


class TestRejectionTransfer:
    def test_dc_rejection_is_total(self):
        assert abs(rejection_transfer(0.0, 1000.0, 0.4, 2)) == pytest.approx(0.0, abs=1e-12)

    def test_leak_leaves_dc_error(self):
        assert abs(rejection_transfer(0.0, 1000.0, 0.4, 2, leak=0.99)) > 0.0

    def test_magnitude_tends_to_one_at_nyquist(self):
        value = abs(rejection_transfer(500.0, 1000.0, 0.4, 2))
        assert value == pytest.approx(2.0 / (2.0 + 0.4), rel=1e-9)

    def test_higher_gain_rejects_more_at_low_frequency(self):
        low = abs(rejection_transfer(10.0, 1000.0, 0.2, 2))
        high = abs(rejection_transfer(10.0, 1000.0, 0.6, 2))
        assert high < low

    def test_gain_peaking_exists(self):
        f = np.linspace(1.0, 500.0, 400)
        assert np.max(np.abs(rejection_transfer(f, 1000.0, 0.5, 2))) > 1.0

    def test_array_input(self):
        f = np.array([1.0, 10.0, 100.0])
        assert rejection_transfer(f, 1000.0, 0.4, 2).shape == (3,)

    @pytest.mark.parametrize("gain", [0.0, -0.1, float("nan")])
    def test_bad_gain(self, gain):
        with pytest.raises(ValueError, match="gain"):
            rejection_transfer(10.0, 1000.0, gain, 2)

    def test_bad_frame_rate(self):
        with pytest.raises(ValueError, match="frame_rate_hz"):
            rejection_transfer(10.0, 0.0, 0.4, 2)

    def test_bad_delay(self):
        with pytest.raises(ValueError, match="delay_frames"):
            rejection_transfer(10.0, 1000.0, 0.4, 0)


class TestNoiseTransfer:
    def test_dc_noise_passes_fully(self):
        assert abs(noise_transfer(0.0, 1000.0, 0.4, 2)) == pytest.approx(1.0, rel=1e-9)

    def test_sums_with_rejection_to_one(self):
        # E/Phi - E/N = 1 identically from the definitions
        f = np.array([3.0, 30.0, 300.0])
        e = rejection_transfer(f, 1000.0, 0.35, 3)
        n = noise_transfer(f, 1000.0, 0.35, 3)
        assert np.allclose(e - n, 1.0)

    def test_bad_gain(self):
        with pytest.raises(ValueError, match="gain"):
            noise_transfer(10.0, 1000.0, -1.0, 2)

    def test_bad_frame_rate(self):
        with pytest.raises(ValueError, match="frame_rate_hz"):
            noise_transfer(10.0, -1.0, 0.4, 2)


class TestNoiseVarianceGain:
    @pytest.mark.parametrize("gain", [0.1, 0.3, 0.5, 0.9, 1.5])
    def test_matches_classical_formula_at_one_frame(self, gain):
        # Madec 1999 Eq. 3.20: eta = g / (2 - g) for a one-frame-delay integrator
        assert noise_variance_gain(gain, 1) == pytest.approx(gain / (2 - gain), rel=1e-6)

    def test_increases_with_gain(self):
        values = [noise_variance_gain(g, 2) for g in (0.1, 0.3, 0.5, 0.7)]
        assert all(a < b for a, b in zip(values, values[1:], strict=False))

    def test_increases_with_delay(self):
        assert noise_variance_gain(0.3, 3) > noise_variance_gain(0.3, 1)

    def test_infinite_when_unstable(self):
        assert np.isinf(noise_variance_gain(1.5, 2))

    def test_matches_parseval_integral(self):
        # sum h_k^2 must equal the mean square of |E/N| over the unit circle
        gain, delay = 0.4, 2
        f = np.linspace(0.0, 1000.0, 20001)[:-1]
        spectrum = np.abs(noise_transfer(f, 1000.0, gain, delay)) ** 2
        assert noise_variance_gain(gain, delay) == pytest.approx(spectrum.mean(), rel=1e-3)

    def test_bad_n_terms(self):
        with pytest.raises(ValueError, match="n_terms"):
            noise_variance_gain(0.4, 2, n_terms=4)


class TestIntegrator:
    def test_initial_command_is_zero(self):
        assert np.allclose(Integrator(3).command, 0.0)

    def test_single_step_with_unit_delay(self):
        # d = 1: the increment is used immediately, c_1 = g * inc
        it = Integrator(2, gain=0.5, delay_frames=1)
        out = it.step(np.array([1.0, -2.0]))
        assert np.allclose(out, [0.5, -1.0])

    def test_delay_two_holds_one_frame(self):
        it = Integrator(1, gain=0.5, delay_frames=2)
        assert it.step(np.array([1.0]))[0] == pytest.approx(0.0)
        assert it.step(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_accumulates(self):
        it = Integrator(1, gain=0.25, delay_frames=1)
        values = [it.step(np.array([1.0]))[0] for _ in range(4)]
        assert values == pytest.approx([0.25, 0.5, 0.75, 1.0])

    def test_leak_decays_the_command(self):
        it = Integrator(1, gain=1.0, delay_frames=1, leak=0.5)
        it.step(np.array([1.0]))
        assert it.step(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_reset_clears_state(self):
        it = Integrator(1, gain=0.5, delay_frames=1)
        it.step(np.array([1.0]))
        it.reset()
        assert np.allclose(it.command, 0.0)
        assert it.step(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_command_limit_clips(self):
        it = Integrator(2, gain=1.0, delay_frames=1, command_limit=0.5)
        out = it.step(np.array([2.0, -0.1]))
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(-0.1)
        assert it.last_saturated_fraction == pytest.approx(0.5)

    def test_no_saturation_without_limit(self):
        it = Integrator(2, gain=1.0, delay_frames=1)
        it.step(np.array([100.0, -100.0]))
        assert it.last_saturated_fraction == 0.0

    def test_stability_properties(self):
        assert Integrator(1, gain=0.5, delay_frames=2).is_stable
        assert not Integrator(1, gain=1.5, delay_frames=2).is_stable
        assert Integrator(1, gain=0.5, delay_frames=2).stability_limit == pytest.approx(
            1.0, abs=1e-6
        )

    def test_wrong_increment_shape(self):
        with pytest.raises(ValueError, match="increment must have shape"):
            Integrator(3).step(np.zeros(2))

    @pytest.mark.parametrize("n", [0, -1, 2.5])
    def test_bad_n_commands(self, n):
        with pytest.raises(ValueError, match="n_commands"):
            Integrator(n)

    def test_bad_command_limit(self):
        with pytest.raises(ValueError, match="command_limit"):
            Integrator(2, command_limit=0.0)

    def test_bad_gain(self):
        with pytest.raises(ValueError, match="gain"):
            Integrator(2, gain=0.0)


class TestTimeDomainMatchesAnalytic:
    """A sinusoidal disturbance through the time-domain loop must reproduce
    the analytic rejection transfer function.  This is the strongest available
    check that the implementation and the equations agree."""

    @staticmethod
    def _measure(frequency_hz, frame_rate_hz, gain, delay, n_frames=3000):
        integrator = Integrator(1, gain=gain, delay_frames=delay)
        command = 0.0
        residuals = np.empty(n_frames)
        for k in range(n_frames):
            phi = np.sin(2 * np.pi * frequency_hz * k / frame_rate_hz)
            error = phi - command
            command = float(integrator.step(np.array([error]))[0])
            residuals[k] = error
        settled = residuals[n_frames // 2 :]
        phase = 2 * np.pi * frequency_hz * np.arange(n_frames // 2, n_frames) / frame_rate_hz
        return 2.0 * abs(np.mean(settled * np.exp(-1j * phase)))

    @pytest.mark.parametrize("delay", [1, 2, 3])
    @pytest.mark.parametrize(("gain", "frequency"), [(0.2, 37.0), (0.5, 113.0)])
    def test_matches(self, delay, gain, frequency):
        measured = self._measure(frequency, 1000.0, gain, delay)
        analytic = abs(rejection_transfer(frequency, 1000.0, gain, delay))
        assert measured == pytest.approx(analytic, rel=1e-4)

    def test_gain_peaking_is_reproduced(self):
        # near f_s/6 with d = 3 the loop amplifies rather than rejects
        measured = self._measure(160.0, 1000.0, 0.3, 3)
        analytic = abs(rejection_transfer(160.0, 1000.0, 0.3, 3))
        assert measured > 1.0
        assert measured == pytest.approx(analytic, rel=1e-3)
