"""The seeded clock model: offset, skew, drift and the three noise processes.

The state model
---------------
The remote clock's time error relative to the local reference, in seconds, is

    theta(t) = theta0 + a * (t - t0) + 0.5 * D * (t - t0)**2 + n(t)         [s]

with

* ``theta0``  initial time offset                                          [s]
* ``a``       fractional frequency offset, the *skew*            [s/s, dimensionless]
* ``D``       frequency drift (ageing) rate                                [1/s]
* ``n(t)``    the stochastic part, below.

The deterministic part is the standard quadratic clock polynomial; see
Riley, *Handbook of Frequency Stability Analysis*, NIST Special Publication
1065 (2008), section 6.6, and IEEE Std 1139-2008 for the definitions of
fractional frequency offset and drift. Units are stated because a skew is
routinely quoted in parts per million and a drift in parts per day, and the
conversion is where errors enter: ``1 ppm = 1e-6 s/s`` and ``1 ppm/day =
1e-6 / 86400 s/s per second``.

The stochastic part
-------------------
``n(t)`` is the sum of the three noise processes whose Allan deviation slopes
are known in closed form, which is exactly why they are the ones used to
verify :mod:`timetransfer.allan`:

1. **White phase modulation (WPM).** Independent per read,
   ``N(0, sigma_wpm**2)``. This is timestamping noise, not clock state, so it
   is applied by the transport at each timestamp and is *not* part of
   ``theta(t)``. ADEV slope ``tau**-1``.
2. **White frequency modulation (WFM).** A Wiener process in phase with
   intensity ``q1`` [s^2/s]: ``dn1 = sqrt(q1 * dt) * N(0,1)``. ADEV slope
   ``tau**-1/2``.
3. **Random walk frequency modulation (RWFM).** A Wiener process in frequency
   with intensity ``q2`` [1/s], integrated into phase. ADEV slope
   ``tau**+1/2``.

Processes 2 and 3 together are the two-state clock model whose relationship to
the Allan variance is given by Zucca and Tavella, "The clock model and its
relationship with the Allan and related variances", *IEEE Transactions on
Ultrasonics, Ferroelectrics and Frequency Control* 52(2):289-296 (2005):
``sigma_y^2(tau) = q1/tau + q2*tau/3``. That identity is what
:mod:`timetransfer.skew` uses to build the Kalman process covariance, and it
is verified numerically in ``validation/validate_allan.py``.

Validity range
--------------
The model is a local expansion. It is valid over an interval short enough that
the quadratic term is a good description of the deterministic frequency
behaviour -- hours to days for an oven-controlled oscillator, far less for an
uncompensated crystal near a thermal transient. It contains no temperature
coefficient, no retrace, no frequency step and no leap second. A real
oscillator has all of these and none of them is simulated here.

Determinism
-----------
:meth:`ClockModel.advance_to` must be called with non-decreasing true times.
The noise state advances by the elapsed interval on each call, so the sequence
of readings is a deterministic function of the seed and the call sequence --
which is what makes simulation/device parity byte-identical rather than merely
close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .errors import ConfigurationError, TimebaseRegressionError

__all__ = ["ClockModel", "ClockSpec"]


@dataclass(frozen=True)
class ClockSpec:
    """Declared parameters of a simulated remote clock.

    Attributes
    ----------
    offset0_s:
        Initial time offset ``theta0`` [s].
    skew_s_per_s:
        Fractional frequency offset ``a`` [s/s]. ``1e-6`` is 1 ppm.
    drift_per_s:
        Frequency drift ``D`` [1/s], i.e. the rate of change of the fractional
        frequency offset per second.
    q1_s2_per_s:
        White-FM intensity [s^2/s]; the phase random-walk variance accrued per
        second.
    q2_per_s:
        Random-walk-FM intensity [1/s]; the frequency random-walk variance
        accrued per second.
    epoch_s:
        The ``t0`` of the polynomial [s], in the local reference timescale.
    """

    offset0_s: float = 0.0
    skew_s_per_s: float = 0.0
    drift_per_s: float = 0.0
    q1_s2_per_s: float = 0.0
    q2_per_s: float = 0.0
    epoch_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("q1_s2_per_s", "q2_per_s"):
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ConfigurationError(f"{name} must be finite and >= 0, got {value!r}")
        for name in ("offset0_s", "skew_s_per_s", "drift_per_s", "epoch_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ConfigurationError(f"{name} must be finite, got {value!r}")
        if abs(float(self.skew_s_per_s)) >= 1.0:
            raise ConfigurationError(
                f"skew_s_per_s must satisfy |a| < 1 (a clock that runs backwards is not "
                f"modelled), got {self.skew_s_per_s!r}"
            )

    def deterministic_offset_s(self, t_s: float | np.ndarray) -> float | np.ndarray:
        """Deterministic part of ``theta(t)`` [s]: polynomial only, no noise.

        Parameters
        ----------
        t_s:
            Local reference time [s], scalar or array.
        """
        dt = np.asarray(t_s, dtype=np.float64) - float(self.epoch_s)
        value = (
            float(self.offset0_s)
            + float(self.skew_s_per_s) * dt
            + 0.5 * float(self.drift_per_s) * dt * dt
        )
        return float(value) if np.isscalar(t_s) or np.ndim(t_s) == 0 else value

    def deterministic_skew_s_per_s(self, t_s: float | np.ndarray) -> float | np.ndarray:
        """Deterministic fractional frequency offset at ``t_s`` [s/s]."""
        dt = np.asarray(t_s, dtype=np.float64) - float(self.epoch_s)
        value = float(self.skew_s_per_s) + float(self.drift_per_s) * dt
        return float(value) if np.isscalar(t_s) or np.ndim(t_s) == 0 else value

    def allan_variance(self, tau_s: float | np.ndarray) -> float | np.ndarray:
        """Theoretical Allan variance of the stochastic part at ``tau_s``.

        ``sigma_y^2(tau) = q1/tau + q2*tau/3`` (Zucca and Tavella 2005, eq. 14).
        The deterministic drift contributes ``(D*tau)**2 / 2`` to the Allan
        variance and is deliberately **excluded** here, because the estimators
        in this package remove the polynomial before computing stability.
        """
        tau = np.asarray(tau_s, dtype=np.float64)
        if np.any(tau <= 0.0):
            raise ConfigurationError("tau_s must be > 0")
        value = float(self.q1_s2_per_s) / tau + float(self.q2_per_s) * tau / 3.0
        return float(value) if np.ndim(tau_s) == 0 else value


@dataclass
class ClockModel:
    """A seeded realisation of a :class:`ClockSpec`.

    The noise state is advanced by :meth:`advance_to`, which must be called
    with non-decreasing true times. :meth:`offset_at` returns the current
    ``theta`` including the realised noise, and is the *truth* against which
    every estimator in this package is scored.

    Parameters
    ----------
    spec:
        Declared clock parameters.
    seed:
        Seed for the ``numpy`` PCG64 generator driving the two noise processes.
    """

    spec: ClockSpec
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _t_s: float = field(init=False, default=0.0)
    _phase_wfm_s: float = field(init=False, default=0.0)
    _freq_rwfm: float = field(init=False, default=0.0)
    _phase_rwfm_s: float = field(init=False, default=0.0)
    advances: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(int(self.seed)))
        self._t_s = float(self.spec.epoch_s)

    @property
    def time_s(self) -> float:
        """True time [s] the noise state has been advanced to."""
        return self._t_s

    def advance_to(self, t_s: float) -> None:
        """Advance the noise state to true time ``t_s`` [s].

        A zero-length advance draws nothing, so calling this twice with the
        same time is a no-op and does not consume randomness.
        """
        t = float(t_s)
        if t < self._t_s:
            raise TimebaseRegressionError(
                f"clock model cannot advance backwards: {t!r} s after {self._t_s!r} s"
            )
        dt = t - self._t_s
        if dt == 0.0:
            return
        q1 = float(self.spec.q1_s2_per_s)
        q2 = float(self.spec.q2_per_s)
        if q1 > 0.0:
            self._phase_wfm_s += float(self._rng.normal(0.0, np.sqrt(q1 * dt)))
        if q2 > 0.0:
            # Trapezoidal integration of the frequency random walk into phase:
            # exact in mean, and the only approximation is the within-step
            # linearity of the Wiener path, which is stated rather than hidden.
            dw = float(self._rng.normal(0.0, np.sqrt(q2 * dt)))
            self._phase_rwfm_s += self._freq_rwfm * dt + 0.5 * dw * dt
            self._freq_rwfm += dw
        self._t_s = t
        self.advances += 1

    def noise_s(self) -> float:
        """Realised stochastic phase [s] at the current state time."""
        return self._phase_wfm_s + self._phase_rwfm_s

    def offset_at(self, t_s: float) -> float:
        """Advance to ``t_s`` and return the true clock offset ``theta(t_s)`` [s].

        This is the estimand. It excludes white phase modulation, which is
        timestamping noise applied by the transport and not a property of the
        clock's state.
        """
        self.advance_to(t_s)
        return float(self.spec.deterministic_offset_s(t_s)) + self.noise_s()

    def snapshot(self) -> dict[str, object]:
        """Complete state, including the generator's bit state, for recovery."""
        return {
            "t_s": self._t_s,
            "phase_wfm_s": self._phase_wfm_s,
            "freq_rwfm": self._freq_rwfm,
            "phase_rwfm_s": self._phase_rwfm_s,
            "advances": self.advances,
            "bit_generator_state": self._rng.bit_generator.state,
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        """Put the model back into a state captured by :meth:`snapshot`."""
        self._t_s = float(snapshot["t_s"])  # type: ignore[arg-type]
        self._phase_wfm_s = float(snapshot["phase_wfm_s"])  # type: ignore[arg-type]
        self._freq_rwfm = float(snapshot["freq_rwfm"])  # type: ignore[arg-type]
        self._phase_rwfm_s = float(snapshot["phase_rwfm_s"])  # type: ignore[arg-type]
        self.advances = int(snapshot["advances"])  # type: ignore[arg-type]
        self._rng.bit_generator.state = snapshot["bit_generator_state"]
