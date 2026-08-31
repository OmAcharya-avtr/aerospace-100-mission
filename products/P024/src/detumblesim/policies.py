"""Gain-selection policies for B-dot detumbling.

Three classical gain rules and one learned one, all exposing the same
``command(b, b_dot, omega)`` interface the simulator expects.  Only
``b`` and ``b_dot`` are ever read: ``omega`` is accepted so the signature
matches the ``simulate.Controller`` protocol, and is deliberately ignored,
because a magnetometer-only detumble controller does not have a rate estimate.

Classical policies, implemented and validated first
---------------------------------------------------
``FixedGainPolicy``
    One constant gain for every spacecraft and every rate.  This is the naive
    baseline: the gain that a designer picks once and reuses.

``PowerLawGainPolicy``
    A three-coefficient log-linear fit of the oracle gain against the dipole
    limit and the inertia, fitted on the same training scenarios the learned
    scheduler sees.  This is the strong classical competitor.

``SizedGainPolicy``
    The gain a competent ADCS engineer would actually use.  It sizes the gain
    so the commanded dipole reaches the coil limit at the initial rate,

        k = c * m_max / ( <|B|> * omega_est )                 [A m^2 s / T]

    where ``omega_est`` comes from the observable rate proxy
    ``|dB/dt| / |B|`` over the first ``window`` control steps and ``<|B|>`` is
    the mean field magnitude over the same window.  Because
    ``|dB/dt|/|B| = |omega_perp| <= |omega|``, the proxy is a *lower bound* on
    the rate that is tight only when the rate is perpendicular to the field;
    the default estimator therefore takes the **maximum** over the sizing
    window, during which the body sweeps the field through a range of
    directions.  ``c = 1`` puts the command exactly at the limit; the
    coefficient is tuned once on training scenarios.  The gain is frozen after
    the sizing window, so this is still a *fixed* gain - fixed per spacecraft
    rather than fixed across the fleet.

Learned policy
--------------
``ScheduledGainPolicy`` re-evaluates the gain every ``update_every`` control
steps from a ``GainScheduler`` (``scheduler.py``), which maps the observable
feature vector to a multiplicative correction on a base gain and reports a
confidence.  Low confidence shrinks the correction back toward the base gain.

References
----------
Stickler, A. C. and Alfriend, K. T., "Elementary Magnetic Attitude Control
    System", J. Spacecraft and Rockets, 13(5), 1976, pp. 282-287.
    doi:10.2514/3.57089
Avanzini, G. and Giulietti, F., "Magnetic Detumbling of a Rigid Spacecraft",
    J. Guidance, Control, and Dynamics, 35(4), 2012, pp. 1326-1334.
    doi:10.2514/1.53074
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .features import TelemetryWindow, rate_proxy
from .spacecraft import Magnetorquer


class FixedGainPolicy:
    """Constant B-dot gain ``m = -k dB/dt``.

    Parameters
    ----------
    gain : float
        ``k`` [A m^2 s T^-1], positive.
    """

    def __init__(self, gain: float) -> None:
        g = float(gain)
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError(f"gain must be positive and finite, got {gain}")
        self.gain = g

    def reset(self) -> None:
        """No state to clear; present so every policy has the same interface."""

    def current_gain(self) -> float:
        """Gain in use at this step [A m^2 s T^-1]."""
        return self.gain

    def command(
        self,
        b_body_t: ArrayLike,
        b_dot_body_t_s: ArrayLike | None = None,
        omega_body: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Commanded (unsaturated) dipole [A m^2]."""
        bd = np.asarray(b_dot_body_t_s, dtype=float)
        if bd.shape != (3,):
            raise ValueError(f"b_dot must have shape (3,), got {bd.shape}")
        return -self.gain * bd


class SizedGainPolicy:
    """Gain sized from the first ``window`` magnetometer samples, then frozen.

    Parameters
    ----------
    magnetorquer : Magnetorquer
        Provides ``m_max``; the smallest per-axis limit is used, so the sizing
        is conservative for an anisotropic torquer set.
    coefficient : float
        ``c`` in ``k = c m_max / (<|B|> omega_est)``.  ``c = 1`` puts the
        command at the saturation limit at the estimated initial rate.
    window : int
        Number of control steps used to estimate the initial rate.
    rate_estimator : {"max", "mean", "median"}
        How the window of rate-proxy samples is reduced to one rate estimate.
        ``"max"`` is the default for the reason given above.
    fallback_gain : float
        Gain used before the sizing window has filled [A m^2 s T^-1].
    max_gain : float
        Hard upper clamp on the sized gain [A m^2 s T^-1]; a safety limit, not
        a tuning knob.
    """

    _ESTIMATORS = {"max": np.max, "mean": np.mean, "median": np.median}

    def __init__(
        self,
        magnetorquer: Magnetorquer,
        coefficient: float = 1.0,
        window: int = 40,
        rate_estimator: str = "max",
        fallback_gain: float = 1.0e4,
        max_gain: float = 1.0e9,
    ) -> None:
        c = float(coefficient)
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError(f"coefficient must be positive, got {coefficient}")
        if int(window) < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if not np.isfinite(fallback_gain) or fallback_gain <= 0.0:
            raise ValueError("fallback_gain must be positive")
        if not np.isfinite(max_gain) or max_gain <= 0.0:
            raise ValueError("max_gain must be positive")
        if rate_estimator not in self._ESTIMATORS:
            raise ValueError(
                f"rate_estimator must be one of {sorted(self._ESTIMATORS)}, "
                f"got {rate_estimator!r}"
            )
        self.rate_estimator = rate_estimator
        self.magnetorquer = magnetorquer
        self.coefficient = c
        self.window = int(window)
        self.fallback_gain = float(fallback_gain)
        self.max_gain = float(max_gain)
        self.reset()

    def reset(self) -> None:
        """Clear the sizing state so the policy can be reused."""
        self._rates: list[float] = []
        self._bmags: list[float] = []
        self._gain = self.fallback_gain
        self._sized = False

    @property
    def sized(self) -> bool:
        """True once the sizing window has closed."""
        return self._sized

    def current_gain(self) -> float:
        """Gain in use at this step [A m^2 s T^-1]."""
        return self._gain

    def command(
        self,
        b_body_t: ArrayLike,
        b_dot_body_t_s: ArrayLike | None = None,
        omega_body: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Commanded (unsaturated) dipole [A m^2]."""
        b = np.asarray(b_body_t, dtype=float)
        bd = np.asarray(b_dot_body_t_s, dtype=float)
        if not self._sized:
            self._rates.append(rate_proxy(b, bd))
            self._bmags.append(float(np.linalg.norm(b)))
            if len(self._rates) >= self.window:
                w_est = float(self._ESTIMATORS[self.rate_estimator](self._rates))
                b_mean = float(np.mean(self._bmags))
                if w_est > 0.0 and b_mean > 0.0:
                    m_max = float(np.min(self.magnetorquer.max_dipole_am2))
                    self._gain = min(
                        self.coefficient * m_max / (b_mean * w_est), self.max_gain
                    )
                self._sized = True
        return -self._gain * bd


class PowerLawGainPolicy(FixedGainPolicy):
    """Gain from a fitted log-linear scaling law, constant for the whole run.

    ``log10 k = a + b log10 m_max + c log10 j``

    The three coefficients are fitted by ordinary least squares to the oracle
    gains of the *training* scenarios (``evaluate.fit_power_law_gain``).  This
    is the strong classical competitor to the learned scheduler: it uses the
    same training data and the same two vehicle parameters, but it has three
    free parameters and no machine learning in it at all.

    Physically ``b = 1`` and ``c = 0`` would be the sizing-rule exponents
    (``k ~ m_max / (B omega)``); the fitted values are reported in
    ``validation/VALIDATION.md`` and are not assumed.

    Parameters
    ----------
    coefficients : array_like, shape (3,)
        ``(a, b, c)``.
    max_dipole_am2, inertia_scale_kgm2 : float
        Vehicle parameters [A m^2], [kg m^2]; both positive.
    """

    def __init__(
        self,
        coefficients: ArrayLike,
        max_dipole_am2: float,
        inertia_scale_kgm2: float,
    ) -> None:
        c = np.asarray(coefficients, dtype=float)
        if c.shape != (3,):
            raise ValueError(f"coefficients must have shape (3,), got {c.shape}")
        if not np.isfinite(max_dipole_am2) or max_dipole_am2 <= 0.0:
            raise ValueError("max_dipole_am2 must be positive")
        if not np.isfinite(inertia_scale_kgm2) or inertia_scale_kgm2 <= 0.0:
            raise ValueError("inertia_scale_kgm2 must be positive")
        log_k = (
            c[0]
            + c[1] * np.log10(float(max_dipole_am2))
            + c[2] * np.log10(float(inertia_scale_kgm2))
        )
        super().__init__(float(10.0**log_k))
        self.coefficients = c


class ScheduledGainPolicy:
    """B-dot with a learned, periodically updated gain.

    Parameters
    ----------
    scheduler : GainScheduler
        Fitted scheduler from ``scheduler.py``.
    base_gain : float
        Gain the scheduler's log-correction is applied to, and the value it
        falls back to before the window fills or when confidence is low.
    max_dipole_am2, inertia_scale_kgm2 : float
        Known vehicle parameters supplied as features.
    window : int
        Trailing window length in control steps.
    update_every : int
        Control steps between gain updates; the gain is held in between, as it
        would be on a duty-cycled ADCS task.
    """

    def __init__(
        self,
        scheduler,
        base_gain: float,
        max_dipole_am2: float,
        inertia_scale_kgm2: float,
        window: int = 60,
        update_every: int = 30,
    ) -> None:
        if not np.isfinite(base_gain) or base_gain <= 0.0:
            raise ValueError(f"base_gain must be positive, got {base_gain}")
        if int(update_every) < 1:
            raise ValueError(f"update_every must be >= 1, got {update_every}")
        self.scheduler = scheduler
        self.base_gain = float(base_gain)
        self.max_dipole_am2 = float(max_dipole_am2)
        self.inertia_scale_kgm2 = float(inertia_scale_kgm2)
        self.window_length = int(window)
        self.update_every = int(update_every)
        self.reset()

    def reset(self) -> None:
        """Clear the telemetry window and gain history."""
        self._win = TelemetryWindow(self.window_length)
        self._gain = self.base_gain
        self._step = 0
        self._last_saturated = False
        self.gain_history: list[tuple[float, float, float]] = []

    def note_saturation(self, saturated: bool) -> None:
        """Report whether the previous command was clipped (feature 3)."""
        self._last_saturated = bool(saturated)

    def current_gain(self) -> float:
        """Gain in use at this step [A m^2 s T^-1]."""
        return self._gain

    def command(
        self,
        b_body_t: ArrayLike,
        b_dot_body_t_s: ArrayLike | None = None,
        omega_body: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Commanded (unsaturated) dipole [A m^2]."""
        b = np.asarray(b_body_t, dtype=float)
        bd = np.asarray(b_dot_body_t_s, dtype=float)
        t = float(self._step)
        self._win.push(t, b, bd, self._last_saturated)
        if (
            self._win.ready()
            and len(self._win) >= self.window_length
            and self._step % self.update_every == 0
        ):
            x = self._win.features(self.max_dipole_am2, self.inertia_scale_kgm2)
            self._gain, conf = self.scheduler.predict_gain(x, self.base_gain)
            self.gain_history.append((t, self._gain, conf))
        self._step += 1
        return -self._gain * bd


def wrap_with_saturation_feedback(policy, magnetorquer: Magnetorquer):
    """Return ``policy`` with saturation reported back to it after each step.

    ``ScheduledGainPolicy`` uses the saturation duty cycle as a feature, but
    the simulator applies saturation *after* calling ``command``.  This thin
    wrapper closes that loop without changing the simulator.
    """

    class _Wrapped:
        def __init__(self, inner, mtq: Magnetorquer) -> None:
            self.inner = inner
            self.mtq = mtq

        def reset(self) -> None:
            self.inner.reset()

        def current_gain(self) -> float:
            return self.inner.current_gain()

        def command(self, b_body_t, b_dot_body_t_s=None, omega_body=None):
            m = self.inner.command(b_body_t, b_dot_body_t_s, omega_body)
            _, sat = self.mtq.saturate(m)
            if hasattr(self.inner, "note_saturation"):
                self.inner.note_saturation(sat)
            return m

    return _Wrapped(policy, magnetorquer)
