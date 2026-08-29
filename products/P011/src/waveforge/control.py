"""Closed-loop integrator control: analytic transfer functions and stability.

Loop model
----------
The adaptive-optics loop implemented in :mod:`waveforge.loop` is the standard
discrete-time integrator with pure frame delay (Madec, P.-Y. 1999, "Control
techniques", in *Adaptive Optics in Astronomy*, ed. F. Roddier, Cambridge
University Press, ch. 3; Hardy 1998 ch. 7):

```
c[k] = c[k-1] + g * e[k - L]
```

where ``c`` is the DM command, ``e`` the reconstructed wavefront error, ``g``
the loop gain [-] and ``L >= 1`` the total loop **latency in frames** (sensor
integration + readout + reconstruction + DM settling). ``L = 1`` is the
minimum achievable: a measurement taken during frame ``k`` cannot influence the
mirror before frame ``k+1``.

In the z domain, with the DM correcting the incoming disturbance ``phi``,

```
open loop        G(z) = g z^-L / (1 - z^-1)
error rejection  E(z) = 1 / (1 + G(z)) = (1 - z^-1) / (1 - z^-1 + g z^-L)
```

``|E(f)|`` is the **error rejection transfer function**: the factor by which a
sinusoidal disturbance at temporal frequency ``f`` survives the loop. Units:
``f`` [Hz], sampled at the frame rate ``f_s`` [Hz], ``z = exp(2 pi i f / f_s)``.
*Validity:* linear, time-invariant, single-input single-output per mode;
assumes perfect modal decoupling and an ideal (instantaneous, unity-gain) DM.
The model does **not** include the sensor's finite integration time, which
adds a ``sinc(f/f_s)`` factor and a further half-frame of delay; that omission
is documented in the README Limitations and is the main reason a real loop
tolerates slightly less gain than :func:`stability_gain_limit` predicts.

Stability. The closed-loop poles are the roots of ``z^L - z^(L-1) + g = 0``.
For ``L = 1`` the single pole is ``z = 1 - g``, stable for ``0 < g < 2``. For
``L = 2`` the poles are ``(1 +- sqrt(1 - 4g))/2`` with modulus ``sqrt(g)`` once
``g > 1/4``, giving a stability limit of exactly ``g = 1``.
:func:`stability_gain_limit` solves the general case numerically and is checked
against these two closed forms in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["Integrator", "rejection_transfer_function", "stability_gain_limit", "closed_loop_poles"]


def closed_loop_poles(gain: float, latency: int) -> NDArray[np.complex128]:
    """Closed-loop poles of the integrator, roots of ``z^L - z^(L-1) + g``.

    Parameters
    ----------
    gain:
        Loop gain ``g`` [-].
    latency:
        Loop latency ``L`` in frames [-], >= 1.
    """
    g = float(gain)
    latency = int(latency)
    if latency < 1:
        raise ValueError(f"latency must be >= 1 frame, got {latency}")
    coeffs = np.zeros(latency + 1, dtype=np.float64)
    coeffs[0] += 1.0
    coeffs[1] += -1.0
    coeffs[latency] += g
    return np.roots(coeffs)


def stability_gain_limit(latency: int, tol: float = 1.0e-9) -> float:
    """Largest loop gain for which the integrator is stable, given ``latency`` frames.

    Found by bisection on ``max |pole| = 1``. Exact reference values:
    ``L = 1 -> 2``, ``L = 2 -> 1``.

    Returns
    -------
    float
        Marginal gain [-]. Gains at or above this diverge.
    """
    latency = int(latency)
    if latency < 1:
        raise ValueError(f"latency must be >= 1 frame, got {latency}")
    lo, hi = 1.0e-9, 2.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if max(abs(closed_loop_poles(mid, latency))) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rejection_transfer_function(
    frequency: NDArray[np.float64], gain: float, latency: int, frame_rate: float
) -> NDArray[np.float64]:
    """Magnitude of the error rejection transfer function ``|E(f)|`` [-].

    Parameters
    ----------
    frequency:
        Temporal frequency [Hz]. Must be >= 0 and < ``frame_rate / 2``.
    gain:
        Loop gain ``g`` [-], > 0.
    latency:
        Loop latency ``L`` in frames [-], >= 1.
    frame_rate:
        Sampling rate ``f_s`` [Hz], > 0.

    Notes
    -----
    ``E(z) = (1 - z^-1) / (1 - z^-1 + g z^-L)`` with ``z = exp(2 pi i f/f_s)``.
    ``|E| -> 0`` as ``f -> 0`` (an integrator rejects DC completely) and
    ``|E| -> 1`` at high frequency, with a resonant overshoot near
    ``f_s / (4 L)`` whose height grows with ``g``.
    """
    f = np.asarray(frequency, dtype=np.float64)
    if np.any(f < 0):
        raise ValueError("frequency must be >= 0")
    g = float(gain)
    if not np.isfinite(g) or g <= 0:
        raise ValueError(f"gain must be > 0, got {gain!r}")
    latency = int(latency)
    if latency < 1:
        raise ValueError(f"latency must be >= 1 frame, got {latency}")
    fs = float(frame_rate)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"frame_rate must be > 0, got {frame_rate!r}")
    if np.any(f >= fs / 2.0):
        raise ValueError(f"frequency must be below Nyquist {fs / 2.0} Hz")
    z = np.exp(2j * np.pi * f / fs)
    zi = 1.0 / z
    return np.abs((1.0 - zi) / (1.0 - zi + g * zi**latency))


@dataclass
class Integrator:
    """Leaky discrete-time integrator with configurable gain, latency and leak.

    Parameters
    ----------
    n_commands:
        Length of the command vector [-].
    gain:
        Loop gain ``g`` [-], > 0.
    latency:
        Total loop latency in frames [-], >= 1. ``latency = 1`` means the error
        pushed at frame ``k`` changes the command applied at frame ``k+1`` (the
        minimum physically achievable); each extra frame adds one place to an
        internal delay queue of length ``latency - 1``.
    leak:
        Optional forgetting factor ``0 <= leak < 1``; the update becomes
        ``c[k] = (1-leak) c[k-1] + g e[k-L]``. Default 0 (pure integrator).
    limit:
        Optional symmetric saturation on the command vector [same units as
        commands]. Applied after every update; the stored state is the
        *saturated* value, which provides integrator anti-windup.

    Notes
    -----
    ``leak > 0`` moves the DC pole inside the unit circle, which bounds the
    command when the reconstructor has null modes (waffle, global piston) at
    the cost of imperfect DC rejection: ``|E(0)| = leak / (leak + g)``.
    """

    n_commands: int
    gain: float
    latency: int = 1
    leak: float = 0.0
    limit: float | None = None
    _state: NDArray[np.float64] = field(init=False, repr=False)
    _queue: list[NDArray[np.float64]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = int(self.n_commands)
        if n < 1:
            raise ValueError(f"n_commands must be >= 1, got {n}")
        self.n_commands = n
        g = float(self.gain)
        if not np.isfinite(g) or g <= 0.0:
            raise ValueError(f"gain must be > 0, got {self.gain!r}")
        self.gain = g
        lat = int(self.latency)
        if lat < 1:
            raise ValueError(f"latency must be >= 1 frame, got {self.latency!r}")
        self.latency = lat
        lk = float(self.leak)
        if not (0.0 <= lk < 1.0):
            raise ValueError(f"leak must be in [0, 1), got {self.leak!r}")
        self.leak = lk
        if self.limit is not None:
            lim = float(self.limit)
            if not np.isfinite(lim) or lim <= 0.0:
                raise ValueError(f"limit must be > 0, got {self.limit!r}")
            self.limit = lim
        self.reset()

    def reset(self) -> None:
        """Zero the command state and the latency queue."""
        self._state = np.zeros(self.n_commands, dtype=np.float64)
        self._queue = [np.zeros(self.n_commands) for _ in range(self.latency - 1)]

    @property
    def state(self) -> NDArray[np.float64]:
        """Current command vector (a copy)."""
        return self._state.copy()

    def step(self, error: NDArray[np.float64]) -> NDArray[np.float64]:
        """Push a new error vector, pop the delayed one, and update the command.

        Returns the command that should be applied for the *next* frame.
        """
        e = np.asarray(error, dtype=np.float64)
        if e.shape != (self.n_commands,):
            raise ValueError(f"error must have shape {(self.n_commands,)}, got {e.shape}")
        self._queue.append(e.copy())
        delayed = self._queue.pop(0)
        self._state = (1.0 - self.leak) * self._state + self.gain * delayed
        if self.limit is not None:
            np.clip(self._state, -self.limit, self.limit, out=self._state)
        return self._state.copy()
