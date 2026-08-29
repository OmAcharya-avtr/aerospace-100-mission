"""Closed-loop control: leaky integrator, latency, and transfer functions.

Loop model
----------
Time is discretised at the WFS frame rate ``f_s = 1 / T``.  With a total loop
latency of ``d >= 1`` frames (one frame is the minimum: the measurement of
frame ``k`` cannot act before frame ``k+1``), the integrator is

    c_k = leak * c_{k-1} + g * R * s_{k-d}                                  (1)

where ``c`` is the DM command vector, ``s`` the measured slope vector, ``R``
the reconstructor and ``g`` the loop gain.  ``leak = 1`` is a pure integrator.

Writing the loop in the scalar (per-mode) domain with a perfect reconstructor,
the residual ``e = phi - c`` obeys

    E(z) / Phi(z) = (1 - leak z^-1) / [ (1 - leak z^-1) + g z^-d ]          (2)

the **error rejection transfer function**, and the measurement noise ``n``
propagates to the residual as

    E(z) / N(z) = - g z^-d / [ (1 - leak z^-1) + g z^-d ]                   (3)

Multiplying (2) by ``z^d`` gives the characteristic polynomial

    z^d - leak z^(d-1) + g = 0                                              (4)

so the loop is stable iff all its roots lie inside the unit circle.  For a pure
integrator this yields the classical limits ``g < 2`` at ``d = 1``, ``g < 1``
at ``d = 2`` and ``g < 2 sin(pi/10) = 0.6180`` at ``d = 3``.

Sources: P.-Y. Madec, "Control techniques", in *Adaptive Optics in Astronomy*,
ed. F. Roddier, Cambridge University Press (1999), Ch. 3, Eqs. 3.8-3.20;
Hardy (1998), Sec. 7.3.  The ``g/(2-g)`` noise-variance amplification of a
one-frame-delay integrator is Madec's Eq. 3.20.

Units: ``gain`` and ``leak`` are dimensionless, ``delay_frames`` is an integer
number of WFS frames, frequencies are in Hz and the frame rate in Hz.
Validity: the model assumes a linear, time-invariant, single-input loop; a real
system's WFS integration adds a further ``sinc`` roll-off which is not included
(documented in the README Limitations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Integrator",
    "noise_transfer",
    "noise_variance_gain",
    "rejection_transfer",
    "stability_limit_gain",
]


def _validate(gain: float, delay_frames: int, leak: float) -> None:
    if not np.isfinite(gain) or gain <= 0.0:
        raise ValueError(f"gain must be finite and > 0, got {gain!r}")
    if int(delay_frames) != delay_frames or delay_frames < 1:
        raise ValueError(f"delay_frames must be an integer >= 1, got {delay_frames!r}")
    if not (0.0 < leak <= 1.0):
        raise ValueError(f"leak must lie in (0, 1], got {leak!r}")


def rejection_transfer(
    frequency_hz: np.ndarray | float,
    frame_rate_hz: float,
    gain: float,
    delay_frames: int = 2,
    leak: float = 1.0,
) -> np.ndarray:
    """Complex error rejection transfer function, Eq. (2).

    Parameters
    ----------
    frequency_hz:
        Temporal frequency [Hz]; values above the Nyquist ``f_s / 2`` alias and
        are not meaningful.
    frame_rate_hz:
        WFS sampling rate ``f_s`` [Hz], ``> 0``.
    gain, delay_frames, leak:
        Loop parameters as in Eq. (1).

    Returns
    -------
    numpy.ndarray
        ``E/Phi`` (complex, dimensionless).  ``|E/Phi| < 1`` is rejection,
        ``> 1`` is the unavoidable gain peaking above the loop bandwidth.
    """
    _validate(gain, delay_frames, leak)
    if not np.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError(f"frame_rate_hz must be finite and > 0, got {frame_rate_hz!r}")
    f = np.asarray(frequency_hz, dtype=float)
    z = np.exp(2j * np.pi * f / frame_rate_hz)
    num = 1.0 - leak / z
    return num / (num + gain * z ** (-int(delay_frames)))


def noise_transfer(
    frequency_hz: np.ndarray | float,
    frame_rate_hz: float,
    gain: float,
    delay_frames: int = 2,
    leak: float = 1.0,
) -> np.ndarray:
    """Complex noise transfer function ``E/N``, Eq. (3) (dimensionless)."""
    _validate(gain, delay_frames, leak)
    if not np.isfinite(frame_rate_hz) or frame_rate_hz <= 0.0:
        raise ValueError(f"frame_rate_hz must be finite and > 0, got {frame_rate_hz!r}")
    f = np.asarray(frequency_hz, dtype=float)
    z = np.exp(2j * np.pi * f / frame_rate_hz)
    num = 1.0 - leak / z
    return -gain * z ** (-int(delay_frames)) / (num + gain * z ** (-int(delay_frames)))


def stability_limit_gain(delay_frames: int = 2, leak: float = 1.0, tol: float = 1e-9) -> float:
    """Largest stable loop gain for Eq. (4), found by bisection on the roots.

    Returns the gain at which the largest closed-loop pole modulus reaches 1.
    Known analytic values for a pure integrator: ``2`` (``d = 1``), ``1``
    (``d = 2``), ``2 sin(pi/10) = 0.618034`` (``d = 3``).
    """
    if int(delay_frames) != delay_frames or delay_frames < 1:
        raise ValueError(f"delay_frames must be an integer >= 1, got {delay_frames!r}")
    if not (0.0 < leak <= 1.0):
        raise ValueError(f"leak must lie in (0, 1], got {leak!r}")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and > 0, got {tol!r}")
    d = int(delay_frames)

    def max_pole(g: float) -> float:
        coeffs = np.zeros(d + 1)
        coeffs[0] = 1.0
        coeffs[1] = -leak
        coeffs[-1] += g
        return float(np.max(np.abs(np.roots(coeffs))))

    lo, hi = 1e-6, 4.0
    if max_pole(lo) >= 1.0:  # pragma: no cover - only for pathological leak
        return 0.0
    while max_pole(hi) < 1.0:  # pragma: no cover - defensive
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("no stability limit found below gain 1e6")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if max_pole(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def noise_variance_gain(
    gain: float,
    delay_frames: int = 2,
    leak: float = 1.0,
    n_terms: int = 8192,
) -> float:
    """Variance amplification of white measurement noise into the residual.

    ``sum_k h_k^2`` where ``h`` is the impulse response of Eq. (3).  For
    ``d = 1``, ``leak = 1`` this equals the classical ``g / (2 - g)``
    (Madec 1999, Eq. 3.20).  Returns ``inf`` for an unstable loop.
    """
    _validate(gain, delay_frames, leak)
    if int(n_terms) != n_terms or n_terms < 16:
        raise ValueError(f"n_terms must be an integer >= 16, got {n_terms!r}")
    if gain >= stability_limit_gain(delay_frames, leak):
        return float("inf")
    d = int(delay_frames)
    n = int(n_terms)
    # h_k from the recursion e_k = leak e_{k-1} - g n_{k-d} + ... expanded as a
    # direct-form filter: E(z)/N(z) = -g z^-d / (1 - leak z^-1 + g z^-d)
    h = np.zeros(n)
    for k in range(n):
        acc = -gain if k == d else 0.0
        if k >= 1:
            acc += leak * h[k - 1]
        if k >= d:
            acc -= gain * h[k - d]
        h[k] = acc
    return float(np.sum(h**2))


@dataclass
class Integrator:
    """Discrete leaky integrator with an explicit latency buffer.

    Parameters
    ----------
    n_commands:
        Length of the command vector, ``>= 1``.
    gain:
        Loop gain ``g``, ``> 0``.
    delay_frames:
        Total loop latency ``d`` in frames, ``>= 1``, exactly the ``d`` of
        Eq. (1)-(4).  One frame of it is the DM application delay that is
        inherent to a discrete loop (the command computed at frame ``k`` shapes
        the mirror for frame ``k+1``), so the internal measurement buffer holds
        ``d - 1`` frames.  ``d = 1`` is therefore the fastest realisable loop.
    leak:
        Leak factor in ``(0, 1]``; ``1`` is a pure integrator.
    command_limit:
        Optional symmetric saturation applied to the accumulated command.

    Notes
    -----
    :meth:`step` takes the *reconstructed increment* ``R s`` (already in
    command units), pushes it through the latency buffer and returns the new
    command vector.
    """

    n_commands: int
    gain: float = 0.4
    delay_frames: int = 2
    leak: float = 1.0
    command_limit: float = float("inf")
    _buffer: list[np.ndarray] = field(init=False, repr=False)
    _command: np.ndarray = field(init=False, repr=False)
    _saturated: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        if int(self.n_commands) != self.n_commands or self.n_commands < 1:
            raise ValueError(f"n_commands must be an integer >= 1, got {self.n_commands!r}")
        _validate(self.gain, self.delay_frames, self.leak)
        if np.isnan(self.command_limit) or self.command_limit <= 0.0:
            raise ValueError(f"command_limit must be > 0, got {self.command_limit!r}")
        self.reset()

    def reset(self) -> None:
        """Zero the command and the latency buffer."""
        self._command = np.zeros(int(self.n_commands))
        self._buffer = [np.zeros(int(self.n_commands)) for _ in range(int(self.delay_frames) - 1)]
        self._saturated = 0.0

    @property
    def command(self) -> np.ndarray:
        """Current command vector (a copy is not made; do not mutate)."""
        return self._command

    @property
    def last_saturated_fraction(self) -> float:
        """Fraction of commands clipped by ``command_limit`` on the last step."""
        return self._saturated

    @property
    def stability_limit(self) -> float:
        """Largest stable gain for this latency and leak."""
        return stability_limit_gain(self.delay_frames, self.leak)

    @property
    def is_stable(self) -> bool:
        """Whether ``gain`` is below the analytic stability limit."""
        return self.gain < self.stability_limit

    def step(self, increment: np.ndarray) -> np.ndarray:
        """Advance one frame with a new reconstructed increment.

        ``increment`` is ``R @ s`` for the frame just measured.  The value
        actually accumulated is the one that entered the buffer
        ``delay_frames - 1`` frames ago; together with the one-frame DM
        application delay this realises the total latency ``d``.  The returned
        command is the mirror shape for the *next* frame.
        """
        increment = np.asarray(increment, dtype=float)
        if increment.shape != (int(self.n_commands),):
            raise ValueError(
                f"increment must have shape ({self.n_commands},), got {increment.shape}"
            )
        self._buffer.append(increment.copy())
        delayed = self._buffer.pop(0)
        new = self.leak * self._command + self.gain * delayed
        if np.isinf(self.command_limit):
            self._saturated = 0.0
        else:
            self._saturated = float(np.count_nonzero(np.abs(new) > self.command_limit)) / new.size
            new = np.clip(new, -self.command_limit, self.command_limit)
        self._command = new
        return self._command
