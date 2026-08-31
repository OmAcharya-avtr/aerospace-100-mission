"""Observable features for the learned gain scheduler.

Everything here is computed from a trailing window of **magnetometer samples
and the controller's own saturation flags**, plus two numbers the flight
software already knows about its own vehicle (the magnetorquer dipole limit
and the nominal inertia).  Nothing uses the true body rate, the true attitude,
or any quantity the simulator knows but a spacecraft would not.

The rate proxy
--------------
For a body rotating fast compared with the orbital rate of change of the
inertial field, the body-frame field derivative is ``dB/dt = -omega x B``, so

    |dB/dt| / |B| = |omega_perp|                             [rad/s]

which is a **lower bound** on ``|omega|`` and equals it only when the rate is
perpendicular to the field.  It is also the quantity B-dot itself acts on, so
it is the natural observable to schedule against.  Near the end of a detumble
it saturates at roughly the orbital rate, because the inertial field direction
is then rotating as fast as the body is: see README "Limitations".

Feature vector (8 entries, in this order)
-----------------------------------------
0. ``log10(rate_proxy)``, median of ``|dB/dt|/|B|`` over the window [log10 rad/s]
1. ``log10(mean |B|)`` over the window [log10 T]
2. rate trend: least-squares slope of ``log10(rate_proxy_i)`` against time,
   per 1000 s
3. saturation duty: fraction of window steps whose commanded dipole was clipped
4. field variability: ``std(|B|) / mean(|B|)`` over the window, dimensionless
5. ``log10(max dipole per axis)`` [log10 A m^2] - known hardware
6. ``log10(nominal inertia scale)`` [log10 kg m^2] - known from CAD
7. ``log10(1 + elapsed seconds since control start)`` [log10 s]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

#: Number of entries in a feature vector.
N_FEATURES: int = 8

#: Names of the feature vector entries, in order.
FEATURE_NAMES: tuple[str, ...] = (
    "log10_rate_proxy",
    "log10_mean_field_t",
    "rate_trend_per_1000s",
    "saturation_duty",
    "field_variability",
    "log10_max_dipole",
    "log10_inertia_scale",
    "log10_elapsed_s",
)

#: Floor applied to the rate proxy before taking a logarithm [rad/s].
RATE_PROXY_FLOOR_RAD_S: float = 1e-7


def rate_proxy(b_t: ArrayLike, b_dot_t_s: ArrayLike) -> float:
    """``|dB/dt| / |B|`` [rad/s]: the observable stand-in for ``|omega_perp|``.

    Returns 0.0 when the field magnitude is zero (never the case in orbit).
    """
    b = np.asarray(b_t, dtype=float)
    bd = np.asarray(b_dot_t_s, dtype=float)
    if b.shape != (3,) or bd.shape != (3,):
        raise ValueError("b_t and b_dot_t_s must have shape (3,)")
    bn = float(np.linalg.norm(b))
    if bn <= 0.0:
        return 0.0
    return float(np.linalg.norm(bd) / bn)


@dataclass
class TelemetryWindow:
    """Fixed-length trailing buffer of magnetometer-derived quantities.

    Parameters
    ----------
    length : int
        Number of control steps retained; must be >= 2.
    """

    length: int = 60
    _rate: deque = field(init=False, repr=False, default_factory=deque)
    _bmag: deque = field(init=False, repr=False, default_factory=deque)
    _sat: deque = field(init=False, repr=False, default_factory=deque)
    _time: deque = field(init=False, repr=False, default_factory=deque)

    def __post_init__(self) -> None:
        if int(self.length) < 2:
            raise ValueError(f"length must be >= 2, got {self.length}")
        self.length = int(self.length)
        for d in (self._rate, self._bmag, self._sat, self._time):
            d.clear()

    def __len__(self) -> int:
        return len(self._rate)

    def push(
        self, t_s: float, b_t: ArrayLike, b_dot_t_s: ArrayLike, saturated: bool
    ) -> None:
        """Append one control-step sample."""
        b = np.asarray(b_t, dtype=float)
        self._rate.append(rate_proxy(b, b_dot_t_s))
        self._bmag.append(float(np.linalg.norm(b)))
        self._sat.append(bool(saturated))
        self._time.append(float(t_s))
        while len(self._rate) > self.length:
            self._rate.popleft()
            self._bmag.popleft()
            self._sat.popleft()
            self._time.popleft()

    def ready(self) -> bool:
        """True once the window holds at least two samples."""
        return len(self._rate) >= 2

    def features(
        self, max_dipole_am2: float, inertia_scale_kgm2: float
    ) -> NDArray[np.float64]:
        """Build the 8-entry feature vector; see the module docstring.

        Raises
        ------
        ValueError
            If the window is not yet ready, or the hardware parameters are
            not positive.
        """
        if not self.ready():
            raise ValueError("window needs at least two samples before features()")
        if not np.isfinite(max_dipole_am2) or max_dipole_am2 <= 0.0:
            raise ValueError(f"max_dipole_am2 must be positive, got {max_dipole_am2}")
        if not np.isfinite(inertia_scale_kgm2) or inertia_scale_kgm2 <= 0.0:
            raise ValueError(
                f"inertia_scale_kgm2 must be positive, got {inertia_scale_kgm2}"
            )
        rate = np.maximum(np.asarray(self._rate, dtype=float), RATE_PROXY_FLOOR_RAD_S)
        bmag = np.asarray(self._bmag, dtype=float)
        t = np.asarray(self._time, dtype=float)
        log_rate = np.log10(rate)
        med = float(np.median(rate))
        span = float(t[-1] - t[0])
        if span > 0.0:
            trend = float(np.polyfit(t - t[0], log_rate, 1)[0]) * 1000.0
        else:
            trend = 0.0
        mean_b = float(np.mean(bmag))
        var_b = float(np.std(bmag) / mean_b) if mean_b > 0.0 else 0.0
        return np.array(
            [
                float(np.log10(max(med, RATE_PROXY_FLOOR_RAD_S))),
                float(np.log10(max(mean_b, 1e-12))),
                trend,
                float(np.mean(np.asarray(self._sat, dtype=float))),
                var_b,
                float(np.log10(max_dipole_am2)),
                float(np.log10(inertia_scale_kgm2)),
                float(np.log10(1.0 + max(t[-1], 0.0))),
            ],
            dtype=float,
        )
